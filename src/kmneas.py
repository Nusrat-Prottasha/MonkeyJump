# kmeans_init_sentence_router.py
# K-means++ initialization for MonkeyJump-LoRA / MoE-LoRA router centers
# Supports both TOKEN-BASED and SEQUENCE-BASED routing modes

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# -----------------------------
# Utilities
# -----------------------------

@torch.no_grad()
def _named_module_map(model: nn.Module) -> Dict[nn.Module, str]:
    """Return {module_obj: fully.qualified.name} for logging."""
    return {m: name for name, m in model.named_modules()}


@torch.no_grad()
def _to_model_inputs(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Keep only model-consumable fields."""
    drop_keys = {"labels", "answer", "label_texts"}
    passthrough_nontensors = {
        "image_sizes", "image_sizes_videos",
        "image_grid_thw", "image_grid_thw_list",
        "pixel_values_videos",
        "modalities", "media_types",
    }
    out = {}
    for k, v in batch.items():
        if k in drop_keys:
            continue
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(x) for x in v):
            out[k] = type(v)(x.to(device, non_blocking=True) for x in v)
        elif k in passthrough_nontensors:
            out[k] = v
    return out


def _coerce_token_mask(
    attention_mask: Optional[torch.Tensor],
    B: int,
    T: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Convert various attention_mask shapes to (B, T)."""
    if attention_mask is None:
        return torch.ones(B, T, dtype=dtype, device=device)

    m = attention_mask.to(device=device)

    if m.dim() == 2 and m.shape == (B, T):
        pass
    elif m.dim() > 2:
        if m.shape[-1] == T:
            m = m.view(B, -1, T)
            if torch.is_floating_point(m):
                valid = m > -1e4
            else:
                valid = m != 0
            m = valid.any(dim=1)
        elif m.shape[-2:] == (T, T):
            if torch.is_floating_point(m):
                valid = m > -1e4
            else:
                valid = m != 0
            m = valid.any(dim=-1)
            if m.dim() > 2:
                m = m.view(B, T)
        else:
            m = torch.ones(B, T, dtype=dtype, device=device)
    else:
        m = torch.ones(B, T, dtype=dtype, device=device)

    if dtype == torch.bool:
        if m.dtype != torch.bool:
            if torch.is_floating_point(m):
                m = m > 0
            else:
                m = m != 0
    else:
        m = m.to(dtype)

    return m


def _sentence_rep_masked(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    prompt_lengths: Optional[torch.Tensor] = None,
    mode: str = "last",
) -> torch.Tensor:
    """
    Compute sentence representation -> (B, H).
    
    Modes:
    - "last": last valid token
    - "mean": mean of all valid tokens
    - "prompt_end": token at prompt_lengths position
    """
    if hidden_states.dim() == 2:
        return hidden_states

    B, T, H = hidden_states.shape
    device = hidden_states.device

    if mode == "prompt_end" and prompt_lengths is not None:
        # Use prompt_lengths position
        batch_idx = torch.arange(B, device=device)
        return hidden_states[batch_idx, prompt_lengths]

    if mode == "last":
        if attention_mask is None:
            return hidden_states[:, -1, :]

        mask = _coerce_token_mask(attention_mask, B, T, device, torch.bool)
        seq_lengths = mask.sum(dim=1).clamp(min=1) - 1
        indices = seq_lengths.view(B, 1, 1).expand(B, 1, H).long()
        return hidden_states.gather(dim=1, index=indices).squeeze(1)

    else:  # mean
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = _coerce_token_mask(attention_mask, B, T, device, hidden_states.dtype)
        mask = mask.view(B, T, 1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom


def _token_reps_masked(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_tokens_per_batch: int = 1024,
) -> torch.Tensor:
    """
    Extract valid token representations -> (N, H) where N <= B*T.
    
    Args:
        hidden_states: [B, T, H]
        attention_mask: [B, T] or various shapes
        max_tokens_per_batch: Maximum tokens to return per batch (for memory)
        
    Returns:
        [N, H] - flattened valid token representations
    """
    if hidden_states.dim() == 2:
        return hidden_states

    B, T, H = hidden_states.shape
    device = hidden_states.device

    # Get mask
    mask = _coerce_token_mask(attention_mask, B, T, device, torch.bool)  # [B, T]
    
    # Flatten and filter
    flat_hidden = hidden_states.reshape(-1, H)  # [B*T, H]
    flat_mask = mask.reshape(-1)  # [B*T]
    
    # Get valid tokens
    valid_tokens = flat_hidden[flat_mask]  # [N, H]
    
    # Subsample if too many tokens (for memory efficiency)
    if valid_tokens.size(0) > max_tokens_per_batch:
        idx = torch.randperm(valid_tokens.size(0), device=device)[:max_tokens_per_batch]
        valid_tokens = valid_tokens[idx]
    
    return valid_tokens


def _is_patched_block(m: nn.Module) -> bool:
    """Check if a module is a patched MoE-LoRA / MonkeyJump block."""
    # Support both old (_moe_*) and new (_mj_*) attribute names
    return (
        hasattr(m, "_sentence_router") or
        hasattr(m, "_moe_lora_forward_patched") or
        hasattr(m, "_moe_router") or
        hasattr(m, "_mj_router") or
        hasattr(m, "_mj_patched")
    )


def _get_router(block: nn.Module) -> Optional[nn.Module]:
    """Get router from a patched block, supporting both old and new names."""
    router = getattr(block, "_sentence_router", None)
    if router is None:
        router = getattr(block, "_moe_router", None)
    if router is None:
        router = getattr(block, "_mj_router", None)
    return router


def _get_block_config(block: nn.Module) -> Optional[Dict]:
    """Get config from a patched block, supporting both old and new names."""
    config = getattr(block, "_moe_config", None)
    if config is None:
        config = getattr(block, "_mj_config", None)
    return config


# -----------------------------
# K-means++ with Cosine Similarity
# -----------------------------

@torch.no_grad()
def kmeans_pp_cosine(
    X: torch.Tensor,
    k: int,
    iters: int = 15,
    seed: int = 42,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cosine k-means++ on L2-normalized X [N, H]. Returns centers [k, H]."""
    N, H = X.shape
    device = X.device
    gen = torch.Generator(device=device).manual_seed(seed)

    X = F.normalize(X, dim=-1, eps=eps)

    if N < k:
        centers = X.clone()
        if N < k:
            noise = torch.randn(k - N, H, device=device, generator=gen)
            noise = F.normalize(noise, dim=-1, eps=eps)
            centers = torch.cat([centers, noise], dim=0)
        return centers

    # K-means++ initialization
    idx0 = torch.randint(0, N, (1,), generator=gen, device=device)
    centers = [X[idx0[0]]]

    d = (1.0 - (X @ centers[0]).clamp(-1.0, 1.0)).clamp_min(0.0)

    for _ in range(1, k):
        s = d.sum()
        if not torch.isfinite(s) or s <= eps:
            sims = torch.stack([X @ c for c in centers], dim=1)
            d_fps = (1.0 - sims.max(dim=1).values.clamp(-1.0, 1.0)).clamp_min(0.0)
            idx = d_fps.argmax()
        else:
            probs = d / s
            idx = torch.multinomial(probs, 1, generator=gen)[0]

        centers.append(X[idx])
        sim_new = (X @ centers[-1]).clamp(-1.0, 1.0)
        d = torch.minimum(d, (1.0 - sim_new).clamp_min(0.0))

    C = torch.stack(centers, dim=0)
    C = F.normalize(C, dim=-1, eps=eps)

    # Lloyd's iterations
    for _ in range(iters):
        sims = X @ C.t()
        assign = sims.argmax(dim=1)

        new_centers = []
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                new_centers.append(F.normalize(X[mask].mean(dim=0), dim=-1, eps=eps))
            else:
                new_centers.append(C[j])

        C = torch.stack(new_centers, dim=0)

    return C


# -----------------------------
# Representation Collector (Unified)
# -----------------------------

@torch.no_grad()
def collect_representations(
    model: nn.Module,
    dataloader: DataLoader,
    max_batches: int = 200,
    per_block_cap: int = 10000,
    warmup_batches: int = 4,
    idle_patience: int = 6,
    use_autocast: bool = True,
    verbose: bool = True,
    rep_mode: str = "last",
    max_tokens_per_batch: int = 1024,
) -> Dict[nn.Module, torch.Tensor]:
    """
    Collect representations for each patched block.
    
    Supports both:
    - Sequence-based (rep_mode="last", "mean", or "prompt_end"): collects sentence-level reps [B, H]
    - Token-based (rep_mode="token"): collects all valid token reps [N, H]
    
    Args:
        model: The model with patched blocks
        dataloader: DataLoader for collecting representations
        max_batches: Maximum number of batches to process
        per_block_cap: Maximum samples per block
        warmup_batches: Number of warmup batches (unused, kept for compatibility)
        idle_patience: Stop after this many consecutive failed batches
        use_autocast: Whether to use autocast for forward passes
        verbose: Print progress
        rep_mode: "last", "mean", "prompt_end", or "token"
        max_tokens_per_batch: For token mode, max tokens to keep per batch
        
    Returns:
        Dict mapping block -> collected representations tensor
    """
    was_training = model.training
    model.eval()

    is_token_mode = (rep_mode == "token")

    # Find all patched blocks (support both old and new attribute names)
    patched_blocks: List[nn.Module] = []
    for m in model.modules():
        if _is_patched_block(m):
            patched_blocks.append(m)

    if not patched_blocks:
        if verbose:
            print("[kmeans-init] No patched blocks found!")
            print("[kmeans-init] Looking for: _moe_router, _mj_router, _sentence_router")
        return {}

    if verbose:
        mode_str = "TOKEN-BASED" if is_token_mode else f"SEQUENCE-BASED ({rep_mode})"
        print(f"[kmeans-init] Found {len(patched_blocks)} patched blocks, mode: {mode_str}")

    buckets: Dict[nn.Module, List[torch.Tensor]] = {b: [] for b in patched_blocks}
    hook_calls: Dict[nn.Module, int] = {b: 0 for b in patched_blocks}
    hooks: List[Any] = []

    current_attention_mask = [None]
    current_prompt_lengths = [None]

    def _make_hook(block: nn.Module):
        def hook_fn(module, args, kwargs):
            hook_calls[block] += 1

            hidden = None
            if len(args) > 0 and torch.is_tensor(args[0]):
                hidden = args[0]
            elif "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                hidden = kwargs["hidden_states"]

            if hidden is None or hidden.dim() < 2:
                return

            # Detach and convert to float32 to avoid autocast issues
            hidden = hidden.detach().float()
            
            attn_mask = current_attention_mask[0]
            prompt_lengths = current_prompt_lengths[0]
            
            if is_token_mode:
                # Token-based: collect all valid tokens
                reps = _token_reps_masked(hidden, attn_mask, max_tokens_per_batch)
            else:
                # Sequence-based: collect sentence-level reps
                reps = _sentence_rep_masked(hidden, attn_mask, prompt_lengths, mode=rep_mode)

            reps = F.normalize(reps.float(), dim=-1)
            buckets[block].append(reps.cpu())

        return hook_fn

    for block in patched_blocks:
        hook = block.register_forward_pre_hook(_make_hook(block), with_kwargs=True)
        hooks.append(hook)

    it = iter(dataloader)
    idle = 0
    batches_seen = 0

    pbar = tqdm(range(max_batches), desc="Collecting representations", disable=not verbose)
    for _ in pbar:
        try:
            raw_batch = next(it)
        except StopIteration:
            break

        dev = next(model.parameters()).device

        current_attention_mask[0] = raw_batch.get("attention_mask", None)
        current_prompt_lengths[0] = raw_batch.get("prompt_lengths", None)

        batch = _to_model_inputs(raw_batch, dev)
        if not batch:
            idle += 1
            if idle >= idle_patience:
                break
            continue

        try:
            with torch.no_grad():
                if use_autocast and torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        _ = model(**batch)
                else:
                    _ = model(**batch)
        except Exception as e:
            if verbose:
                print(f"[warn] batch error: {str(e)[:120]}")
            continue

        batches_seen += 1

        total_collected = sum(sum(x.size(0) for x in lst) for lst in buckets.values())
        pbar.set_postfix({"samples": total_collected})

        all_done = all(
            sum(x.size(0) for x in lst) >= per_block_cap
            for lst in buckets.values()
        )
        if all_done:
            break

        idle = 0

    for h in hooks:
        h.remove()

    if was_training:
        model.train()

    current_attention_mask[0] = None
    current_prompt_lengths[0] = None

    result: Dict[nn.Module, torch.Tensor] = {}
    for block, lst in buckets.items():
        if not lst:
            continue
        X = torch.cat(lst, dim=0)
        if X.size(0) > per_block_cap:
            idx = torch.randperm(X.size(0))[:per_block_cap]
            X = X[idx]
        result[block] = X

        if verbose:
            unit = "tokens" if is_token_mode else "sentences"
            print(f"[kmeans-init] Block collected {X.size(0)} {unit}, dim={X.size(1)}")

    return result


# Keep old function name for backward compatibility
def collect_sentence_reps(
    model: nn.Module,
    dataloader: DataLoader,
    max_batches: int = 200,
    per_block_cap: int = 10000,
    warmup_batches: int = 4,
    idle_patience: int = 6,
    use_autocast: bool = True,
    verbose: bool = True,
    rep_mode: str = "last",
) -> Dict[nn.Module, torch.Tensor]:
    """Backward-compatible wrapper for collect_representations."""
    return collect_representations(
        model=model,
        dataloader=dataloader,
        max_batches=max_batches,
        per_block_cap=per_block_cap,
        warmup_batches=warmup_batches,
        idle_patience=idle_patience,
        use_autocast=use_autocast,
        verbose=verbose,
        rep_mode=rep_mode,
    )


# -----------------------------
# Auto Expert Selection
# -----------------------------

def compute_optimal_num_experts(
    X: torch.Tensor,
    max_experts: int,
    sim_lo: float = 0.70,
    sim_hi: float = 0.98,
    min_cluster_frac: float = 0.10,
    max_center_cos: float = 0.95,
    iters: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[int, torch.Tensor, float]:
    """Automatically determine optimal number of experts based on data diversity."""
    N, H = X.shape
    device = X.device

    C_full = kmeans_pp_cosine(X, k=max_experts, iters=iters, seed=seed)

    sims = X @ C_full.t()
    max_sim, assign = sims.max(dim=1)
    avg_sim = max_sim.mean().item()

    if sim_hi <= sim_lo:
        sim_hi = sim_lo + 1e-6

    r = (sim_hi - avg_sim) / (sim_hi - sim_lo)
    r = max(0.0, min(1.0, r))
    E_star = int(math.ceil(1.0 + r * (max_experts - 1)))
    E_star = max(1, min(E_star, max_experts))

    def is_valid_E(E: int) -> bool:
        if E == 1:
            return True

        sizes = torch.stack([(assign == j).sum() for j in range(E)]).float()

        if (sizes == 0).any():
            return False

        if sizes.min().item() / (sizes.mean().item() + 1e-9) < min_cluster_frac:
            return False

        cc = C_full[:E] @ C_full[:E].t()
        cc.fill_diagonal_(0)
        if cc.max().item() > max_center_cos:
            return False

        return True

    while E_star < max_experts and not is_valid_E(E_star):
        E_star += 1

    while E_star > 1 and not is_valid_E(E_star):
        E_star -= 1

    if verbose:
        print(f"[auto-expert] avg_sim={avg_sim:.3f} -> E*={E_star}/{max_experts}")

    return E_star, C_full, avg_sim


# -----------------------------
# Detect Routing Mode
# -----------------------------

def _detect_routing_mode(model: nn.Module) -> str:
    """
    Auto-detect routing mode from model configuration.
    
    Looks for config on patched blocks to determine if
    token-based or sequence-based routing is used.
    
    Returns:
        "token", "last", "mean", or "prompt_end" (default: "last")
    """
    for m in model.modules():
        # Check for config (both old and new attribute names)
        config = _get_block_config(m)
        if config is not None and "rep_mode" in config:
            return config["rep_mode"]
        
        # Check block repr for hints
        if hasattr(m, "_moe_repr_patched") and m._moe_repr_patched:
            try:
                repr_str = m.__repr__()
                if "rep_mode: token" in repr_str:
                    return "token"
                elif "rep_mode: prompt_end" in repr_str:
                    return "prompt_end"
                elif "rep_mode: mean" in repr_str:
                    return "mean"
                elif "rep_mode: last" in repr_str:
                    return "last"
            except Exception:
                pass

        # Check MonkeyJump repr
        if hasattr(m, "_mj_repr_patched") and m._mj_repr_patched:
            try:
                repr_str = m.__repr__()
                if "rep_mode: token" in repr_str:
                    return "token"
                elif "rep_mode: prompt_end" in repr_str:
                    return "prompt_end"
                elif "rep_mode: mean" in repr_str:
                    return "mean"
                elif "rep_mode: last" in repr_str:
                    return "last"
            except Exception:
                pass
    
    # Default to sequence-based (last token)
    return "last"


# -----------------------------
# Main Initialization Function
# -----------------------------

@torch.no_grad()
def init_router_centers(
    trainer,
    subset_size: int = 2000,
    loader_batch_size: int = 8,
    collect_batches: int = 200,
    per_block_cap: int = 10000,
    kmeans_iters: int = 15,
    seed: int = 42,
    verbose: bool = True,
    rep_mode: Optional[str] = None,
    max_tokens_per_batch: int = 1024,
    # Auto-expert selection parameters
    auto_select_experts: bool = False,
    sim_lo: float = 0.70,
    sim_hi: float = 0.98,
    min_cluster_frac: float = 0.10,
    max_center_cos: float = 0.95,
):
    """
    Initialize router centers using k-means++ on actual data.
    
    Supports both TOKEN-BASED and SEQUENCE-BASED routing modes.
    Compatible with both MoE-LoRA (_moe_*) and MonkeyJump-LoRA (_mj_*).

    Args:
        trainer: HuggingFace Trainer with model and train_dataset
        subset_size: Number of samples to use for initialization
        loader_batch_size: Batch size for collection
        collect_batches: Maximum batches to process
        per_block_cap: Maximum samples per block
        kmeans_iters: Lloyd's algorithm iterations
        seed: Random seed
        verbose: Print progress
        rep_mode: "last", "mean", "prompt_end", or "token". If None, auto-detect.
        max_tokens_per_batch: For token mode, max tokens per batch (memory control)
        auto_select_experts: If True, automatically determine optimal E per block
        sim_lo, sim_hi: Similarity thresholds for auto-expert selection
        min_cluster_frac: Minimum cluster size fraction
        max_center_cos: Maximum cosine similarity between centers
    """
    ds_full = trainer.train_dataset
    actual_subset = min(subset_size, len(ds_full))

    if hasattr(ds_full, "select"):
        idx = torch.randperm(len(ds_full))[:actual_subset].tolist()
        ds_sub = ds_full.select(idx)
    else:
        ds_sub = ds_full

    dl = DataLoader(
        ds_sub,
        batch_size=loader_batch_size,
        shuffle=True,
        collate_fn=trainer.data_collator,
        drop_last=False,
    )

    acc = getattr(trainer, "accelerator", None)
    model = acc.unwrap_model(trainer.model) if acc is not None else trainer.model

    name_map = _named_module_map(model)

    # Auto-detect rep_mode if not specified
    if rep_mode is None:
        rep_mode = _detect_routing_mode(model)
        if verbose:
            print(f"[kmeans-init] Auto-detected routing mode: {rep_mode}")

    is_token_mode = (rep_mode == "token")

    if verbose:
        print("=" * 60)
        mode_str = "TOKEN-BASED" if is_token_mode else f"SEQUENCE-BASED ({rep_mode})"
        print(f"[kmeans-init] Collecting representations ({mode_str})...")
        print("=" * 60)

    reps_per_block = collect_representations(
        model,
        dl,
        max_batches=collect_batches,
        per_block_cap=per_block_cap,
        verbose=verbose,
        rep_mode=rep_mode,
        max_tokens_per_batch=max_tokens_per_batch,
    )

    if not reps_per_block:
        if verbose:
            print("[kmeans-init] No representations collected!")
        return

    if verbose:
        print("\n" + "=" * 60)
        print("[kmeans-init] Initializing router centers...")
        print("=" * 60)

    summary_rows = []

    for block, X_cpu in reps_per_block.items():
        if X_cpu.numel() == 0:
            continue

        # Get router (supports both old and new attribute names)
        router = _get_router(block)
        
        if router is None:
            if verbose:
                print(f"[warn] Block has no router, skipping")
            continue

        device = router.centers.device
        dtype = router.centers.dtype
        X = F.normalize(X_cpu.to(device=device, dtype=torch.float32), dim=-1)

        num_experts = router.num_experts
        block_name = name_map.get(block, block.__class__.__name__)

        if auto_select_experts:
            E_star, C_full, avg_sim = compute_optimal_num_experts(
                X,
                max_experts=num_experts,
                sim_lo=sim_lo,
                sim_hi=sim_hi,
                min_cluster_frac=min_cluster_frac,
                max_center_cos=max_center_cos,
                iters=kmeans_iters,
                seed=seed,
                verbose=False,
            )

            C = F.normalize(C_full[:E_star], dim=-1)

            if E_star < num_experts:
                C_padded = torch.zeros(num_experts, C.size(1), device=device, dtype=torch.float32)
                C_padded[:E_star] = C
                C = C_padded

            if not hasattr(router, "active_E"):
                router.register_buffer(
                    "active_E",
                    torch.tensor(E_star, dtype=torch.int32, device=device),
                    persistent=False,
                )
            else:
                router.active_E.fill_(E_star)

            summary_rows.append((block_name, num_experts, E_star, avg_sim))

            if verbose:
                print(f"[auto-expert] {block_name}: E={E_star}/{num_experts} (avg_sim={avg_sim:.3f})")
        else:
            C = kmeans_pp_cosine(X, k=num_experts, iters=kmeans_iters, seed=seed)

            sims = X @ C.t()
            avg_sim = sims.max(dim=1).values.mean().item()

            summary_rows.append((block_name, num_experts, num_experts, avg_sim))

            if verbose:
                print(f"[kmeans-init] {block_name}: E={num_experts} (avg_sim={avg_sim:.3f})")

        router.centers.data.copy_(C.to(dtype=dtype))

    if verbose and summary_rows:
        print("\n" + "=" * 60)
        print(" INITIALIZATION SUMMARY")
        print("=" * 60)
        unit = "tokens" if is_token_mode else "sentences"
        print(f" Mode: {rep_mode.upper()} (clustering {unit})")
        print("-" * 60)
        print(f" {'Block':<40} {'E_init':>8} {'E_used':>8} {'AvgSim':>8}")
        print("-" * 60)
        for name, E_init, E_used, avg_sim in summary_rows:
            tag = "✓" if E_init != E_used else " "
            print(f" {tag}{name:<39} {E_init:>8} {E_used:>8} {avg_sim:>8.3f}")
        print("=" * 60)
        print("[kmeans-init] Done!")


# Alias for backward compatibility
init_router_centers_before_training = init_router_centers


# -----------------------------
# Standalone Initialization (without Trainer)
# -----------------------------

@torch.no_grad()
def init_router_centers_standalone(
    model: nn.Module,
    dataloader: DataLoader,
    collect_batches: int = 200,
    per_block_cap: int = 10000,
    kmeans_iters: int = 15,
    seed: int = 42,
    verbose: bool = True,
    rep_mode: Optional[str] = None,
    max_tokens_per_batch: int = 1024,
    auto_select_experts: bool = False,
    sim_lo: float = 0.70,
    sim_hi: float = 0.98,
    min_cluster_frac: float = 0.10,
    max_center_cos: float = 0.95,
):
    """
    Initialize router centers without a Trainer object.
    
    Useful when you want to initialize centers outside of HuggingFace Trainer.
    Compatible with both MoE-LoRA (_moe_*) and MonkeyJump-LoRA (_mj_*).
    
    Args:
        model: Model with MoE-LoRA or MonkeyJump-LoRA applied
        dataloader: DataLoader providing batches
        collect_batches: Maximum batches to process
        per_block_cap: Maximum samples per block
        kmeans_iters: Lloyd's algorithm iterations
        seed: Random seed
        verbose: Print progress
        rep_mode: "last", "mean", "prompt_end", or "token". If None, auto-detect.
        max_tokens_per_batch: For token mode, max tokens per batch
        auto_select_experts: Auto-determine optimal E per block
        sim_lo, sim_hi: Similarity thresholds for auto-expert selection
        min_cluster_frac: Minimum cluster size fraction
        max_center_cos: Maximum cosine similarity between centers
    """
    name_map = _named_module_map(model)

    # Auto-detect rep_mode if not specified
    if rep_mode is None:
        rep_mode = _detect_routing_mode(model)
        if verbose:
            print(f"[kmeans-init] Auto-detected routing mode: {rep_mode}")

    is_token_mode = (rep_mode == "token")

    if verbose:
        print("=" * 60)
        mode_str = "TOKEN-BASED" if is_token_mode else f"SEQUENCE-BASED ({rep_mode})"
        print(f"[kmeans-init] Collecting representations ({mode_str})...")
        print("=" * 60)

    reps_per_block = collect_representations(
        model,
        dataloader,
        max_batches=collect_batches,
        per_block_cap=per_block_cap,
        verbose=verbose,
        rep_mode=rep_mode,
        max_tokens_per_batch=max_tokens_per_batch,
    )

    if not reps_per_block:
        if verbose:
            print("[kmeans-init] No representations collected!")
        return

    if verbose:
        print("\n" + "=" * 60)
        print("[kmeans-init] Initializing router centers...")
        print("=" * 60)

    summary_rows = []

    for block, X_cpu in reps_per_block.items():
        if X_cpu.numel() == 0:
            continue

        # Get router (supports both old and new attribute names)
        router = _get_router(block)
        
        if router is None:
            if verbose:
                print(f"[warn] Block has no router, skipping")
            continue

        device = router.centers.device
        dtype = router.centers.dtype
        X = F.normalize(X_cpu.to(device=device, dtype=torch.float32), dim=-1)

        num_experts = router.num_experts
        block_name = name_map.get(block, block.__class__.__name__)

        if auto_select_experts:
            E_star, C_full, avg_sim = compute_optimal_num_experts(
                X,
                max_experts=num_experts,
                sim_lo=sim_lo,
                sim_hi=sim_hi,
                min_cluster_frac=min_cluster_frac,
                max_center_cos=max_center_cos,
                iters=kmeans_iters,
                seed=seed,
                verbose=False,
            )

            C = F.normalize(C_full[:E_star], dim=-1)

            if E_star < num_experts:
                C_padded = torch.zeros(num_experts, C.size(1), device=device, dtype=torch.float32)
                C_padded[:E_star] = C
                C = C_padded

            if not hasattr(router, "active_E"):
                router.register_buffer(
                    "active_E",
                    torch.tensor(E_star, dtype=torch.int32, device=device),
                    persistent=False,
                )
            else:
                router.active_E.fill_(E_star)

            summary_rows.append((block_name, num_experts, E_star, avg_sim))

            if verbose:
                print(f"[auto-expert] {block_name}: E={E_star}/{num_experts} (avg_sim={avg_sim:.3f})")
        else:
            C = kmeans_pp_cosine(X, k=num_experts, iters=kmeans_iters, seed=seed)

            sims = X @ C.t()
            avg_sim = sims.max(dim=1).values.mean().item()

            summary_rows.append((block_name, num_experts, num_experts, avg_sim))

            if verbose:
                print(f"[kmeans-init] {block_name}: E={num_experts} (avg_sim={avg_sim:.3f})")

        router.centers.data.copy_(C.to(dtype=dtype))

    if verbose and summary_rows:
        print("\n" + "=" * 60)
        print(" INITIALIZATION SUMMARY")
        print("=" * 60)
        unit = "tokens" if is_token_mode else "sentences"
        print(f" Mode: {rep_mode.upper()} (clustering {unit})")
        print("-" * 60)
        print(f" {'Block':<40} {'E_init':>8} {'E_used':>8} {'AvgSim':>8}")
        print("-" * 60)
        for name, E_init, E_used, avg_sim in summary_rows:
            tag = "✓" if E_init != E_used else " "
            print(f" {tag}{name:<39} {E_init:>8} {E_used:>8} {avg_sim:>8.3f}")
        print("=" * 60)
        print("[kmeans-init] Done!")