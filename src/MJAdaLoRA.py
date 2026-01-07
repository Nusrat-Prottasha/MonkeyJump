# monkeyjump_adalora.py
# MonkeyJump-AdaLoRA: MoE-AdaLoRA with routing + probability-weighted experts + jitter noise
# - Uses SVD-based parameterization (P, Λ, Q) instead of standard LoRA (A, B)
# - Importance-based rank allocation with pruning
# - Each linear is ONE expert (not all experts)
# - LOGICALLY SELECTIVE: only routed tokens/sequences receive non-zero LoRA
# - FIXED: P, Λ, Q kept in float32 for gradient precision in bfloat16 training

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze_module_parameters(mod: nn.Module):
    for p in mod.parameters():
        p.requires_grad = False


# -----------------------------
# AdaLoRA Linear (One Expert per Linear)
# -----------------------------

class MonkeyJumpAdaLoRALinear(nn.Module):
    """
    AdaLoRA adapter - ONE expert per linear layer.

    AdaLoRA uses SVD-based parameterization:
        ΔW = P @ diag(Λ) @ Q  (instead of B @ A in standard LoRA)

    Where:
        - P: [out_features, rank] - left singular vectors
        - Λ: [rank] - singular values (diagonal)
        - Q: [rank, in_features] - right singular vectors

    Importance scoring allows dynamic rank pruning during training.
    
    NOTE: P, Λ, Q are kept in float32 for gradient precision during bfloat16 training.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        expert_id: Optional[int] = None,
        always_on: bool = False,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.linear = linear
        _freeze_module_parameters(self.linear)

        self.rank = rank
        self.alpha = alpha
        self.expert_id = expert_id
        self.always_on = always_on

        H, Din = linear.out_features, linear.in_features
        self.out_features = H
        self.in_features = Din

        base_device = linear.weight.device

        # FIX: Store scaling in float32 for precision
        self.register_buffer(
            "_scaling",
            torch.tensor(alpha / rank, device=base_device, dtype=torch.float32),
        )

        # AdaLoRA SVD parameterization: ΔW = P @ diag(Λ) @ Q
        # P: left singular vectors [out_features, rank]
        # Λ: singular values [rank] (stored as 1D, used as diagonal)
        # Q: right singular vectors [rank, in_features]
        
        # FIX: All parameters in float32 for gradient precision
        P_init = torch.empty(H, rank, device=base_device, dtype=torch.float32)
        Q_init = torch.empty(rank, Din, device=base_device, dtype=torch.float32)
        nn.init.orthogonal_(P_init)
        nn.init.orthogonal_(Q_init)
        
        # FIX: Keep in float32, don't convert to base_dtype
        self.P = nn.Parameter(P_init)
        self.Q = nn.Parameter(Q_init)
        self.Lambda = nn.Parameter(torch.full((rank,), init_scale, device=base_device, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Rank mask for pruning (1 = active, 0 = pruned)
        self.register_buffer("rank_mask", torch.ones(rank, device=base_device, dtype=torch.float32))

        # Importance scores (computed externally, stored here)
        self.register_buffer("importance_scores", torch.zeros(rank, device=base_device, dtype=torch.float32))

        self._top_k_ids: Optional[torch.Tensor] = None
        self._top_k_weights: Optional[torch.Tensor] = None

    def set_routing(self, top_k_ids: Optional[torch.Tensor], top_k_weights: Optional[torch.Tensor]):
        """Set routing weights."""
        self._top_k_ids = top_k_ids
        self._top_k_weights = top_k_weights

    def get_effective_rank(self) -> int:
        """Get current effective rank (non-pruned singular values)."""
        return int(self.rank_mask.sum().item())

    def prune_to_rank(self, target_rank: int):
        """Prune to target rank based on importance scores."""
        if target_rank >= self.rank:
            self.rank_mask.fill_(1.0)
            return

        # Keep top-k most important singular values
        _, indices = torch.topk(self.importance_scores, k=target_rank)
        new_mask = torch.zeros_like(self.rank_mask)
        new_mask[indices] = 1.0
        self.rank_mask.copy_(new_mask)

    def update_importance_scores(self):
        """
        Update importance scores based on gradient sensitivity.
        Call this after backward pass.

        Importance = |Λ| * (|∂L/∂P|_F * |∂L/∂Q|_F + |∂L/∂Λ|)
        """
        if self.P.grad is None or self.Q.grad is None or self.Lambda.grad is None:
            return

        # Compute importance per singular value
        # Following AdaLoRA paper: sensitivity-based importance
        P_grad_norm = self.P.grad.abs().sum(dim=0)  # [rank]
        Q_grad_norm = self.Q.grad.abs().sum(dim=1)  # [rank]
        Lambda_grad = self.Lambda.grad.abs()  # [rank]

        # Combined importance score
        importance = self.Lambda.abs() * (P_grad_norm * Q_grad_norm + Lambda_grad)

        # EMA update
        momentum = 0.9
        self.importance_scores.mul_(momentum).add_(importance * (1 - momentum))

    def _compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute AdaLoRA delta: x @ Q^T @ diag(Λ * mask) @ P^T * scaling

        Casts float32 parameters to input dtype for forward pass.
        """
        compute_dtype = x.dtype

        # Apply rank mask to Lambda
        effective_lambda = self.Lambda * self.rank_mask  # [rank], float32

        # x: [..., Din]
        # x @ Q^T: [..., rank]
        xQ = x @ self.Q.t().to(compute_dtype)  # [..., rank]

        # Scale by Lambda (element-wise)
        xQL = xQ * effective_lambda.to(compute_dtype)  # [..., rank]

        # xQL @ P^T: [..., H]
        delta = xQL @ self.P.t().to(compute_dtype)  # [..., H]

        delta = delta * self._scaling.to(compute_dtype)

        if self.dropout is not None and self.training:
            delta = self.dropout(delta)

        return delta

    def _forward_sequence_selective(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Sequence-based routing: same weight for all tokens in a sequence."""
        B, T, _ = x.shape
        target_dtype = base.dtype

        expert_mask = (self._top_k_ids == self.expert_id)  # [B, k]
        mask = (self._top_k_weights * expert_mask.float()).sum(dim=-1)  # [B]

        if not mask.any():
            return base

        delta = self._compute_delta(x).to(target_dtype)  # [B, T, H]
        out = base + delta * mask.to(target_dtype).view(B, 1, 1)
        return out

    def _forward_token_selective(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Token-based routing: per-token weights using a mask matrix."""
        B, T, _ = x.shape
        target_dtype = base.dtype

        expert_mask = (self._top_k_ids == self.expert_id)  # [B, T, k]
        mask = (self._top_k_weights * expert_mask.float()).sum(dim=-1)  # [B, T]

        if not mask.any():
            return base

        delta = self._compute_delta(x).to(target_dtype)  # [B, T, H]
        out = base + delta * mask.to(target_dtype).unsqueeze(-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        target_dtype = self.linear.weight.dtype
        base = self.linear(x)

        # Always-on expert (shared): compute AdaLoRA for all, ignore routing
        if self.always_on:
            delta = self._compute_delta(x).to(target_dtype)
            out = base + delta
            return out.squeeze(1) if is_2d else out

        # No routing info or no expert_id: fallback to plain AdaLoRA for all
        if self._top_k_ids is None or self.expert_id is None:
            delta = self._compute_delta(x).to(target_dtype)
            out = base + delta
            return out.squeeze(1) if is_2d else out

        # Determine routing mode from tensor shape
        is_token_based = self._top_k_ids.dim() == 3  # [B, T, k] vs [B, k]

        if is_token_based:
            out = self._forward_token_selective(x, base)
        else:
            out = self._forward_sequence_selective(x, base)

        return out.squeeze(1) if is_2d else out

    def __repr__(self):
        mode = "shared" if self.always_on else f"expert_{self.expert_id}"
        eff_rank = self.get_effective_rank()
        return (f"MonkeyJumpAdaLoRALinear[{mode}](in={self.in_features}, "
                f"out={self.out_features}, rank={self.rank}, eff_rank={eff_rank}, dtype=float32)")


# -----------------------------
# MonkeyJump Router (unchanged from original)
# -----------------------------

class MonkeyJumpRouter(nn.Module):
    """
    Cosine-sim router with EMA centers and jitter noise.
    (Same as original - routing logic is independent of LoRA vs AdaLoRA)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        temperature: float = 1.0,
        ema_momentum: float = 0.9,
        top_k: int = 1,
        rep_mode: str = "last",
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.hidden_dim = hidden_dim
        self.momentum = ema_momentum
        self.rep_mode = rep_mode
        self.jitter_noise = jitter_noise

        self.register_buffer("_temperature", torch.tensor(temperature, dtype=torch.float32))
        centers = F.normalize(torch.randn(num_experts, hidden_dim), dim=-1)
        self.register_buffer("centers", centers)

    def _compute_logits(self, reps: torch.Tensor) -> torch.Tensor:
        reps_n = F.normalize(reps, dim=-1)
        centers_n = F.normalize(self.centers, dim=-1)
        logits = (reps_n @ centers_n.t()) / self._temperature.clamp(min=1e-6)

        if self.training and self.jitter_noise > 0:
            noise = torch.empty_like(logits).uniform_(
                1.0 - self.jitter_noise,
                1.0 + self.jitter_noise
            )
            logits = logits * noise

        return logits

    def _forward_token(
        self,
        hidden_states: torch.Tensor,
        update: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, H).float()

        logits = self._compute_logits(flat_hidden)
        probs = logits.softmax(dim=-1)

        top_k_weights, top_k_ids = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        top_k_ids = top_k_ids.reshape(B, T, self.top_k)
        top_k_weights = top_k_weights.reshape(B, T, self.top_k)

        if self.training and update:
            self._update_centers_ema(flat_hidden, top_k_ids.reshape(-1, self.top_k)[:, 0])

        return top_k_ids, top_k_weights

    def _forward_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: Optional[torch.Tensor],
        update: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rep_mode == "mean":
            seq_reps = _extract_mean_rep(hidden_states, attention_mask)
        elif self.rep_mode == "prompt_end":
            seq_reps = _extract_prompt_end_rep(hidden_states, prompt_lengths)
        else:
            seq_reps = _extract_last_token_rep(hidden_states, attention_mask)

        reps = seq_reps.float()
        logits = self._compute_logits(reps)
        probs = logits.softmax(dim=-1)

        top_k_weights, top_k_ids = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training and update:
            self._update_centers_ema(reps, top_k_ids[:, 0])

        return top_k_ids, top_k_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
        update: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)

        if self.rep_mode == "token":
            return self._forward_token(hidden_states, update)
        else:
            return self._forward_sequence(hidden_states, attention_mask, prompt_lengths, update)

    @torch.no_grad()
    def _update_centers_ema(self, reps: torch.Tensor, route_ids: torch.Tensor):
        N, H = reps.shape
        E = self.num_experts

        sums = torch.zeros(E, H, device=reps.device, dtype=torch.float32)
        counts = torch.zeros(E, 1, device=reps.device, dtype=torch.float32)

        sums.index_add_(0, route_ids, reps)
        counts.index_add_(0, route_ids, torch.ones(N, 1, device=reps.device))

        mask = counts > 0
        if mask.any():
            means = sums / (counts + 1e-6)
            updated = self.momentum * self.centers + (1 - self.momentum) * means
            updated = F.normalize(updated, dim=-1)
            self.centers.data = torch.where(mask, updated, self.centers)

    def __repr__(self):
        return (f"MonkeyJumpRouter(E={self.num_experts}, k={self.top_k}, "
                f"T={self._temperature.item():.2f}, momentum={self.momentum}, "
                f"mode={self.rep_mode}, jitter={self.jitter_noise})")


# -----------------------------
# Helpers (unchanged)
# -----------------------------

def _extract_last_token_rep(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if hidden_states.dim() == 2:
        return hidden_states

    B, T, H = hidden_states.shape

    if attention_mask is None:
        return hidden_states[:, -1, :]

    if attention_mask.dim() > 2:
        attention_mask = attention_mask.reshape(B, -1)
        if attention_mask.shape[1] > T:
            attention_mask = attention_mask[:, -T:]
        elif attention_mask.shape[1] < T:
            pad = torch.ones(B, T - attention_mask.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pad, attention_mask], dim=1)

    if attention_mask.dtype != torch.bool:
        if attention_mask.dtype.is_floating_point:
            attention_mask = attention_mask > -1e4
        else:
            attention_mask = attention_mask != 0

    seq_lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(B, device=hidden_states.device)
    return hidden_states[batch_idx, seq_lengths]


def _extract_mean_rep(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if hidden_states.dim() == 2:
        return hidden_states

    B, T, H = hidden_states.shape

    if attention_mask is None:
        return hidden_states.mean(dim=1)

    if attention_mask.dim() > 2:
        attention_mask = attention_mask.reshape(B, -1)
        if attention_mask.shape[1] > T:
            attention_mask = attention_mask[:, -T:]
        elif attention_mask.shape[1] < T:
            pad = torch.ones(B, T - attention_mask.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pad, attention_mask], dim=1)

    if attention_mask.dtype == torch.bool:
        mask = attention_mask.float()
    elif attention_mask.dtype.is_floating_point:
        mask = (attention_mask > -1e4).float()
    else:
        mask = (attention_mask != 0).float()

    mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def _extract_prompt_end_rep(
    hidden_states: torch.Tensor,
    prompt_lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    if hidden_states.dim() == 2:
        return hidden_states

    B, T, H = hidden_states.shape

    if prompt_lengths is None:
        return hidden_states[:, -1, :]

    batch_idx = torch.arange(B, device=hidden_states.device)
    return hidden_states[batch_idx, prompt_lengths]


def _get_parent_and_name(module: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _find_linear_paths(block: nn.Module, target_names: Sequence[str]) -> Dict[str, str]:
    result = {}
    target_set = set(target_names)

    for path, module in block.named_modules():
        if isinstance(module, nn.Linear):
            leaf_name = path.split(".")[-1] if path else ""
            if leaf_name in target_set and leaf_name not in result:
                result[leaf_name] = path

    return result


# -----------------------------
# Block wrapping
# -----------------------------

def _wrap_block(
    block: nn.Module,
    routed_linears: List[str],
    shared_linears: set,
    rank: int,
    alpha: float,
    dropout: float,
    temperature: float,
    ema_momentum: float,
    top_k: int,
    rep_mode: str,
    jitter_noise: float,
    init_scale: float,
) -> bool:
    """Wrap linear layers in a block with AdaLoRA. Returns True if anything was wrapped."""

    all_targets = list(routed_linears) + list(shared_linears)
    path_map = _find_linear_paths(block, all_targets)

    if not path_map:
        return False

    existing_routed = [name for name in routed_linears if name in path_map]
    existing_shared = [name for name in shared_linears if name in path_map]
    num_experts = len(existing_routed)

    if num_experts == 0 and len(existing_shared) == 0:
        return False

    routed_modules: List[MonkeyJumpAdaLoRALinear] = []

    # Wrap routed linears - each is ONE expert
    for expert_id, linear_name in enumerate(existing_routed):
        path = path_map[linear_name]
        parent, attr_name = _get_parent_and_name(block, path)
        old_linear = getattr(parent, attr_name)

        wrapped = MonkeyJumpAdaLoRALinear(
            old_linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            expert_id=expert_id,
            always_on=False,
            init_scale=init_scale,
        )
        setattr(parent, attr_name, wrapped)
        routed_modules.append(wrapped)

    # Wrap shared linears (always-on, no routing)
    for linear_name in existing_shared:
        path = path_map[linear_name]
        parent, attr_name = _get_parent_and_name(block, path)
        old_linear = getattr(parent, attr_name)

        wrapped = MonkeyJumpAdaLoRALinear(
            old_linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            expert_id=None,
            always_on=True,
            init_scale=init_scale,
        )
        setattr(parent, attr_name, wrapped)

    # Store config
    block._mj_routed_modules = routed_modules
    block._mj_num_experts = num_experts
    block._mj_config = {
        "temperature": temperature,
        "ema_momentum": ema_momentum,
        "top_k": min(top_k, num_experts) if num_experts > 0 else 1,
        "rep_mode": rep_mode,
        "jitter_noise": jitter_noise,
    }
    block._mj_router = None
    block._mj_routed_names = existing_routed
    block._mj_shared_names = existing_shared

    return True


def _patch_block_forward(block: nn.Module):
    """Monkey-patch block's forward for routing."""
    if getattr(block, "_mj_patched", False):
        return

    orig_forward = block.forward

    def new_forward(*args, **kwargs):
        hidden = None
        if args and torch.is_tensor(args[0]):
            hidden = args[0]
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]

        attention_mask = kwargs.get("attention_mask", None)
        prompt_lengths = kwargs.get("prompt_lengths", None)
        top_k_ids, top_k_weights = None, None

        if hidden is not None and hidden.dim() >= 2 and block._mj_num_experts > 0:
            was_2d = hidden.dim() == 2
            if was_2d:
                hidden_for_routing = hidden.unsqueeze(1)
            else:
                hidden_for_routing = hidden

            B, T, H = hidden_for_routing.shape

            if block._mj_router is None:
                block._mj_router = MonkeyJumpRouter(
                    hidden_dim=H,
                    num_experts=block._mj_num_experts,
                    temperature=block._mj_config["temperature"],
                    ema_momentum=block._mj_config["ema_momentum"],
                    top_k=block._mj_config["top_k"],
                    rep_mode=block._mj_config["rep_mode"],
                    jitter_noise=block._mj_config["jitter_noise"],
                ).to(hidden.device)

            block._mj_router.to(hidden.device)

            update_flag = kwargs.pop("update_router", False) or getattr(
                block, "_trainer_update_flag", False
            )
            top_k_ids, top_k_weights = block._mj_router(
                hidden_for_routing,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
                update=update_flag,
            )

            if was_2d and block._mj_config["rep_mode"] == "token":
                top_k_ids = top_k_ids.squeeze(1)
                top_k_weights = top_k_weights.squeeze(1)

        for module in block._mj_routed_modules:
            module.set_routing(top_k_ids, top_k_weights)

        try:
            return orig_forward(*args, **kwargs)
        finally:
            for module in block._mj_routed_modules:
                module.set_routing(None, None)

    block.forward = new_forward
    block._mj_patched = True


def _add_block_repr(block: nn.Module):
    if getattr(block, "_mj_repr_patched", False):
        return

    orig_repr = block.__repr__

    def new_repr():
        base = orig_repr()
        cfg = block._mj_config
        lines = [
            "",
            "  [MonkeyJump-AdaLoRA]",
            f"    routed: {block._mj_routed_names}",
            f"    shared: {block._mj_shared_names}",
            f"    num_experts: {block._mj_num_experts}",
            f"    top_k: {cfg['top_k']}",
            f"    rep_mode: {cfg['rep_mode']}",
            f"    jitter_noise: {cfg['jitter_noise']}",
            f"    adalora_dtype: float32",
        ]
        return base + "\n" + "\n".join(lines)

    block.__repr__ = new_repr
    block._mj_repr_patched = True


# -----------------------------
# Public API
# -----------------------------

def apply_monkeyjump(
    model: nn.Module,
    *,
    blocks: Dict[str, Sequence[int]],
    linears: Sequence[str],
    shared_expert: Optional[Union[str, Sequence[str]]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    temperature: float = 1.0,
    ema_momentum: float = 0.9,
    top_k: int = 1,
    rep_mode: str = "last",
    jitter_noise: float = 0.0,
    init_scale: float = 0.01,
) -> nn.Module:
    """
    Apply MonkeyJump-AdaLoRA to selected blocks.

    MonkeyJump-AdaLoRA uses SVD-based parameterization (P, Λ, Q) with:
    - Per-linear experts (each linear = one expert)
    - Importance-based rank pruning
    - Cosine-similarity routing with EMA centers
    - Jitter noise for exploration and load balancing

    Args:
        model: Base model to wrap
        blocks: Dict mapping class names to layer indices
        linears: List of linear layer names to wrap as routed experts
        shared_expert: Linear name(s) for always-on shared expert(s)
        rank: Initial AdaLoRA rank (can be pruned during training)
        alpha: Scaling factor (scaling = alpha / rank)
        dropout: Dropout rate
        temperature: Router softmax temperature
        ema_momentum: EMA momentum for center updates
        top_k: Number of experts to route to
        rep_mode: Routing mode ("token", "last", "mean", "prompt_end")
        jitter_noise: Multiplicative noise for router logits
        init_scale: Initial scale for singular values (Lambda)

    Returns:
        Modified model with MonkeyJump-AdaLoRA applied
        
    Note:
        P, Λ, Q parameters are kept in float32 regardless of model dtype to ensure
        gradient updates are not lost due to bfloat16 precision limits.
    """
    _freeze_module_parameters(model)

    if shared_expert is None:
        shared_experts_set = set()
    elif isinstance(shared_expert, str):
        shared_experts_set = {shared_expert}
    else:
        shared_experts_set = set(shared_expert)

    seen = set()
    routed_linears = []
    for l in linears:
        if l not in shared_experts_set and l not in seen:
            routed_linears.append(l)
            seen.add(l)

    class_counters = {cls: 0 for cls in blocks}

    def _walk(parent: nn.Module):
        for child_name, child in list(parent.named_children()):
            cls_name = child.__class__.__name__

            if cls_name in blocks:
                idx = class_counters[cls_name]
                target_indices = set(int(i) for i in blocks[cls_name])

                if idx in target_indices:
                    if _wrap_block(
                        child,
                        routed_linears=routed_linears,
                        shared_linears=shared_experts_set,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        temperature=temperature,
                        ema_momentum=ema_momentum,
                        top_k=top_k,
                        rep_mode=rep_mode,
                        jitter_noise=jitter_noise,
                        init_scale=init_scale,
                    ):
                        _patch_block_forward(child)
                        _add_block_repr(child)

                class_counters[cls_name] = idx + 1

            _walk(getattr(parent, child_name))

    _walk(model)
    return model


# -----------------------------
# AdaLoRA-specific Utilities
# -----------------------------

def get_adalora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all AdaLoRA P, Lambda, and Q parameters."""
    params = []
    for module in model.modules():
        if isinstance(module, MonkeyJumpAdaLoRALinear):
            params.extend([module.P, module.Lambda, module.Q])
    return params


def get_router_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all router parameters (if any are trainable)."""
    params = []
    for module in model.modules():
        if isinstance(module, MonkeyJumpRouter):
            params.extend([p for p in module.parameters() if p.requires_grad])
    return params


def get_trainable_parameter_count(model: nn.Module) -> Dict[str, int]:
    """Count trainable parameters by category."""
    adalora = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, MonkeyJumpAdaLoRALinear)
        for p in [m.P, m.Lambda, m.Q]
    )
    other = sum(p.numel() for p in model.parameters() if p.requires_grad) - adalora
    return {"adalora": adalora, "other": other, "total": adalora + other}


def update_all_importance_scores(model: nn.Module):
    """
    Update importance scores for all AdaLoRA modules.
    Call this after optimizer.step() in training loop.
    """
    for module in model.modules():
        if isinstance(module, MonkeyJumpAdaLoRALinear):
            module.update_importance_scores()


def prune_all_to_rank(model: nn.Module, target_rank: int):
    """Prune all AdaLoRA modules to target rank."""
    for module in model.modules():
        if isinstance(module, MonkeyJumpAdaLoRALinear):
            module.prune_to_rank(target_rank)


def get_rank_distribution(model: nn.Module) -> Dict[str, int]:
    """Get effective rank for each AdaLoRA module."""
    dist = {}
    for name, module in model.named_modules():
        if isinstance(module, MonkeyJumpAdaLoRALinear):
            dist[name] = module.get_effective_rank()
    return dist


def get_total_effective_rank(model: nn.Module) -> int:
    """Get sum of effective ranks across all AdaLoRA modules."""
    return sum(
        m.get_effective_rank()
        for m in model.modules()
        if isinstance(m, MonkeyJumpAdaLoRALinear)
    )


def apply_rank_budget(
    model: nn.Module,
    total_budget: int,
    min_rank: int = 1,
):
    """
    Distribute rank budget across modules based on importance.

    Args:
        model: Model with AdaLoRA modules
        total_budget: Total rank budget to distribute
        min_rank: Minimum rank per module
    """
    modules = [
        m for m in model.modules()
        if isinstance(m, MonkeyJumpAdaLoRALinear)
    ]

    if not modules:
        return

    # Collect all importance scores with module reference
    all_scores = []
    for m in modules:
        for i in range(m.rank):
            all_scores.append((m.importance_scores[i].item(), m, i))

    # Sort by importance (descending)
    all_scores.sort(key=lambda x: x[0], reverse=True)

    # Reset all masks
    for m in modules:
        m.rank_mask.zero_()

    # Ensure minimum rank per module
    remaining_budget = total_budget
    for m in modules:
        # Find top min_rank indices for this module
        module_scores = [(m.importance_scores[i].item(), i) for i in range(m.rank)]
        module_scores.sort(reverse=True)
        for j in range(min(min_rank, m.rank)):
            if remaining_budget > 0:
                m.rank_mask[module_scores[j][1]] = 1.0
                remaining_budget -= 1

    # Distribute remaining budget by global importance
    for score, m, i in all_scores:
        if remaining_budget <= 0:
            break
        if m.rank_mask[i] == 0:  # Not already allocated
            m.rank_mask[i] = 1.0
            remaining_budget -= 1


def print_monkeyjump_summary(model: nn.Module):
    """Print a summary of MonkeyJump-AdaLoRA configuration."""
    print("=" * 60)
    print("MonkeyJump-AdaLoRA Summary")
    print("=" * 60)

    adalora_modules = sum(1 for m in model.modules() if isinstance(m, MonkeyJumpAdaLoRALinear))
    routers = sum(1 for m in model.modules() if isinstance(m, MonkeyJumpRouter))

    mode = "unknown"
    jitter = 0.0
    for m in model.modules():
        if isinstance(m, MonkeyJumpRouter):
            mode = m.rep_mode
            jitter = m.jitter_noise
            break

    total_rank = sum(m.rank for m in model.modules() if isinstance(m, MonkeyJumpAdaLoRALinear))
    eff_rank = get_total_effective_rank(model)

    print(f"AdaLoRA modules: {adalora_modules}")
    print(f"Routers: {routers}")
    print(f"Routing mode: {mode.upper()}")
    print(f"Jitter noise: {jitter}")
    print(f"Total rank capacity: {total_rank}")
    print(f"Effective rank (active): {eff_rank}")
    print(f"AdaLoRA dtype: float32 (for gradient precision)")
    print("Parameterization: SVD (P, Λ, Q)")

    counts = get_trainable_parameter_count(model)
    print(f"Trainable params: {counts['adalora']:,}")
    print("=" * 60)


# -----------------------------
# Diagnostic Utilities
# -----------------------------

def check_adalora_gradients(model: nn.Module):
    """
    Check gradient flow to AdaLoRA P, Lambda, Q weights. Call after loss.backward().
    
    Usage:
        loss.backward()
        check_adalora_gradients(model)
    """
    for name, m in model.named_modules():
        if isinstance(m, MonkeyJumpAdaLoRALinear):
            print(f"{name}:")
            if m.P.grad is not None:
                print(f"  P grad norm: {m.P.grad.norm().item():.6f}, dtype: {m.P.dtype}")
            else:
                print(f"  P grad: None")
            if m.Lambda.grad is not None:
                print(f"  Λ grad norm: {m.Lambda.grad.norm().item():.6f}, dtype: {m.Lambda.dtype}")
            else:
                print(f"  Λ grad: None")
            if m.Q.grad is not None:
                print(f"  Q grad norm: {m.Q.grad.norm().item():.6f}, dtype: {m.Q.dtype}")
            else:
                print(f"  Q grad: None")


def fix_adalora_dtype(model: nn.Module):
    """
    Fix AdaLoRA dtype if you have an existing model with weights in bfloat16.
    Call this to convert to float32 for better gradient precision.
    
    Args:
        model: The model
    """
    for name, m in model.named_modules():
        if isinstance(m, MonkeyJumpAdaLoRALinear):
            old_dtype = m.P.dtype
            if old_dtype != torch.float32:
                m.P.data = m.P.data.float()
                m.Lambda.data = m.Lambda.data.float()
                m.Q.data = m.Q.data.float()
                m._scaling.data = m._scaling.data.float()
                m.rank_mask.data = m.rank_mask.data.float()
                m.importance_scores.data = m.importance_scores.data.float()
                print(f"{name}: converted from {old_dtype} to float32")


# -----------------------------
# Router Utilities
# -----------------------------

def enable_router_updates(model: nn.Module, enable: bool = True):
    """Enable or disable EMA center updates during training."""
    for module in model.modules():
        if hasattr(module, "_mj_patched") and module._mj_patched:
            module._trainer_update_flag = enable


def set_all_routers_top_k(model: nn.Module, top_k: int):
    """Set top_k for all routers."""
    for module in model.modules():
        if hasattr(module, "_mj_router") and module._mj_router is not None:
            module._mj_router.top_k = min(top_k, module._mj_router.num_experts)
        if hasattr(module, "_mj_config"):
            num_exp = getattr(module, "_mj_num_experts", top_k)
            module._mj_config["top_k"] = min(top_k, num_exp)


def set_all_routers_temperature(model: nn.Module, temperature: float):
    """Set temperature for all routers."""
    for module in model.modules():
        if hasattr(module, "_mj_router") and module._mj_router is not None:
            module._mj_router._temperature.fill_(temperature)
        if hasattr(module, "_mj_config"):
            module._mj_config["temperature"] = temperature


def set_all_routers_jitter(model: nn.Module, jitter_noise: float):
    """Set jitter noise for all routers."""
    for module in model.modules():
        if hasattr(module, "_mj_router") and module._mj_router is not None:
            module._mj_router.jitter_noise = jitter_noise
        if hasattr(module, "_mj_config"):
            module._mj_config["jitter_noise"] = jitter_noise


def get_router_statistics(model: nn.Module) -> Dict[str, Dict]:
    """Get statistics for all routers."""
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, "_mj_router") and module._mj_router is not None:
            router = module._mj_router
            stats[name] = {
                "num_experts": router.num_experts,
                "top_k": router.top_k,
                "temperature": router._temperature.item(),
                "ema_momentum": router.momentum,
                "rep_mode": router.rep_mode,
                "jitter_noise": router.jitter_noise,
            }
    return stats


# -----------------------------
# Prompt Length Helpers
# -----------------------------

def find_prompt_lengths(
    input_ids: torch.Tensor,
    assistant_token_id: int,
) -> torch.Tensor:
    """Find position of assistant token in each sequence."""
    B, T = input_ids.shape
    prompt_lengths = []

    for i in range(B):
        positions = (input_ids[i] == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            prompt_lengths.append(positions[-1].item())
        else:
            prompt_lengths.append(T - 1)

    return torch.tensor(prompt_lengths, device=input_ids.device, dtype=torch.long)


def find_prompt_lengths_batch(
    input_ids: torch.Tensor,
    assistant_token_id: int,
) -> torch.Tensor:
    """Vectorized version of find_prompt_lengths."""
    B, T = input_ids.shape

    is_assistant = (input_ids == assistant_token_id)
    reversed_mask = is_assistant.flip(dims=[1])
    has_token = is_assistant.any(dim=1)

    first_in_reversed = reversed_mask.float().argmax(dim=1)
    last_position = T - 1 - first_in_reversed

    fallback = torch.full((B,), T - 1, device=input_ids.device, dtype=torch.long)
    prompt_lengths = torch.where(has_token, last_position, fallback)

    return prompt_lengths