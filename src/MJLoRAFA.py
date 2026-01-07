# monkeyjump_lora_fa.py
# MonkeyJump-LoRA-FA: MoE-LoRA with routing + probability-weighted experts + jitter noise
# - LoRA-FA variant: A is frozen (random init), only B is trainable
# - Each linear is ONE expert (not all experts)
# - LOGICALLY SELECTIVE: only routed tokens/sequences receive non-zero LoRA
#   (computation is still dense; non-selected positions are masked to 0)
# - Supports: token, last, mean, prompt_end routing modes
# - Jitter noise for better exploration and load balancing
# - FIXED: LoRA weights in float32 for gradient precision in bfloat16 training

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze_module_parameters(mod: nn.Module):
    for p in mod.parameters():
        p.requires_grad = False


# -----------------------------
# Masked LoRA-FA Linear (One Expert per Linear)
# -----------------------------

class MonkeyJumpLinear(nn.Module):
    """
    LoRA-FA adapter - ONE expert per linear layer.
    
    LoRA-FA: A is frozen (random initialization), only B is trainable.
    This reduces trainable parameters by ~50% while maintaining performance.

    Routing logic:
    - Router produces top-k experts and probabilities.
    - For each expert (MonkeyJumpLinear with expert_id):
      - It contributes only where it appears in top-k.
      - Contribution is scaled by its routing probability.
      - Positions where it is not in top-k get a 0 mask → no effect.

    Important:
    - Computation is dense: we still compute delta for all tokens/sequences,
      then multiply by a mask that is 0 for non-selected positions.
      This keeps it close to standard LoRA in speed while being
      *logically* selective.

    Supports both:
    - Sequence-based routing: weights [B, k] -> same weight for all tokens in a sequence
    - Token-based routing: weights [B, T, k] -> per-token weights
    
    NOTE: LoRA weights (A buffer, B parameter) are kept in float32 for gradient 
    precision during bfloat16 training.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        expert_id: Optional[int] = None,
        always_on: bool = False,
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

        # LoRA-FA: A is frozen (buffer), only B is trainable (parameter)
        # FIX: Both in float32 for gradient precision
        A = torch.empty(rank, Din, device=base_device, dtype=torch.float32)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.register_buffer("A", A)  # Frozen, not trainable

        # FIX: B in float32 for gradient precision during bfloat16 training
        self.B = nn.Parameter(torch.empty(H, rank, device=base_device, dtype=torch.float32))
        nn.init.zeros_(self.B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._top_k_ids: Optional[torch.Tensor] = None
        self._top_k_weights: Optional[torch.Tensor] = None

    def set_routing(self, top_k_ids: Optional[torch.Tensor], top_k_weights: Optional[torch.Tensor]):
        """
        Set routing weights.

        For sequence-based: top_k_ids [B, k], top_k_weights [B, k]
        For token-based: top_k_ids [B, T, k], top_k_weights [B, T, k]
        """
        self._top_k_ids = top_k_ids
        self._top_k_weights = top_k_weights

    def _compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta: x @ A^T @ B^T * scaling. Casts to input dtype for computation."""
        compute_dtype = x.dtype
        # Cast float32 weights to compute dtype for forward pass
        xA = x @ self.A.t().to(compute_dtype)
        xAB = xA @ self.B.t().to(compute_dtype)
        delta = xAB * self._scaling.to(compute_dtype)
        if self.dropout is not None and self.training:
            delta = self.dropout(delta)
        return delta

    def _forward_sequence_selective(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """
        Sequence-based routing: same weight for all tokens in a sequence.

        Mask vector [B]:
        - Non-selected sequences → 0 (no LoRA contribution)
        - Selected sequences → probability score from routing

        Output = base + (delta * mask)
        - Where mask=0: output = base (no LoRA effect)
        - Where mask>0: output = base + delta * prob (weighted LoRA)

        Args:
            x: [B, T, D] input
            base: [B, T, H] base linear output
        Returns:
            [B, T, H] output
        """
        B, T, _ = x.shape
        target_dtype = base.dtype

        # Build mask vector [B]: probability score if this expert is selected, else 0
        # top_k_ids: [B, k], top_k_weights: [B, k]
        expert_mask = (self._top_k_ids == self.expert_id)  # [B, k]
        mask = (self._top_k_weights * expert_mask.float()).sum(dim=-1)  # [B]

        # If this expert is never selected for any sequence in the batch, skip LoRA
        if not mask.any():
            return base

        # Dense compute: LoRA for all sequences
        delta = self._compute_delta(x).to(target_dtype)  # [B, T, H]

        # Apply mask: [B, 1, 1] broadcasts to [B, T, H]
        out = base + delta * mask.to(target_dtype).view(B, 1, 1)
        return out

    def _forward_token_selective(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """
        Token-based routing: per-token weights using a mask matrix.

        Mask matrix [B, T]:
        - Non-selected tokens → 0 (no LoRA contribution)
        - Selected tokens → probability score from routing

        Output = base + (delta * mask)
        - Where mask=0: output = base (no LoRA effect)
        - Where mask>0: output = base + delta * prob (weighted LoRA)

        Args:
            x: [B, T, D] input
            base: [B, T, H] base linear output
        Returns:
            [B, T, H] output
        """
        B, T, _ = x.shape
        target_dtype = base.dtype

        # Build mask matrix [B, T]: probability score if this expert is selected, else 0
        # top_k_ids: [B, T, k], top_k_weights: [B, T, k]
        expert_mask = (self._top_k_ids == self.expert_id)  # [B, T, k]
        mask = (self._top_k_weights * expert_mask.float()).sum(dim=-1)  # [B, T]

        # If this expert is never selected for any token in the batch, skip LoRA
        if not mask.any():
            return base

        # Dense compute: LoRA for all tokens
        delta = self._compute_delta(x).to(target_dtype)  # [B, T, H]

        # Apply mask: [B, T, 1] broadcasts to [B, T, H]
        out = base + delta * mask.to(target_dtype).unsqueeze(-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        target_dtype = self.linear.weight.dtype
        base = self.linear(x)

        # Always-on expert (shared): compute LoRA for all, ignore routing
        if self.always_on:
            delta = self._compute_delta(x).to(target_dtype)
            out = base + delta
            return out.squeeze(1) if is_2d else out

        # No routing info or no expert_id: fallback to plain LoRA for all
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
        return f"MonkeyJumpLinear[{mode}](in={self.in_features}, out={self.out_features}, rank={self.rank}, frozen_A=True, dtype=float32)"


# -----------------------------
# MonkeyJump Router
# -----------------------------

class MonkeyJumpRouter(nn.Module):
    """
    Cosine-sim router with EMA centers and jitter noise.

    Supports four modes:
    - "token": Per-token routing (each token routes independently)
    - "last": Sequence routing using last token representation
    - "mean": Sequence routing using mean of all token representations
    - "prompt_end": Sequence routing using token at prompt_lengths position

    Jitter noise (training only):
    - Multiplicative noise applied to logits for better exploration
    - Helps prevent router collapse and improves load balancing
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        temperature: float = 1.0,
        ema_momentum: float = 0.9,
        top_k: int = 1,
        rep_mode: str = "last",  # "token", "last", "mean", or "prompt_end"
        jitter_noise: float = 0.0,  # 0.0 = no noise, typical: 0.01 - 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.hidden_dim = hidden_dim
        self.momentum = ema_momentum
        self.rep_mode = rep_mode
        self.jitter_noise = jitter_noise

        self.register_buffer("_temperature", torch.tensor(temperature, dtype=torch.float32))

        # Initialize centers with unit norm
        centers = F.normalize(torch.randn(num_experts, hidden_dim), dim=-1)
        self.register_buffer("centers", centers)

    def _compute_logits(self, reps: torch.Tensor) -> torch.Tensor:
        """
        Compute routing logits with optional jitter noise.

        Args:
            reps: [N, H] normalized representations

        Returns:
            [N, E] logits
        """
        reps_n = F.normalize(reps, dim=-1)
        centers_n = F.normalize(self.centers, dim=-1)

        logits = (reps_n @ centers_n.t()) / self._temperature.clamp(min=1e-6)

        # Apply multiplicative jitter noise during training
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
        """
        Token-based routing - each token routes independently.

        Args:
            hidden_states: [B, T, H]
            update: whether to update EMA centers

        Returns:
            top_k_ids: [B, T, k]
            top_k_weights: [B, T, k]
        """
        B, T, H = hidden_states.shape

        # Flatten: [B*T, H]
        flat_hidden = hidden_states.reshape(-1, H).float()

        # Compute logits with jitter
        logits = self._compute_logits(flat_hidden)
        probs = logits.softmax(dim=-1)

        # Top-k selection
        top_k_weights, top_k_ids = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Reshape back to [B, T, k]
        top_k_ids = top_k_ids.reshape(B, T, self.top_k)
        top_k_weights = top_k_weights.reshape(B, T, self.top_k)

        # EMA update
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
        """
        Sequence-based routing using last token, mean, or prompt_end.

        Args:
            hidden_states: [B, T, H]
            attention_mask: [B, T] optional
            prompt_lengths: [B] optional - position of prompt end token
            update: whether to update EMA centers

        Returns:
            top_k_ids: [B, k]
            top_k_weights: [B, k]
        """
        # Extract sequence representation based on mode
        if self.rep_mode == "mean":
            seq_reps = _extract_mean_rep(hidden_states, attention_mask)
        elif self.rep_mode == "prompt_end":
            seq_reps = _extract_prompt_end_rep(hidden_states, prompt_lengths)
        else:  # "last"
            seq_reps = _extract_last_token_rep(hidden_states, attention_mask)

        reps = seq_reps.float()

        # Compute logits with jitter
        logits = self._compute_logits(reps)
        probs = logits.softmax(dim=-1)

        # Top-k selection
        top_k_weights, top_k_ids = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # EMA update
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
        """
        Compute routing weights.

        Args:
            hidden_states: [B, T, H] or [B, H]
            attention_mask: [B, T] optional (for sequence modes)
            prompt_lengths: [B] optional (for prompt_end mode)
            update: whether to update EMA centers

        Returns:
            For token mode: top_k_ids [B, T, k], top_k_weights [B, T, k]
            For sequence modes: top_k_ids [B, k], top_k_weights [B, k]
        """
        # Handle 2D input
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)

        if self.rep_mode == "token":
            return self._forward_token(hidden_states, update)
        else:
            return self._forward_sequence(hidden_states, attention_mask, prompt_lengths, update)

    @torch.no_grad()
    def _update_centers_ema(self, reps: torch.Tensor, route_ids: torch.Tensor):
        """
        Update centers using EMA.

        Args:
            reps: [N, H] - representations
            route_ids: [N] - primary expert assignment
        """
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
            # Re-normalize to keep centers unit norm
            updated = F.normalize(updated, dim=-1)
            self.centers.data = torch.where(mask, updated, self.centers)

    def __repr__(self):
        return (f"MonkeyJumpRouter(E={self.num_experts}, k={self.top_k}, "
                f"T={self._temperature.item():.2f}, momentum={self.momentum}, "
                f"mode={self.rep_mode}, jitter={self.jitter_noise})")


# -----------------------------
# Helpers
# -----------------------------

def _extract_last_token_rep(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Extract LAST VALID TOKEN representation per sequence.

    Args:
        hidden_states: [B, T, H]
        attention_mask: [B, T] or various shapes

    Returns:
        [B, H] - last token representation for each sequence
    """
    if hidden_states.dim() == 2:
        return hidden_states

    B, T, H = hidden_states.shape

    if attention_mask is None:
        return hidden_states[:, -1, :]

    # Handle various mask shapes
    if attention_mask.dim() > 2:
        attention_mask = attention_mask.reshape(B, -1)
        if attention_mask.shape[1] > T:
            attention_mask = attention_mask[:, -T:]
        elif attention_mask.shape[1] < T:
            pad = torch.ones(
                B,
                T - attention_mask.shape[1],
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=1)

    # Convert to boolean
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
    """
    Extract MEAN representation per sequence (masked).

    Args:
        hidden_states: [B, T, H]
        attention_mask: [B, T] or various shapes

    Returns:
        [B, H] - mean representation for each sequence
    """
    if hidden_states.dim() == 2:
        return hidden_states

    B, T, H = hidden_states.shape

    if attention_mask is None:
        return hidden_states.mean(dim=1)

    # Handle various mask shapes
    if attention_mask.dim() > 2:
        attention_mask = attention_mask.reshape(B, -1)
        if attention_mask.shape[1] > T:
            attention_mask = attention_mask[:, -T:]
        elif attention_mask.shape[1] < T:
            pad = torch.ones(
                B,
                T - attention_mask.shape[1],
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=1)

    # Convert to float for masking
    if attention_mask.dtype == torch.bool:
        mask = attention_mask.float()
    elif attention_mask.dtype.is_floating_point:
        mask = (attention_mask > -1e4).float()
    else:
        mask = (attention_mask != 0).float()

    mask = mask.unsqueeze(-1)  # [B, T, 1]
    denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
    return (hidden_states * mask).sum(dim=1) / denom


def _extract_prompt_end_rep(
    hidden_states: torch.Tensor,
    prompt_lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Extract representation at prompt end position (e.g., assistant token).

    This provides consistent routing at both train and inference time,
    since the prompt_end position is the same in both cases.

    Args:
        hidden_states: [B, T, H]
        prompt_lengths: [B] - index of prompt end token for each sequence

    Returns:
        [B, H] - representation at prompt end position
    """
    if hidden_states.dim() == 2:
        return hidden_states

    B, T, H = hidden_states.shape

    # Fallback to last token if prompt_lengths not provided
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
) -> bool:
    """Wrap linear layers in a block. Returns True if anything was wrapped."""

    all_targets = list(routed_linears) + list(shared_linears)
    path_map = _find_linear_paths(block, all_targets)

    if not path_map:
        return False

    existing_routed = [name for name in routed_linears if name in path_map]
    existing_shared = [name for name in shared_linears if name in path_map]
    num_experts = len(existing_routed)

    if num_experts == 0 and len(existing_shared) == 0:
        return False

    routed_modules: List[MonkeyJumpLinear] = []

    # Wrap routed linears - each is ONE expert
    for expert_id, linear_name in enumerate(existing_routed):
        path = path_map[linear_name]
        parent, attr_name = _get_parent_and_name(block, path)
        old_linear = getattr(parent, attr_name)

        wrapped = MonkeyJumpLinear(
            old_linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            expert_id=expert_id,
            always_on=False,
        )
        setattr(parent, attr_name, wrapped)
        routed_modules.append(wrapped)

    # Wrap shared linears (always-on, no routing)
    for linear_name in existing_shared:
        path = path_map[linear_name]
        parent, attr_name = _get_parent_and_name(block, path)
        old_linear = getattr(parent, attr_name)

        wrapped = MonkeyJumpLinear(
            old_linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            expert_id=None,
            always_on=True,
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

            # Lazy router creation
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

            # Handle 2D input for token mode
            if was_2d and block._mj_config["rep_mode"] == "token":
                top_k_ids = top_k_ids.squeeze(1)
                top_k_weights = top_k_weights.squeeze(1)

        # Set routing on all modules
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
            "  [MonkeyJump-LoRA-FA]",
            f"    routed: {block._mj_routed_names}",
            f"    shared: {block._mj_shared_names}",
            f"    num_experts: {block._mj_num_experts}",
            f"    top_k: {cfg['top_k']}",
            f"    rep_mode: {cfg['rep_mode']}",
            f"    jitter_noise: {cfg['jitter_noise']}",
            f"    frozen_A: True",
            f"    lora_dtype: float32",
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
    rep_mode: str = "last",  # "token", "last", "mean", or "prompt_end"
    jitter_noise: float = 0.0,  # 0.0 = no noise, typical: 0.01 - 0.1
) -> nn.Module:
    """
    Apply MonkeyJump-LoRA-FA to selected blocks.

    MonkeyJump-LoRA-FA is a Mixture-of-Experts LoRA with:
    - LoRA-FA: A is frozen (random init), only B is trainable (~50% fewer params)
    - Per-linear experts (each linear = one expert)
    - Cosine-similarity routing with EMA centers
    - Jitter noise for exploration and load balancing
    - Multiple routing modes for different use cases

    Args:
        model: Base model to wrap
        blocks: Dict mapping class names to layer indices, e.g.
                {"Qwen2DecoderLayer": [0, 1, 2, ...]}
        linears: List of linear layer names to wrap as routed experts,
                 e.g. ["q_proj", "k_proj", "v_proj", "o_proj"]
        shared_expert: Linear name(s) for always-on shared expert(s).
                       These bypass routing and apply LoRA everywhere.
        rank: LoRA rank
        alpha: LoRA alpha (scaling = alpha / rank)
        dropout: Dropout rate for LoRA
        temperature: Router softmax temperature (lower = sharper routing)
        ema_momentum: EMA momentum for center updates
        top_k: Number of experts to route to
        rep_mode: Routing mode:
            - "token": Each token routes independently [B, T, k]
            - "last": Sequence routing using last token [B, k]
            - "mean": Sequence routing using mean pooling [B, k]
            - "prompt_end": Sequence routing using prompt_lengths position [B, k]
        jitter_noise: Multiplicative noise for router logits (training only).
                      0.0 = disabled, typical range: 0.01 - 0.1
                      Helps prevent router collapse and improves load balancing.

    Returns:
        Modified model with MonkeyJump-LoRA-FA applied
        
    Note:
        LoRA B weights are kept in float32 regardless of model dtype to ensure
        gradient updates are not lost due to bfloat16 precision limits.
        A weights are frozen (not trainable) but also stored in float32.
    """
    _freeze_module_parameters(model)

    # Normalize shared_expert to a set
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
                    ):
                        _patch_block_forward(child)
                        _add_block_repr(child)

                class_counters[cls_name] = idx + 1

            _walk(getattr(parent, child_name))

    _walk(model)
    return model


# Alias for backward compatibility
apply_MonkeyJump = apply_monkeyjump


# -----------------------------
# Utilities
# -----------------------------

def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA B parameters (A is frozen in LoRA-FA)."""
    params = []
    for module in model.modules():
        if isinstance(module, MonkeyJumpLinear):
            params.append(module.B)
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
    lora = sum(
        m.B.numel()
        for m in model.modules()
        if isinstance(m, MonkeyJumpLinear)
    )
    other = sum(p.numel() for p in model.parameters() if p.requires_grad) - lora
    return {"lora": lora, "other": other, "total": lora + other}


def print_monkeyjump_summary(model: nn.Module):
    """Print a summary of MonkeyJump-LoRA-FA configuration."""
    print("=" * 60)
    print("MonkeyJump-LoRA-FA Summary")
    print("=" * 60)

    lora_modules = sum(1 for m in model.modules() if isinstance(m, MonkeyJumpLinear))
    routers = sum(1 for m in model.modules() if isinstance(m, MonkeyJumpRouter))

    mode = "unknown"
    jitter = 0.0
    for m in model.modules():
        if isinstance(m, MonkeyJumpRouter):
            mode = m.rep_mode
            jitter = m.jitter_noise
            break

    print(f"LoRA modules: {lora_modules}")
    print(f"Routers: {routers}")
    print(f"Routing mode: {mode.upper()}")
    print(f"Jitter noise: {jitter}")
    print(f"LoRA dtype: float32 (for gradient precision)")
    print("Computation: dense LoRA with routing masks (logical selectivity)")
    print("LoRA-FA: A frozen (random init), only B trainable")

    counts = get_trainable_parameter_count(model)
    print(f"Trainable params (B only): {counts['lora']:,}")
    print("=" * 60)


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
# Diagnostic utilities
# -----------------------------

def check_lora_gradients(model: nn.Module):
    """
    Check gradient flow to LoRA B weights. Call after loss.backward().
    
    Usage:
        loss.backward()
        check_lora_gradients(model)
    """
    for name, m in model.named_modules():
        if isinstance(m, MonkeyJumpLinear):
            if m.B.grad is not None:
                b_grad_norm = m.B.grad.norm().item()
                print(f"{name}:")
                print(f"  B grad norm: {b_grad_norm:.6f}, dtype: {m.B.dtype}")
                print(f"  A dtype: {m.A.dtype} (frozen)")
            else:
                print(f"{name}: B grad=None")


def fix_lora_dtype(model: nn.Module):
    """
    Fix LoRA dtype if you have an existing model with weights in bfloat16.
    Call this to convert to float32 for better gradient precision.
    
    Args:
        model: The model
    """
    for name, m in model.named_modules():
        if isinstance(m, MonkeyJumpLinear):
            old_dtype = m.B.dtype
            if old_dtype != torch.float32:
                m.B.data = m.B.data.float()
                m.A.data = m.A.data.float()
                m._scaling.data = m._scaling.data.float()
                print(f"{name}: converted from {old_dtype} to float32")


# -----------------------------
# Helper for finding prompt lengths
# -----------------------------

def find_prompt_lengths(
    input_ids: torch.Tensor,
    assistant_token_id: int,
) -> torch.Tensor:
    """
    Find position of assistant token in each sequence.
    Run ONCE during data preparation, not in forward pass.

    Args:
        input_ids: [B, T] input token ids
        assistant_token_id: token id for "assistant" or similar marker

    Returns:
        [B] - position of assistant token for each sequence
    """
    B, T = input_ids.shape
    prompt_lengths = []

    for i in range(B):
        positions = (input_ids[i] == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            # Take the last occurrence (in case there are multiple)
            prompt_lengths.append(positions[-1].item())
        else:
            # Fallback to last position if not found
            prompt_lengths.append(T - 1)

    return torch.tensor(prompt_lengths, device=input_ids.device, dtype=torch.long)


def find_prompt_lengths_batch(
    input_ids: torch.Tensor,
    assistant_token_id: int,
) -> torch.Tensor:
    """
    Vectorized version of find_prompt_lengths for better performance.

    Args:
        input_ids: [B, T] input token ids
        assistant_token_id: token id for "assistant" or similar marker

    Returns:
        [B] - position of assistant token for each sequence
    """
    B, T = input_ids.shape

    # Create mask where assistant token appears
    is_assistant = (input_ids == assistant_token_id)  # [B, T]

    # Find last occurrence using argmax on reversed tensor
    reversed_mask = is_assistant.flip(dims=[1])
    has_token = is_assistant.any(dim=1)  # [B]

    # For sequences with assistant token: find last position
    # For sequences without: use T-1 as fallback
    first_in_reversed = reversed_mask.float().argmax(dim=1)  # [B]
    last_position = T - 1 - first_in_reversed  # [B]

    # Handle sequences without assistant token
    fallback = torch.full((B,), T - 1, device=input_ids.device, dtype=torch.long)
    prompt_lengths = torch.where(has_token, last_position, fallback)

    return prompt_lengths