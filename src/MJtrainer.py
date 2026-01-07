# monkey_trainer.py
# Trainer that toggles MonkeyJump/MoE-LoRA router EMA updates on a schedule.

from typing import Any, Dict, Iterable, Optional

import torch
from transformers import Trainer


class MonkeyTrainer(Trainer):
    """
    Trainer that toggles router EMA updates on a schedule.

    Works with blocks patched by:
    - apply_monkeyjump_lora(...) [new: _mj_*]
    - apply_blockwise_lora_selective_byclass(...) [old: _moe_*]

    Args:
        step_interval: update every N steps
        stop_update_step: permanently stop updates at/after this step
        momentum: sets router momentum before forward (if provided)
        update_on: "micro" -> schedule over training_step calls
                   "optimizer" -> schedule over optimizer steps
    """

    def __init__(
        self,
        *args,
        step_interval: int = 10,
        stop_update_step: int = 1000,
        momentum: Optional[float] = None,
        update_on: str = "micro",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert step_interval >= 1
        assert stop_update_step >= 0
        assert update_on in ("micro", "optimizer")

        self.step_interval = int(step_interval)
        self.stop_update_step = int(stop_update_step)
        self.momentum_override = float(momentum) if momentum is not None else None
        self.update_on = update_on
        self._micro_step = 0

        if self.is_world_process_zero():
            print(
                f"[MonkeyTrainer] interval={self.step_interval}, stop_at={self.stop_update_step}, "
                f"update_on={self.update_on}, momentum={self.momentum_override}"
            )

        self._set_block_router_flag(False)

    def _is_patched_block(self, m: Any) -> bool:
        """Check if module is a patched MoE-LoRA / MonkeyJump block."""
        return (
            hasattr(m, "_sentence_router") or      # legacy
            hasattr(m, "_moe_router") or           # old MoE-LoRA
            hasattr(m, "_mj_router") or            # new MonkeyJump
            hasattr(m, "_mj_patched")              # new MonkeyJump flag
        )

    def _get_router(self, block: Any) -> Optional[Any]:
        """Get router from block, supporting all attribute names."""
        router = getattr(block, "_sentence_router", None)
        if router is None:
            router = getattr(block, "_moe_router", None)
        if router is None:
            router = getattr(block, "_mj_router", None)
        return router

    def _all_patched_blocks(self) -> Iterable[Any]:
        """Iterate modules that are patched blocks."""
        real = self.accelerator.unwrap_model(self.model)
        for m in real.modules():
            if self._is_patched_block(m):
                yield m

    def _set_block_router_flag(self, flag: bool) -> None:
        """Set _trainer_update_flag on patched blocks."""
        for blk in self._all_patched_blocks():
            setattr(blk, "_trainer_update_flag", bool(flag))

    def _set_router_momentum(self, momentum: float) -> None:
        """Set EMA momentum on each attached router."""
        for blk in self._all_patched_blocks():
            router = self._get_router(blk)
            if router is not None and hasattr(router, "momentum"):
                router.momentum = float(momentum)

    def _set_router_jitter(self, jitter_noise: float) -> None:
        """Set jitter noise on each attached router (MonkeyJump only)."""
        for blk in self._all_patched_blocks():
            router = self._get_router(blk)
            if router is not None and hasattr(router, "jitter_noise"):
                router.jitter_noise = float(jitter_noise)

    def _current_step_index(self) -> int:
        """Returns the scheduling step index."""
        if self.update_on == "optimizer":
            return int(self.state.global_step)
        return int(self._micro_step)

    def _should_update_now(self) -> bool:
        """Decide whether to enable updates for THIS call."""
        step = self._current_step_index()
        if self.update_on == "micro":
            if step >= self.stop_update_step * self.args.gradient_accumulation_steps:
                return False
        else:
            if step >= self.stop_update_step:
                return False
        return (step % self.step_interval) == 0

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.update_on == "micro":
            do_update = self._should_update_now()
            if self.momentum_override is not None:
                self._set_router_momentum(self.momentum_override)
            self._set_block_router_flag(do_update)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        if self.update_on == "micro":
            self._set_block_router_flag(False)

        self._micro_step += 1

        return loss.detach()

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Optional[Any] = None,
        on_tpu: bool = False,
        **kwargs,
    ):
        if self.update_on == "optimizer":
            do_update = self._should_update_now()
            if self.momentum_override is not None:
                self._set_router_momentum(self.momentum_override)
            self._set_block_router_flag(do_update)

        result = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, on_tpu, **kwargs)

        if self.update_on == "optimizer":
            self._set_block_router_flag(False)

        return result

    def evaluate(self, *args, **kwargs):
        self._set_block_router_flag(False)
        return super().evaluate(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        self._set_block_router_flag(False)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)