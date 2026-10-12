"""Small optimizer-side helpers shared by training and profiling entrypoints."""
from __future__ import annotations

from collections.abc import Iterable
from collections import defaultdict

import torch
from torch import Tensor


_GRADIENT_HEALTH_EXECUTIONS = frozenset({"auto", "foreach", "amp_foreach"})


def resolve_gradient_health_execution(
    execution: str,
    optimizer: torch.optim.Optimizer,
) -> str:
    """Resolve the finite-gradient guard without changing optimizer semantics."""
    if execution not in _GRADIENT_HEALTH_EXECUTIONS:
        raise ValueError(
            "gradient health execution must be auto, foreach, or amp_foreach"
        )
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    amp_supported = bool(
        parameters
        and all(parameter.device.type == "cuda" for parameter in parameters)
        and hasattr(torch, "_amp_foreach_non_finite_check_and_unscale_")
        and getattr(optimizer, "_step_supports_amp_scaling", False)
    )
    if execution == "auto":
        return "amp_foreach" if amp_supported else "foreach"
    if execution == "amp_foreach" and not amp_supported:
        raise RuntimeError(
            "amp_foreach requires CUDA and an optimizer with PyTorch AMP "
            "found_inf support"
        )
    return execution


def _finite_gradient_check(grads: list[Tensor]) -> Tensor:
    """Check all gradients with one compact reduction per device group."""
    if not grads:
        return torch.ones((), dtype=torch.bool)
    devices = {gradient.device for gradient in grads}
    if len(devices) == 1 and hasattr(torch, "_foreach_norm"):
        try:
            # Infinity norms detect NaN/Inf without constructing a flattened
            # copy of every gradient. The list of scalar norms is tiny compared
            # with the parameter tensors and stays on the accelerator.
            norms = torch._foreach_norm(grads, float("inf"))
            # ``foreach_norm`` already returns one scalar per gradient in the
            # gradient's device/dtype (BF16 on the production H20 path).
            # Casting every scalar to FP32 creates one tiny ``aten::to`` launch
            # per parameter; stacking the scalars and checking finiteness is
            # dtype agnostic and preserves the same NaN/Inf decision.
            return torch.stack(norms).isfinite().all()
        except (RuntimeError, TypeError):
            pass
    finite = torch.ones((), dtype=torch.bool, device=grads[0].device)
    for gradient in grads:
        finite = finite & torch.isfinite(gradient).all()
    return finite


def gradient_health(
    parameters: Iterable[torch.nn.Parameter],
    *,
    compute_norm: bool,
    previous_norm: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(norm, finite)`` while sampling the expensive L2 norm.

    BrainWM uses ``clip_grad_norm_(inf)`` as a finite-gradient guard; an
    infinite max norm never scales gradients. On unsampled steps we therefore
    retain that guard with an L-infinity foreach reduction and reuse the last
    reported L2 norm for telemetry. The returned ``finite`` scalar still needs
    to be inspected by the caller before ``optimizer.step``.
    """
    parameter_list = tuple(parameters)
    grads = [parameter.grad for parameter in parameter_list if parameter.grad is not None]
    if compute_norm:
        norm = torch.nn.utils.clip_grad_norm_(parameter_list, float("inf"))
        return norm, torch.isfinite(norm)
    finite = _finite_gradient_check(grads)
    if previous_norm is not None:
        norm = previous_norm.detach()
    elif grads:
        norm = grads[0].new_zeros(())
    else:
        norm = torch.zeros(())
    return norm, finite


class GradientHealthMonitor:
    """Finite-gradient checking with an asynchronous fused-optimizer path.

    ``amp_foreach`` reuses the same PyTorch multi-tensor primitive used by
    ``torch.amp.GradScaler``.  The inverse scale is exactly one for BrainWM's
    native BF16 training, so gradients are unchanged.  PyTorch's fused AdamW
    consumes the device-resident ``found_inf`` scalar and skips a bad update
    without requiring a host synchronization first.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        *,
        execution: str = "auto",
    ) -> None:
        self.parameters = tuple(parameters)
        self.execution = resolve_gradient_health_execution(execution, optimizer)
        self.last_norm: Tensor | None = None
        self._found_inf: Tensor | None = None
        self._inverse_scale: Tensor | None = None
        if self.execution == "amp_foreach":
            device = self.parameters[0].device
            self._found_inf = torch.zeros((), device=device, dtype=torch.float32)
            self._inverse_scale = torch.ones((), device=device, dtype=torch.float32)

    def _amp_check(self, gradients: list[Tensor]) -> Tensor:
        assert self._found_inf is not None and self._inverse_scale is not None
        self._found_inf.zero_()
        grouped: dict[tuple[torch.device, torch.dtype], list[Tensor]] = defaultdict(list)
        for gradient in gradients:
            value = gradient._values() if gradient.is_sparse else gradient
            grouped[(value.device, value.dtype)].append(value)
        for values in grouped.values():
            torch._amp_foreach_non_finite_check_and_unscale_(
                values, self._found_inf, self._inverse_scale,
            )
        return self._found_inf.eq(0)

    def check(self, *, compute_norm: bool) -> tuple[Tensor, Tensor]:
        """Return the sampled L2 norm and a device-resident finite flag."""
        gradients = [
            parameter.grad
            for parameter in self.parameters
            if parameter.grad is not None
        ]
        if compute_norm:
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, float("inf"))
            finite = torch.isfinite(norm)
            if self.execution == "amp_foreach":
                assert self._found_inf is not None
                self._found_inf.zero_()
                torch.maximum(
                    self._found_inf,
                    (~finite).to(self._found_inf.dtype),
                    out=self._found_inf,
                )
                finite = self._found_inf.eq(0)
            self.last_norm = norm.detach()
            return norm, finite
        if self.execution == "amp_foreach":
            finite = self._amp_check(gradients)
        else:
            finite = _finite_gradient_check(gradients)
        if self.last_norm is not None:
            norm = self.last_norm
        elif gradients:
            norm = gradients[0].new_zeros(())
        else:
            norm = torch.zeros(())
        return norm, finite

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        loss: Tensor,
        finite_gradients: Tensor,
    ) -> Tensor:
        """Run an update, skipping it on-device if loss or gradients are bad."""
        finite_loss = torch.isfinite(loss.detach())
        if self.execution != "amp_foreach":
            if not bool(finite_loss):
                raise FloatingPointError("nonfinite loss")
            if not bool(finite_gradients):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            return finite_loss & finite_gradients

        assert self._found_inf is not None
        torch.maximum(
            self._found_inf,
            (~finite_loss).to(self._found_inf.dtype),
            out=self._found_inf,
        )
        torch.maximum(
            self._found_inf,
            (~finite_gradients).to(self._found_inf.dtype),
            out=self._found_inf,
        )
        optimizer.grad_scale = None
        optimizer.found_inf = self._found_inf
        try:
            optimizer.step()
        finally:
            del optimizer.grad_scale
            del optimizer.found_inf
        return self._found_inf.eq(0)
