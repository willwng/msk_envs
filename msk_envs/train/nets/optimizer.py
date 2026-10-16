import math
from collections import defaultdict
from typing import Iterable, Literal

import torch
import torch.nn as nn
import torch.optim as optim


def make_optimizer(
        params: nn.Module | Iterable[nn.Parameter],
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        optimizer: Literal["adamw", "muon"] = "adamw",
        eps: float = 1e-8,
        fused: bool = True,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
) -> optim.Optimizer:
    """
    AdamW over all parameters, or Muon for the weight matrices and AdamW (same lr, betas, eps, weight decay) for the
    rest. Muon's learning rate is scaled to match AdamW's update size, so AdamW-tuned learning rates carry over.
    """
    params = list(params.parameters() if isinstance(params, nn.Module) else params)
    if optimizer == "adamw":
        return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=fused or None)
    return MuonAdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=muon_momentum,
                     nesterov=muon_nesterov, ns_steps=muon_ns_steps, adjust_lr_fn="match_rms_adamw")


def _adjust_lr(lr: float, adjust_lr_fn: str | None, shape: torch.Size) -> float:
    """ Muon's per-shape learning rate (as torch.optim.Muon) """
    rows, cols = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1, rows / cols))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    return lr


def _batched_newton_schulz(updates: torch.Tensor, coefficients: tuple[float, float, float], steps: int,
                           eps: float) -> torch.Tensor:
    """ Orthogonalize each matrix of a [n, rows, cols] stack (as torch.optim.Muon does one at a time, in bfloat16) """
    a, b, c = coefficients
    x = updates.bfloat16()
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.mT
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, gram_update, x, beta=a)
    return x.mT if transpose else x


class MuonAdamW(optim.Optimizer):
    """
    Muon for 2D parameters (weight matrices) and AdamW for the others (biases, norms, ...), as one optimizer, so it
    works with learning-rate schedulers, GradScaler and a single state dict.

    The Muon update is torch.optim.Muon's (same results and state), computed in batches: the Newton-Schulz
    orthogonalization runs once per group of same-shaped matrices instead of once per matrix. The AdamW update is
    torch.optim.AdamW's (without amsgrad), with foreach ops.
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
            ns_eps: float = 1e-7,
            ns_steps: int = 5,
            adjust_lr_fn: str | None = "match_rms_adamw",
    ):
        params = list(params)
        if params and isinstance(params[0], dict):
            groups = params  # explicit groups, each with "use_muon"
        else:
            groups = [{"params": [p for p in params if p.ndim == 2], "use_muon": True},
                      {"params": [p for p in params if p.ndim != 2], "use_muon": False}]
            groups = [g for g in groups if g["params"]]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_coefficients=ns_coefficients, ns_eps=ns_eps, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(groups, defaults)
        for group in self.param_groups:
            if group["use_muon"] and any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon parameter groups may only contain 2D parameters")

    @torch.compiler.disable  # runs eagerly when called from compiled code (the AdamW step count is a Python int)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if params:
                (self._muon_step if group["use_muon"] else self._adamw_step)(group, params)
        return loss

    def _muon_step(self, group: dict, params: list[torch.Tensor]) -> None:
        grads = [p.grad for p in params]
        bufs = []
        for p in params:
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            bufs.append(state["momentum_buffer"])

        momentum = group["momentum"]
        torch._foreach_lerp_(bufs, grads, 1 - momentum)
        updates = torch._foreach_lerp(grads, bufs, momentum) if group["nesterov"] else bufs

        lr = group["lr"]
        if group["weight_decay"] != 0:
            torch._foreach_mul_(params, 1 - lr * group["weight_decay"])

        # Orthogonalize each group of same-shaped matrices together
        by_shape = defaultdict(list)
        for i, p in enumerate(params):
            by_shape[p.shape].append(i)
        for shape, idx in by_shape.items():
            ortho = _batched_newton_schulz(torch.stack([updates[i] for i in idx]), group["ns_coefficients"],
                                           group["ns_steps"], group["ns_eps"])
            torch._foreach_add_([params[i] for i in idx], list(ortho.unbind(0)),
                                alpha=-_adjust_lr(lr, group["adjust_lr_fn"], shape))

    def _adamw_step(self, group: dict, params: list[torch.Tensor]) -> None:
        grads = [p.grad for p in params]
        exp_avgs, exp_avg_sqs = [], []
        for p in params:
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["step"] += 1
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
        step = self.state[params[0]]["step"]  # all parameters of a group step together

        lr, (beta1, beta2), eps = group["lr"], group["betas"], group["eps"]
        if group["weight_decay"] != 0:
            torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denom, math.sqrt(bias_correction2))
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-lr / bias_correction1)
