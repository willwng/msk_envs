import math
from collections import defaultdict

import torch
import torch.optim as optim


def make_optimizer(model, lr: float, betas: tuple[float, float], weight_decay: float):
    """ Fused AdamW over all of [model]'s parameters """
    return optim.AdamW(
        list(model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        fused=True,
        betas=betas,
    )


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


class BatchedMuon(optim.Optimizer):
    """
    torch.optim.Muon with the same update and state, computed in batches: momentum and parameter updates use foreach
    ops, and the Newton-Schulz orthogonalization runs once for each group of same-shaped matrices instead of once per
    matrix. Only 2D parameters are supported.
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            weight_decay: float = 0.1,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
            eps: float = 1e-7,
            ns_steps: int = 5,
            adjust_lr_fn: str | None = None,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_coefficients=ns_coefficients, eps=eps, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon only supports 2D parameters, got a parameter of size {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
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
                                               group["ns_steps"], group["eps"])
                torch._foreach_add_([params[i] for i in idx], list(ortho.unbind(0)),
                                    alpha=-_adjust_lr(lr, group["adjust_lr_fn"], shape))
        return loss
