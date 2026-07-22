"""
torch <-> JAX bridging for driving the reference ``FlowExp`` algorithm from
    the torch/warp env
"""

from __future__ import annotations

from msk_envs.train.qflex_ref import jax_compat  # noqa: F401  (installs jax.core shim)

import jax
import jax.numpy as jnp
import numpy as np
import torch

from relax.utils.experience import Experience


def torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Convert a torch tensor to a JAX array, zero-copy on GPU when possible."""
    t = t.contiguous()
    try:
        return jax.dlpack.from_dlpack(t)
    except Exception:
        return jnp.asarray(t.detach().cpu().numpy())


def jax_to_torch(a: jax.Array, device: torch.device) -> torch.Tensor:
    """Convert a JAX array to a torch tensor on ``device``, zero-copy when possible."""
    try:
        out = torch.utils.dlpack.from_dlpack(a)
    except Exception:
        out = torch.as_tensor(np.asarray(a))
    return out.to(device=device, dtype=torch.float32)


def build_experience(batch, device: torch.device) -> Experience:
    """Build ``Experience`` (JAX arrays) from a sampled minibatch. """
    obs = batch["observations"]
    actions = batch["actions"]
    next_obs = batch["next"]["observations"]
    rewards = batch["next"]["rewards"]
    dones = batch["next"]["dones"].bool()
    truncations = batch["next"]["truncations"].bool()
    terminated = dones & ~truncations  # [B]

    return Experience(
        obs=torch_to_jax(obs.float()),
        action=torch_to_jax(actions.float()),
        reward=torch_to_jax(rewards.float()),
        done=torch_to_jax(terminated.float()).astype(jnp.bool_),
        next_obs=torch_to_jax(next_obs.float()),
    )
