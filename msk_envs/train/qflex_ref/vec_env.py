from __future__ import annotations

from msk_envs.train.qflex_ref import jax_compat  # noqa: F401  (installs jax.core shim first)

import numpy as np
import torch
from gymnasium.spaces import Box

from relax.env.vector import VectorEnv


class MSKVectorEnv(VectorEnv):
    def __init__(self, msk_env, device: torch.device):
        self.env = msk_env
        self.device = device
        self.num_envs = msk_env.num_worlds
        self.obs_dim = msk_env.num_obs()
        self.act_dim = msk_env.num_actions()
        low, high = msk_env.action_range

        self.single_observation_space = Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.single_action_space = Box(low, high, (self.act_dim,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (self.num_envs, self.obs_dim), np.float32)
        self.action_space = Box(low, high, (self.num_envs, self.act_dim), np.float32)

        # Current obs after the msk env's internal auto-reset. None until first reset.
        self._post_reset_obs: np.ndarray | None = None

    def reset(self, *, seed=None, options=None):
        # First reset resets all envs; later resets just surface the obs the msk
        # env already produced (it auto-resets done envs inside step()).
        if self._post_reset_obs is None:
            obs = self.env.reset()
            self._post_reset_obs = obs.detach().to(torch.float32).cpu().numpy()
        return self._post_reset_obs.copy(), {}

    def step(self, action: np.ndarray):
        actions = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        next_obs, rewards, terminated, truncated, info = self.env.step(actions)

        term = terminated.reshape(-1).bool()
        trunc = truncated.reshape(-1).bool()
        done = term | trunc
        # Terminal obs for done envs (pre-reset); stepped obs otherwise.
        terminal_obs = torch.where(done[:, None], info["final_observation"], next_obs)

        # Stash the post-(internal-)reset obs for the trainer's follow-up reset().
        self._post_reset_obs = next_obs.detach().to(torch.float32).cpu().numpy()

        return (
            terminal_obs.detach().to(torch.float32).cpu().numpy(),
            rewards.detach().to(torch.float64).cpu().numpy(),
            term.cpu().numpy(),
            trunc.cpu().numpy(),
            {},
        )

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self
