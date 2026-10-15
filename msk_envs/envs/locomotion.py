import math

import torch

from msk_envs.geometry import FWD_IDX, SIDE_IDX
from msk_envs.envs.utils.rewards import joint_penalty, has_fallen, muscle_passive_penalty
from .base import MSKEnv


class LocomotionEnv(MSKEnv):
    """ General locomotion: track a per-episode commanded horizontal velocity. """

    # Command sampling settings
    speed_min: float = 0.0
    speed_max: float = 4.0
    heading_range_deg: float = 120.0
    """ Half-cone about +forward that commanded headings are sampled from """
    stand_prob: float = 0.1
    """ Fraction of commands that are "stand still" (zero velocity), for robustness """
    track_sigma: float = 0.25

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.heading_range = math.radians(self.heading_range_deg)

        # Per-env commanded horizontal velocity (FWD, SIDE), world frame.
        # Sampled on every reset (initialized here so _get_obs works pre-reset).
        self.command_vel = torch.zeros((self.num_worlds, 2), device=self.device)

        # Obs excludes absolute horizontal root translation so the policy is
        # position-invariant (it can move in any direction). Keep everything else
        # including root height and orientation.
        self.obs_qpos_ids = self._qpos_ids_excluding("pelvis_tx", "pelvis_tz")

        # Logging
        self.last_track_err = 0.0

    def _sample_commands(self, reset_mask: torch.Tensor) -> None:
        """ Sample a new velocity command for the envs flagged in reset_mask. """
        n = int(reset_mask.sum())
        if n == 0:
            return
        speed = torch.rand(n, device=self.device) * (self.speed_max - self.speed_min) + self.speed_min
        angle = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.heading_range
        stand = torch.rand(n, device=self.device) < self.stand_prob
        speed = torch.where(stand, torch.zeros_like(speed), speed)
        cmd = torch.stack([speed * torch.cos(angle), speed * torch.sin(angle)], dim=1)  # (FWD, SIDE)
        self.command_vel[reset_mask] = cmd

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        self._sample_commands(reset_mask)

    def _root_vel_xy(self) -> torch.Tensor:
        """ World-frame horizontal (FWD, SIDE) linear velocity of the root. """
        return self.body_lin_velocities[:, self.root_id][:, [FWD_IDX, SIDE_IDX]]

    def _get_obs(self) -> torch.Tensor:
        """
        Observation space:
         0. Commanded horizontal velocity (FWD, SIDE)
         1. Muscle activations, fiber lengths
         2. Actuator activations
         3. Joint positions (q), excluding absolute horizontal root translation
         4. Joint velocities (qv)
        """
        obs = torch.cat([
            self.command_vel,
            self.muscle_activations,
            self.muscle_norm_fiber_lengths,
            self.actuator_activations,
            self.joint_positions[:, self.obs_qpos_ids],
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        # Velocity tracking: exponential of squared error to the command.
        vel_err = self._root_vel_xy() - self.command_vel
        rew_vel_track = torch.exp(-self.track_sigma * vel_err.pow(2).sum(dim=1))

        # Alive bonus (kept positive while upright; falling terminates the episode).
        rew_alive = torch.ones(self.num_worlds, device=self.device)

        # Penalties (shared with the lane environments).
        rew_limit = joint_penalty(self.ufrc_limit, squared=False)
        rew_muscle_passive = muscle_passive_penalty(
            self.muscle_passive_length_multiplier, threshold=0.1, squared=False)

        self.reward_dict = {
            "rew_vel_track": rew_vel_track.detach(),
            "rew_alive": rew_alive.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
        }

    def _get_terminated(self):
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation)
        return fallen.float().detach()

    def update_metrics(self) -> None:
        # Mean speed-tracking error this step (norm of velocity error).
        vel_err = self._root_vel_xy() - self.command_vel
        self.last_track_err = vel_err.norm(dim=1).mean().item()

    def additional_metrics(self) -> dict:
        return {
            "vel_track_err": self.last_track_err,
            "command_speed_mean": self.command_vel.norm(dim=1).mean().item(),
        }
