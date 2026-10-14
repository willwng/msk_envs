import torch

from msk_envs.geometry import UP_IDX, build_axis
from msk_envs.envs.utils.rewards import joint_penalty, has_fallen, velocity_reward, tolerance
from .base import MSKEnv


class VerticalEnv(MSKEnv):
    """ Jump to reach the right hand to rim height within the first second """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hand_r_id = self.body_id_lookup["hand_r"]

        self.up_axis = torch.tensor(build_axis(UP_IDX, 1.0), device=self.device).unsqueeze(0)

        self.jump_height = 3.05  # NBA rim height
        self.max_velocity = 5.0
        self.max_height_reached = 0.0

    def _compute_raw_reward_dict(self):
        # Based on humenv
        hand_height = self.body_positions[:, self.hand_r_id, UP_IDX]
        center_of_mass_velocity = velocity_reward(self.body_velocities, self.root_id, UP_IDX, linear=True)

        jumping = tolerance(
            hand_height,
            bounds=(self.jump_height, self.jump_height + 0.1),
            margin=self.jump_height,
            value_at_margin=0.01,
        )
        up_velocity = tolerance(
            center_of_mass_velocity,
            bounds=(self.max_velocity, torch.inf),
            margin=self.max_velocity,
            value_at_margin=0.0,
        )

        rew_jump = jumping * up_velocity
        times_up = self.time > 1.0  # Only reward for first second (one jump)
        rew_jump[times_up] = 0.0

        rew_limit = joint_penalty(self.ufrc_limit, squared=False)
        rew_alive = torch.ones_like(rew_limit)

        self.reward_dict = {
            "rew_jump": rew_jump,
            "rew_limit": rew_limit.detach(),
            "rew_alive": rew_alive.detach(),
        }

    def _get_obs(self) -> torch.Tensor:
        obs = torch.cat([
            self.time.view(self.num_worlds, 1),
            self.muscle_activations,
            self.muscle_norm_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _get_terminated(self):
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation, min_root=0.3)
        terminated = fallen.float()
        return terminated.detach()

    def update_metrics(self) -> None:
        max_hand_height = self.body_positions[:, self.hand_r_id, UP_IDX].max()
        self.max_height_reached = max(max_hand_height, self.max_height_reached)

    def additional_metrics(self) -> dict:
        return {
            "max_height_reached": self.max_height_reached,
        }
