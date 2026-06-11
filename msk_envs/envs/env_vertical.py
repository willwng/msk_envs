import torch

from msk_envs.utils.global_params import UP_IDX, build_axis
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import joint_penalty, has_fallen, velocity_reward, tolerance
from .env_base import MSKEnv
from .env_config import EnvConfig


class VerticalEnv(MSKEnv):
    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool,
    ):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, requires_visuals=requires_visuals,
                         cuda_graph=cuda_graph)
        self.hand_r_id = self.body_id_lookup["hand_r"]
        self.hand_l_id = self.body_id_lookup["hand_l"]

        self.up_axis = torch.tensor(build_axis(UP_IDX, 1.0), device=self.device).unsqueeze(0)

        self.jump_height = 3.05  # NBA rim height
        self.max_velocity = 5.0
        self.max_height_reached = 0.0
        return

    def _compute_raw_reward_dict(self):
        # Based on humenv
        hand_r_height = self.body_positions[:, self.hand_r_id, UP_IDX]
        hand_l_height = self.body_positions[:, self.hand_l_id, UP_IDX]
        # hand_height = (hand_r_height + hand_l_height) / 2.0
        hand_height = hand_r_height
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
            self.muscle_fiber_lengths,
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
        hand_r_height = self.body_positions[:, self.hand_r_id, UP_IDX]
        hand_l_height = self.body_positions[:, self.hand_l_id, UP_IDX]
        # hand_height = (hand_r_height + hand_l_height) / 2.0
        hand_height = hand_r_height
        max_hand_height = hand_height.max()
        self.max_height_reached = max(max_hand_height, self.max_height_reached)
        return

    def additional_metrics(self) -> dict:
        return {
            "max_height_reached": self.max_height_reached,
        }
