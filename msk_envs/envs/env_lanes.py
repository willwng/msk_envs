import torch

from msk_envs.utils.global_params import UP_IDX, SIDE_IDX, FWD_IDX, build_axis
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import velocity_reward, joint_penalty, mid_lane_reward, has_fallen
from .env_base import MSKEnv
from .env_config import EnvConfig
from ..utils.scene_settings import SceneSettings


class LanesEnv(MSKEnv):
    """ Represents an environment where the agent must face a specific direction and stay within lanes """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool,
            target_dir: list[float],
            angle_tolerance: float = 30.0,
            lane_width: float = 0.6,
    ):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, requires_visuals=requires_visuals,
                         cuda_graph=cuda_graph)
        assert target_dir[UP_IDX] == 0.0, "Target direction should be horizontal only"

        # Precompute 2d target facing direction
        self.target_facing = torch.tensor(target_dir, device=self.device).unsqueeze(0)
        self.target_facing = self.target_facing[:, [FWD_IDX, SIDE_IDX]]
        self.target_facing = self.target_facing / torch.norm(self.target_facing, dim=1, keepdim=True)

        self.fwd_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device).unsqueeze(0)
        self.toes_ids = [self.body_id_lookup["toes_l"], self.body_id_lookup["toes_r"]]
        self.cos_angle_threshold = torch.cos(torch.deg2rad(torch.tensor(angle_tolerance, device=self.device)))
        self.lane_width = lane_width
        self.max_distance_reached = 0.0
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
         5. Relative body positions and rotations (wrst root), ignore ground
        """
        # Exclude root x coordinate (forward along lane)
        root_x_qpos_id = self.qpos_id_lookup["pelvis_tx"]
        joint_positions_without_x = torch.cat((
            self.joint_positions[:, :root_x_qpos_id],
            self.joint_positions[:, root_x_qpos_id + 1:]),
            dim=1)
        obs = torch.cat([
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            joint_positions_without_x,
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_vel = velocity_reward(self.body_velocities, self.root_id, FWD_IDX, linear=True)
        rew_mid_lane = mid_lane_reward(self.root_pos)

        # Joint passive penalty
        squared_penalties = False
        rew_spring = joint_penalty(self.ufrc_spring, squared=squared_penalties)
        rew_damper = joint_penalty(self.ufrc_damper, squared=squared_penalties)
        rew_limit = joint_penalty(self.ufrc_limit, squared=squared_penalties)

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_mid_lane": rew_mid_lane.detach(),
            "rew_spring": rew_spring.detach(),
            "rew_damper": rew_damper.detach(),
            "rew_limit": rew_limit.detach(),
        }

    def _is_body_facing_direction(self, body_id):
        body_rot = self.body_rotations[:, body_id]
        body_fwd = rotate_vec(body_rot, self.fwd_axis)
        body_fwd = body_fwd[:, [FWD_IDX, SIDE_IDX]]  # Only care about x/z components
        body_fwd = body_fwd / torch.norm(body_fwd, dim=1, keepdim=True)
        body_fwd_dot_target = torch.sum(body_fwd * self.target_facing, dim=1)
        facing_direction = body_fwd_dot_target >= self.cos_angle_threshold
        return facing_direction.detach()

    def _get_terminated(self):
        # Has fallen
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation)

        # Any of the toes are out of the lanes
        toes_out = torch.zeros_like(fallen, dtype=torch.bool)
        for body_idx in self.toes_ids:
            body_pos = self.body_positions[:, body_idx]
            toes_out |= (torch.abs(body_pos[:, SIDE_IDX]) > self.lane_width)

        # Pelvis no longer facing target direction (within N degrees)
        pelvis_facing_direction = self._is_body_facing_direction(self.root_id)
        not_facing_direction = ~pelvis_facing_direction

        terminated = (fallen | toes_out | not_facing_direction).float()
        return terminated.detach()

    def scene_settings(self) -> SceneSettings:
        if self.ground_rotation[3] == 1:  # Flat ground
            return SceneSettings(
                lanes=True,
                lane_width=self.lane_width,
                meter_markers=True,
                axes=False
            )
        else:
            return SceneSettings(
                lanes=False,
                meter_markers=False,
                axes=False
            )

    def update_metrics(self) -> None:
        # Update max distance reached
        max_forward = self.root_pos[:, FWD_IDX].max()
        self.max_distance_reached = max(max_forward, self.max_distance_reached)
        return

    def additional_metrics(self) -> dict:
        return {
            "max_distance_reached": self.max_distance_reached,
        }
