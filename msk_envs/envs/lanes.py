import torch

from msk_envs.geometry import UP_IDX, SIDE_IDX, FWD_IDX, build_axis, rotate_vec
from msk_envs.envs.utils.rewards import velocity_reward, joint_penalty, mid_lane_reward, has_fallen, \
    muscle_passive_penalty
from msk_envs.envs.utils.scene_settings import SceneSettings
from .base import MSKEnv


class LanesEnv(MSKEnv):
    """ The agent must move forward (+x) while facing [target_dir] and staying within its lane """

    target_dir: list[float]
    """ Horizontal direction the pelvis must face """
    angle_tolerance: float = 30.0
    """ Terminate when the pelvis faces more than this many degrees away from target_dir """
    lane_width: float = 0.6
    """ Terminate when either toe is further than this from the lane center """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.target_dir[UP_IDX] == 0.0, "Target direction should be horizontal only"

        # Precompute 2d target facing direction
        self.target_facing = torch.tensor(self.target_dir, device=self.device).unsqueeze(0)
        self.target_facing = self.target_facing[:, [FWD_IDX, SIDE_IDX]]
        self.target_facing = self.target_facing / torch.norm(self.target_facing, dim=1, keepdim=True)

        self.fwd_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device).unsqueeze(0)
        self.toes_ids = [self.body_id_lookup["toes_l"], self.body_id_lookup["toes_r"]]
        self.cos_angle_threshold = torch.cos(torch.deg2rad(torch.tensor(self.angle_tolerance, device=self.device)))
        # Observations exclude the forward root position, so the policy is invariant to progress along the lane
        self.obs_qpos_ids = self._qpos_ids_excluding("pelvis_tx")
        self.max_distance_reached = 0.0

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths
         2. Actuator activations
         3. Joint positions (q), excluding forward position
         4. Joint velocities (qv)
        """
        obs = torch.cat([
            self.muscle_activations,
            self.muscle_norm_fiber_lengths,
            self.actuator_activations,
            self.joint_positions[:, self.obs_qpos_ids],
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_vel = velocity_reward(self.body_velocities, self.root_id, FWD_IDX, linear=True)
        rew_mid_lane = mid_lane_reward(self.root_pos)

        # Joint passive penalty
        squared_penalties = False
        rew_muscle_passive = muscle_passive_penalty(
            self.muscle_passive_length_multiplier, threshold=0.1, squared=squared_penalties)
        rew_spring = joint_penalty(self.ufrc_spring, squared=squared_penalties)
        rew_damper = joint_penalty(self.ufrc_damper, squared=squared_penalties)
        rew_limit = joint_penalty(self.ufrc_limit, squared=squared_penalties)

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_mid_lane": rew_mid_lane.detach(),
            "rew_spring": rew_spring.detach(),
            "rew_damper": rew_damper.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
        }

    def _is_body_facing_direction(self, body_id):
        body_rot = self.body_rotations[:, body_id]
        body_fwd = rotate_vec(body_rot, self.fwd_axis)
        body_fwd = body_fwd[:, [FWD_IDX, SIDE_IDX]]  # Only care about x/z components
        body_fwd = body_fwd / torch.norm(body_fwd, dim=1, keepdim=True)
        body_fwd_dot_target = torch.sum(body_fwd * self.target_facing, dim=1)
        facing_direction = body_fwd_dot_target >= self.cos_angle_threshold
        return facing_direction.detach()

    def _get_lane_terminated(self) -> torch.Tensor:
        """ Fallen, a toe out of the lane, or the pelvis not facing the target direction """
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation)

        toes_out = torch.zeros_like(fallen, dtype=torch.bool)
        for body_idx in self.toes_ids:
            body_pos = self.body_positions[:, body_idx]
            toes_out |= (torch.abs(body_pos[:, SIDE_IDX]) > self.lane_width)

        not_facing_direction = ~self._is_body_facing_direction(self.root_id)
        return fallen | toes_out | not_facing_direction

    def _get_terminated(self):
        return self._get_lane_terminated().float().detach()

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
        max_forward = self.root_pos[:, FWD_IDX].max()
        self.max_distance_reached = max(max_forward, self.max_distance_reached)

    def additional_metrics(self) -> dict:
        return {
            "max_distance_reached": self.max_distance_reached,
        }


class SprintingEnv(LanesEnv):
    target_dir = build_axis(FWD_IDX, 1.0)


class BackPedalEnv(LanesEnv):
    """ Run forward while facing backwards """
    target_dir = build_axis(FWD_IDX, -1.0)
    angle_tolerance = 15.0


class SideShuffleEnv(LanesEnv):
    """ Face sideways and move forward without crossing legs """
    target_dir = build_axis(SIDE_IDX, 1.0)

    def _get_terminated(self):
        # The right toe went past the left toe
        left_toe_x = self.body_positions[:, self.toes_ids[0], FWD_IDX]
        right_toe_x = self.body_positions[:, self.toes_ids[1], FWD_IDX]
        right_toe_crossed = right_toe_x > left_toe_x

        return (self._get_lane_terminated() | right_toe_crossed).float().detach()


class HopEnv(LanesEnv):
    """ Hop forward on the right leg, keeping the left foot off the ground and the left knee bent """
    target_dir = build_axis(FWD_IDX, 1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.left_knee_qpos_idx = self.qpos_id_lookup["knee_angle_l"]
        self.left_knee_min_angle = -torch.deg2rad(torch.tensor(50.0, device=self.device))

    def _get_terminated(self):
        # Left toe is too low
        left_toe_height = self.body_positions[:, self.toes_ids[0], UP_IDX]
        left_toe_on_ground = left_toe_height < 0.2

        # Left knee is extended too much
        left_knee_angle = self.joint_positions[:, self.left_knee_qpos_idx]
        left_knee_extended = left_knee_angle > self.left_knee_min_angle

        return (self._get_lane_terminated() | left_toe_on_ground | left_knee_extended).float().detach()
