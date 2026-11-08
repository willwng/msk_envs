import torch

from msk_envs.geometry import SIDE_IDX, FWD_IDX, build_axis, rotate_vec
from msk_envs.envs.utils.rewards import joint_penalty, has_fallen
from msk_envs.envs.utils.scene_settings import SceneSettings
from .base import MSKEnv


class CurvedTrackEnv(MSKEnv):
    """
    Run around the bend of a standard track, staying within a lane. The bend turns right (clockwise seen from above)
    or left (counterclockwise, as on a real track), drawn at random each episode; the observation tells which
    """
    angle_tolerance: float = 30.0
    """ Terminate when the pelvis faces more than this many degrees away from the track tangent """
    lane_index: int = 0
    """ 0 for Lane 1 """
    direction: str = "random"
    """ "random" (each episode), "clockwise" (turning right) or "counterclockwise" (turning left) """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.toes_ids = [self.body_id_lookup["toes_l"], self.body_id_lookup["toes_r"]]

        # Olympic Track Specs
        self.lane_width = 1.22
        self.inner_radius = 36.5

        # Target radius is the center of the chosen lane
        self.target_radius = (self.inner_radius + (self.lane_index * self.lane_width) + (self.lane_width / 2))

        # The agent starts at (x=0, z=0) facing +x, so the bend's center is at z = sign * R, where sign (per world) is
        # +1 for a clockwise bend (turning right, toward +z) and -1 for a counterclockwise one (turning left). The
        # geometry below is computed for the clockwise bend, in coordinates mirrored in z by sign
        assert self.direction in ("random", "clockwise", "counterclockwise"), self.direction
        self.curve_sign = torch.ones(self.num_worlds, device=self.device)
        self.curve_center = torch.zeros((self.num_worlds, 3), device=self.device)
        self._sample_direction(torch.ones(self.num_worlds, dtype=torch.bool, device=self.device))

        self.fwd_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device, dtype=torch.float32).unsqueeze(0)
        self.cos_angle_threshold = torch.cos(torch.deg2rad(torch.tensor(self.angle_tolerance, device=self.device)))
        self.max_distance_reached = 0.0

    def _sample_direction(self, reset_mask: torch.Tensor) -> None:
        n = int(reset_mask.sum())
        if self.direction == "random":
            sign = torch.randint(0, 2, (n,), device=self.device).float() * 2.0 - 1.0
        else:
            sign = torch.full((n,), 1.0 if self.direction == "clockwise" else -1.0, device=self.device)
        self.curve_sign[reset_mask] = sign
        self.curve_center[reset_mask, SIDE_IDX] = sign * self.target_radius

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        self._sample_direction(reset_mask)

    def _bend_coords(self, pos) -> torch.Tensor:
        """ [num_worlds, 2] (x, z) of positions [num_worlds, 3] from the bend's center, z mirrored to a clockwise bend """
        rel = (pos - self.curve_center)[:, [FWD_IDX, SIDE_IDX]]
        return torch.stack([rel[:, 0], self.curve_sign * rel[:, 1]], dim=1)

    def _get_obs(self) -> torch.Tensor:
        obs = torch.cat([
            self.curve_sign.view(self.num_worlds, 1),  # +1 clockwise (turn right), -1 counterclockwise (turn left)
            self._get_angle_progress().view(self.num_worlds, 1),
            self.muscle_activations,
            self.muscle_norm_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _get_lane_deviation(self, body_pos) -> torch.Tensor:
        """ Calculates the deviation from center of lane for a body """
        dist_from_center = torch.norm(self._bend_coords(body_pos), dim=1)
        return dist_from_center - self.target_radius

    def _get_target_dir(self):
        """ The (x, z) direction of the track (its tangent, in the running direction) at the agent's root """
        pos_2d = self._bend_coords(self.root_pos)
        radial_vec = pos_2d / (torch.norm(pos_2d, dim=1, keepdim=True) + 1e-6)  # from the center to the agent
        # Tangent of the clockwise bend, then mirrored back
        return torch.stack([-radial_vec[:, 1], self.curve_sign * radial_vec[:, 0]], dim=1)

    def _compute_raw_reward_dict(self):
        target_dir = self._get_target_dir()
        lane_deviation = self._get_lane_deviation(self.root_pos)

        # Velocity Reward: project actual 2D velocity onto the tangent target
        root_vel_2d = self.body_lin_velocities[:, self.root_id][:, [FWD_IDX, SIDE_IDX]]
        rew_vel = torch.sum(root_vel_2d * target_dir, dim=1)

        rew_mid_lane = torch.exp(-2.5 * lane_deviation.pow(2))
        rew_limit = joint_penalty(self.ufrc_limit, squared=False)

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_mid_lane": rew_mid_lane.detach(),
            "rew_limit": rew_limit.detach(),
        }

    def _is_body_facing_direction(self, body_id):
        target_dir = self._get_target_dir()
        body_rot = self.body_rotations[:, body_id]
        body_fwd = rotate_vec(body_rot, self.fwd_axis)
        body_fwd_2d = body_fwd[:, [FWD_IDX, SIDE_IDX]]
        body_fwd_2d = body_fwd_2d / (torch.norm(body_fwd_2d, dim=1, keepdim=True) + 1e-6)
        facing_dot = torch.sum(body_fwd_2d * target_dir, dim=1)
        return facing_dot >= self.cos_angle_threshold

    def _get_terminated(self):
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation)

        # Termination if toes leave the lane width
        toes_out = torch.zeros_like(fallen, dtype=torch.bool)
        for body_idx in self.toes_ids:
            body_pos = self.body_positions[:, body_idx]
            lane_deviation = self._get_lane_deviation(body_pos)
            toes_out |= (torch.abs(lane_deviation) > (self.lane_width / 2))

        not_facing = ~self._is_body_facing_direction(self.root_id)
        return (fallen | toes_out | not_facing).float().detach()

    def _get_angle_progress(self):
        """ Angle (rad) run around the bend since the start, for either direction """
        pos_2d = self._bend_coords(self.root_pos)
        angle = torch.atan2(pos_2d[:, 1], pos_2d[:, 0])  # z, x

        # Reference angle at start position:
        # start is (x=0, z=0), center is (0, +R) (mirrored)
        # so initial vector is (0, -R) -> angle = -pi/2
        start_angle = -torch.pi / 2
        angle_progress = angle - start_angle
        angle_progress = (angle_progress + torch.pi) % (2 * torch.pi) - torch.pi
        return angle_progress

    def _get_curve_progress(self):
        arc_length = self.target_radius * self._get_angle_progress()
        return arc_length

    def update_metrics(self) -> None:
        max_forward = self._get_curve_progress().max()
        self.max_distance_reached = max(max_forward, self.max_distance_reached)

    def additional_metrics(self) -> dict:
        return {
            "max_distance_reached": self.max_distance_reached,
        }

    def scene_settings(self, world_id: int = 0) -> SceneSettings:
        return SceneSettings(
            lanes=False,
            curve=True,
            curve_direction=float(self.curve_sign[world_id]),
            axes=False,
        )
