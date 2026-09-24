"""
According to worldathletics:
    The distance between each hurdle in the men’s 110m hurdles is 9.14m (30 feet).
    The exceptions are the first and last hurdles:
        there is a distance of 13.72m (45 feet) to the first hurdle
        and 14.02m (46 feet) from the final hurdle to the finish.
    The height of each hurdle in the men’s 110m hurdles is 106.7cm
"""

import torch
import bolt
import warp as wp

from msk_envs.utils.global_params import FWD_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv

HURDLE_START_HEIGHT = 0.7
HURDLE_END_HEIGHT = 1.0
HURDLE_POSITIONS = [13.72 + i * 9.14 for i in range(10)]
HURDLE_HEIGHTS = [HURDLE_START_HEIGHT + i * (HURDLE_END_HEIGHT - HURDLE_START_HEIGHT) / 9 for i in range(10)]


class HurdlesEnv(LanesEnv):

    def _add_colliders(self, env_config: EnvConfig) -> None:
        colliders = self.load_result.colliders

        hurdle_width = 0.7
        hurdle_thickness = 0.035
        for i, (hurdle_position, hurdle_height) in enumerate(zip(HURDLE_POSITIONS, HURDLE_HEIGHTS)):
            hurdle_top = bolt.UserGeomData(
                name=f"hurdle_top_{hurdle_position}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.CAPSULE,
                transform=wp.transform(wp.vec3(hurdle_position, hurdle_height, 0.0), wp.quat_identity(dtype=float)),
                size=wp.vec3(hurdle_thickness * 2.0, hurdle_width, hurdle_thickness * 2.0),
                priority=9,
            )
            hurdle_side1 = bolt.UserGeomData(
                name=f"hurdle_side1_{hurdle_position}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.CAPSULE,
                transform=wp.transform(
                    wp.vec3(hurdle_position, hurdle_height / 2.0, -hurdle_width),
                    wp.quat(0.707, 0.0, 0.0, 0.707)
                ),
                size=wp.vec3(hurdle_thickness, hurdle_height / 2.0, hurdle_thickness),
                priority=9,
            )
            hurdle_side2 = bolt.UserGeomData(
                name=f"hurdle_side2_{hurdle_position}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.CAPSULE,
                transform=wp.transform(
                    wp.vec3(hurdle_position, hurdle_height / 2.0, hurdle_width),
                    wp.quat(0.707, 0.0, 0.0, 0.707)
                ),
                size=wp.vec3(hurdle_thickness, hurdle_height / 2.0, hurdle_thickness),
                priority=9,
            )
            hurdle_bot1 = bolt.UserGeomData(
                name=f"hurdle_bot1_{hurdle_position}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.CAPSULE,
                transform=wp.transform(
                    wp.vec3(hurdle_position - hurdle_height / 2.0, hurdle_thickness, -hurdle_width),
                    wp.quat(0.0, 0.70710678, 0.0, 0.70710678)
                ),
                size=wp.vec3(hurdle_thickness, hurdle_height / 2.0, hurdle_thickness),
                priority=9,
            )
            hurdle_bot2 = bolt.UserGeomData(
                name=f"hurdle_bot2_{hurdle_position}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.CAPSULE,
                transform=wp.transform(
                    wp.vec3(hurdle_position - hurdle_height / 2.0, hurdle_thickness, hurdle_width),
                    wp.quat(0.0, 0.70710678, 0.0, 0.70710678)
                ),
                size=wp.vec3(hurdle_thickness, hurdle_height / 2.0, hurdle_thickness),
                priority=9,
            )
            colliders.append(bolt.convert_user_collider(hurdle_top))
            colliders.append(bolt.convert_user_collider(hurdle_side1))
            colliders.append(bolt.convert_user_collider(hurdle_side2))
            colliders.append(bolt.convert_user_collider(hurdle_bot1))
            colliders.append(bolt.convert_user_collider(hurdle_bot2))
        return

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            cuda_graph: bool
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, 1.0),
        )
        self.hurdle_positions = torch.tensor(HURDLE_POSITIONS, device=self.device, dtype=torch.float32)
        self.hurdle_heights = torch.tensor(HURDLE_HEIGHTS, device=self.device, dtype=torch.float32)
        return

    def _get_obs(self) -> torch.Tensor:
        root_x = self.root_pos[:, FWD_IDX].unsqueeze(1)
        # Positive means hurdle is ahead
        hurdle_deltas = self.hurdle_positions.unsqueeze(0) - root_x
        # Ignore passed hurdles
        hurdle_deltas = torch.where(hurdle_deltas >= 0.0, hurdle_deltas, torch.full_like(hurdle_deltas, float("inf")))
        # Get nearest hurdle distance + index
        nearest_hurdle_dist, nearest_hurdle_idx = torch.min(hurdle_deltas, dim=1, keepdim=True)
        # Gather nearest hurdle height
        nearest_hurdle_height = self.hurdle_heights[nearest_hurdle_idx.squeeze(1)].unsqueeze(1)
        # Handle case where all hurdles are passed
        passed_all = torch.isinf(nearest_hurdle_dist)
        nearest_hurdle_dist = torch.where(passed_all, torch.zeros_like(nearest_hurdle_dist), nearest_hurdle_dist)
        nearest_hurdle_height = torch.where(passed_all, torch.zeros_like(nearest_hurdle_height), nearest_hurdle_height)

        root_x_qpos_id = self.qpos_id_lookup["pelvis_tx"]
        joint_positions_without_x = torch.cat((
            self.joint_positions[:, :root_x_qpos_id],
            self.joint_positions[:, root_x_qpos_id + 1:]),
            dim=1)

        obs = torch.cat([
            self.muscle_activations,
            self.muscle_norm_fiber_lengths,
            self.actuator_activations,
            joint_positions_without_x,
            self.joint_velocities,
            nearest_hurdle_dist,
            nearest_hurdle_height
        ], dim=1)
        return obs.detach().clone()
