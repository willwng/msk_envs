import bolt
import warp as wp

from msk_envs.geometry import FWD_IDX, build_axis
from .config import EnvConfig
from .lanes import LanesEnv


class BlockStartSprintingEnv(LanesEnv):
    """ Sprint from starting blocks """
    target_dir = build_axis(FWD_IDX, 1.0)

    def _add_colliders(self, env_config: EnvConfig) -> None:
        colliders = self.load_result.colliders
        starting_block1 = bolt.UserGeomData(
            name="starting_block1",
            body_name=bolt.GROUND,
            geom_type=bolt.GeomType.CAPSULE,
            transform=wp.transform(wp.vec3(-0.3, 0.0, 0.1), wp.quat_identity(dtype=float)),
            size=wp.vec3(0.1, 0.05, 0.1),
            priority=9,
        )
        starting_block2 = bolt.UserGeomData(
            name="starting_block2",
            body_name=bolt.GROUND,
            geom_type=bolt.GeomType.CAPSULE,
            transform=wp.transform(wp.vec3(-0.73, 0.0, -0.1), wp.quat_identity(dtype=float)),
            size=wp.vec3(0.1, 0.05, 0.1),
            priority=9,
        )
        colliders.append(bolt.convert_user_collider(starting_block1))
        colliders.append(bolt.convert_user_collider(starting_block2))
