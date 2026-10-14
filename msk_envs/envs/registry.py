import torch

from .base import MSKEnv
from .block_start import BlockStartSprintingEnv
from .carioca import CariocaEnv
from .config import DerivedEnv, EnvConfig
from .curve import CurvedTrackEnv
from .hurdles import HurdlesEnv
from .lanes import BackPedalEnv, HopEnv, SideShuffleEnv, SprintingEnv
from .locomotion import LocomotionEnv
from .vertical import VerticalEnv

ENV_CLASSES: dict[DerivedEnv, type[MSKEnv]] = {
    DerivedEnv.SPRINT: SprintingEnv,
    DerivedEnv.SPRINT_BLOCK_START: BlockStartSprintingEnv,
    DerivedEnv.BACKPEDAL: BackPedalEnv,
    DerivedEnv.SIDE_SHUFFLE: SideShuffleEnv,
    DerivedEnv.HURDLES: HurdlesEnv,
    DerivedEnv.RUN_THE_BEND: CurvedTrackEnv,
    DerivedEnv.HOP: HopEnv,
    DerivedEnv.CARIOCA: CariocaEnv,
    DerivedEnv.VERTICAL: VerticalEnv,
    DerivedEnv.LOCOMOTION: LocomotionEnv,
}


def make_env(
        env_config: EnvConfig,
        num_envs: int,
        device: torch.device,
        requires_visuals: bool,
        cuda_graph: bool,
) -> MSKEnv:
    """ Build the env for env_config.env_variant """
    env_cls = ENV_CLASSES[env_config.env_variant]
    return env_cls(
        num_envs=num_envs,
        env_config=env_config,
        device=device,
        requires_visuals=requires_visuals,
        cuda_graph=cuda_graph,
    )
