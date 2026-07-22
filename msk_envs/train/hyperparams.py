from dataclasses import dataclass, asdict, fields, field
from datetime import datetime
from typing import Union
import sys
import tyro
from typing_extensions import Annotated

from msk_envs.envs.env_config import EnvConfig, EnvConfigUnion
from msk_envs.envs.env_variants import DerivedEnv
from msk_envs.train.dep.dep_config import DEPConfig
from msk_envs.train.sac.sac_config import SACConfig
from msk_envs.train.td3.td3_config import TD3Config
from msk_envs.train.qflex.qflex_config import QFlexConfig
from msk_envs.train.qflex_ref.qflex_ref_config import QFlexRefConfig
from msk_envs.train.ppo.ppo_config import PPOConfig
from msk_envs.utils.train_utils import find_latest_checkpoint


@dataclass
class BaseArgs:
    algo: str = "td3"
    project: str = "msk_sprinter"
    exp_prefix: str = ""
    exp_name: str = ""
    disable_wandb: bool = False
    resume: bool = False
    override_wandb_config: bool = False
    use_dep: bool = False
    seed: int = 1
    gpu_id: int = 0

    env_config: EnvConfigUnion = field(default_factory=EnvConfig)
    td3_config: TD3Config = field(default_factory=TD3Config)
    sac_config: SACConfig = field(default_factory=SACConfig)
    qflex_config: QFlexConfig = field(default_factory=QFlexConfig)
    qflex_ref_config: QFlexRefConfig = field(default_factory=QFlexRefConfig)
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    dep_config: DEPConfig = field(default_factory=DEPConfig)

    def _apply_env_overrides(self, pose_name: str = None, **overrides):
        """
        Applies task defaults first, and THEN overlays the command line changes
        specified by the user.
        """
        # Identify the model folder name from the chosen class name
        cls_name = self.env_config.__class__.__name__
        model_folder = cls_name.replace("EnvConfig", "").lower() or "base"
        # Find starting pose
        if pose_name and hasattr(self.env_config, "starting_pose_path"):
            overrides["starting_pose_path"] = f"../msk_models/{model_folder}/poses/{pose_name}"

        # CLI-set fields
        valid_fields = {f.name for f in fields(self.env_config.__class__)}
        user_specified_fields = set()
        for arg in sys.argv:
            if arg.startswith("--env-config."):
                clean_arg = arg.split("=")[0].replace("--env-config.", "").replace("-", "_")
                if clean_arg in valid_fields:
                    user_specified_fields.add(clean_arg)
                # Handle tyro's boolean flag inversions (e.g., --env-config.no-noise-start)
                elif clean_arg.startswith("no_") and clean_arg[3:] in valid_fields:
                    user_specified_fields.add(clean_arg[3:])

        saved_cli_changes = {
            field_name: getattr(self.env_config, field_name)
            for field_name in user_specified_fields
            if hasattr(self.env_config, field_name)
        }

        # Apply all environment task overrides
        for key, task_default_value in overrides.items():
            if hasattr(self.env_config, key):
                setattr(self.env_config, key, task_default_value)

        # Re-apply the command line changes on top of the overrides
        for key, user_cli_value in saved_cli_changes.items():
            setattr(self.env_config, key, user_cli_value)
        return

    def __post_init__(self):
        if self.resume and self.exp_prefix:
            checkpoint_path, found_exp_name, global_step = find_latest_checkpoint(self.exp_prefix)
            if checkpoint_path:
                self.exp_name = found_exp_name
                if self.algo.lower() == "sac":
                    self.sac_config.checkpoint_path = checkpoint_path
                elif self.algo.lower() == "td3":
                    self.td3_config.checkpoint_path = checkpoint_path
                print(f"Resuming training from checkpoint: {checkpoint_path} at global_step={global_step}")
            else:
                print(f"No checkpoint found for exp_prefix '{self.exp_prefix}'. Starting new training.")

        # Control whether optimizer/scheduler state is included in checkpoints
        self.td3_config.save_optimizer_state = self.resume

        # Generate exp_name if not set by resume
        if not self.exp_name:
            date_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M")
            self.exp_name = f"{self.exp_prefix}_{date_name}" if self.exp_prefix else date_name

        self.traj_out_folder = f"dashboard/trajectories/{self.exp_name}"
        self.analytics_out_folder = f"models/frame_data/{self.exp_name}"

        reward_lambdas = {k: v for k, v in self.__dict__.items() if k.startswith("lambda_")}
        imitation_weights = {k: v for k, v in self.__dict__.items() if k.startswith("imitation_weight_")}

        self.env_config.reward_lambdas = reward_lambdas
        self.env_config.imitation_weights = imitation_weights


def pretty_print_base_args(args: BaseArgs):
    args_dict = asdict(args)
    base_field_names = [f.name for f in fields(BaseArgs)]
    base_items = {k: args_dict.get(k) for k in base_field_names}
    extra_items = {k: v for k, v in args_dict.items() if k not in base_items}

    line = "=" * 80
    print(line)
    print(f"Experiment config: {args_dict.get('exp_name', '')}  "
          f"(env_variant={args_dict.get('env_variant')})")
    print(line)

    EnvConfig.pretty_print(args.env_config)
    if args.algo.lower() == "sac":
        SACConfig.pretty_print(args.sac_config)
    elif args.algo.lower() == "td3":
        TD3Config.pretty_print(args.td3_config)
    elif args.algo.lower() == "qflex":
        QFlexConfig.pretty_print(args.qflex_config)
    elif args.algo.lower() in ("qflex_jax", "qflex_ref"):
        QFlexRefConfig.pretty_print(args.qflex_ref_config)
    elif args.algo.lower() == "ppo":
        PPOConfig.pretty_print(args.ppo_config)

    print("BaseArgs:")
    for k in base_field_names:
        if k in ["env_config", "td3_config"]:
            continue
        print(f"  {k:24} = {base_items[k]}")

    if extra_items:
        print("-" * 80)
        print(f"{type(args).__name__} extras:")
        for k in sorted(extra_items.keys()):
            print(f"  {k:24} = {extra_items[k]}")

    print(line)
    return


@dataclass
class LaneConfig(BaseArgs):
    """ Reusable hyperparams for lane environments """
    lambda_vel: float = 1e-2
    lambda_mid_lane: float = 1e-2
    lambda_spring: float = 0.0
    lambda_damper: float = 0.0
    lambda_limit: float = -3e-4
    lambda_muscle_passive: float = 0.0


@dataclass
class SprintConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_run.yaml",
            env_variant=DerivedEnv.SPRINT,
            delta_t=1.0 / 30.0,
            max_episode_duration=10.0,
        )


@dataclass
class SprintBlockStartConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_blockstart.yaml",
            env_variant=DerivedEnv.SPRINT_BLOCK_START,
            delta_t=1.0 / 30.0,
            max_episode_duration=10.0,
            apply_start_noise=False,
            swap_lr=False,
            default_activation=0.01,
        )


@dataclass
class BackpedalConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_backpedal.yaml",
            env_variant=DerivedEnv.BACKPEDAL,
            delta_t=1.0 / 30.0,
            max_episode_duration=10.0,
        )


@dataclass
class SideShuffleConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_side.yaml",
            env_variant=DerivedEnv.SIDE_SHUFFLE,
            delta_t=1.0 / 30.0,
            max_episode_duration=10.0,
        )


@dataclass
class HurdlesConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_run.yaml",
            env_variant=DerivedEnv.HURDLES,
            delta_t=1.0 / 30.0,
            max_episode_duration=12.0,
        )


@dataclass
class UphillSprintConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_run.yaml",
            env_variant=DerivedEnv.SPRINT,
            delta_t=1.0 / 30.0,
            max_episode_duration=10.0,
            ground_rotation=(0.0, 0.0, 0.2, 0.8),
        )


@dataclass
class RunTheBendConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_run.yaml",
            env_variant=DerivedEnv.RUN_THE_BEND,
            delta_t=1.0 / 30.0,
            max_episode_duration=12.0,
        )


@dataclass
class HopConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_hop.yaml",
            env_variant=DerivedEnv.HOP,
            delta_t=1.0 / 30.0,
            max_episode_duration=10.0,
            swap_lr=False,
        )


@dataclass
class CariocaConfig(LaneConfig):
    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_carioca.yaml",
            env_variant=DerivedEnv.CARIOCA,
            delta_t=1.0 / 30.0,
            max_episode_duration=10.0,
            apply_start_noise=False,
            swap_lr=False,
        )


@dataclass
class LocomotionConfig(BaseArgs):
    """ General locomotion: track a commanded horizontal velocity. """
    lambda_vel_track: float = 1e-1
    lambda_alive: float = 1e-2
    lambda_limit: float = -3e-4
    lambda_muscle_passive: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_run.yaml",
            env_variant=DerivedEnv.LOCOMOTION,
            delta_t=1.0 / 30.0,
            max_episode_duration=10.0,
        )


@dataclass
class VerticalConfig(BaseArgs):
    lambda_jump: float = 1e-1
    lambda_limit: float = -3e-4
    lambda_alive: float = 1e-2

    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            pose_name="starting_pose_vertical.yaml",
            env_variant=DerivedEnv.VERTICAL,
            delta_t=1.0 / 30.0,
            default_activation=0.01,
            apply_start_noise=False,
        )


@dataclass
class ImitateConfig(BaseArgs):
    lambda_track_q: float = 0.0
    lambda_track_u: float = 0.0
    lambda_track_bp: float = 1e-1
    lambda_track_mp: float = 1e-1

    def __post_init__(self):
        super().__post_init__()
        self._apply_env_overrides(
            env_variant=DerivedEnv.IMITATE,
            delta_t=1.0 / 60.0,
            apply_start_noise=False,
            enforce_ground_contact=False,
            default_activation=0.1,
        )


Config = Union[
    Annotated[SprintConfig, tyro.conf.subcommand(name="sprint")],
    Annotated[SprintBlockStartConfig, tyro.conf.subcommand(name="blockstart")],
    Annotated[BackpedalConfig, tyro.conf.subcommand(name="backpedal")],
    Annotated[SideShuffleConfig, tyro.conf.subcommand(name="sideshuffle")],
    Annotated[HurdlesConfig, tyro.conf.subcommand(name="hurdles")],
    Annotated[UphillSprintConfig, tyro.conf.subcommand(name="uphillsprint")],
    Annotated[RunTheBendConfig, tyro.conf.subcommand(name="sprintcurve")],
    Annotated[HopConfig, tyro.conf.subcommand(name="hop")],
    Annotated[CariocaConfig, tyro.conf.subcommand(name="carioca")],
    Annotated[VerticalConfig, tyro.conf.subcommand(name="vertical")],
    Annotated[LocomotionConfig, tyro.conf.subcommand(name="locomotion")],
    Annotated[ImitateConfig, tyro.conf.subcommand(name="imitate")],
]


def get_args():
    args = tyro.cli(Config)
    args.use_wandb = not args.disable_wandb
    return args
