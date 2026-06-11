import json
from dataclasses import dataclass, field
from typing import Union

import bolt
import tyro
from typing_extensions import Annotated

from .env_variants import DerivedEnv


@dataclass
class EnvConfig:
    env_variant: DerivedEnv = DerivedEnv.SPRINT
    """ Environment type """

    # --- Control and Simulation Frequency ---
    delta_t: float = 1.0 / 30.0
    """ Control/policy step size """
    delta_t_sim: float = 1.0 / 10000.0
    """ Simulator/physics step size. Ignored if using adaptive integrator """
    max_episode_duration: float = 10.0
    """ Max episode duration in seconds """
    integrator: bolt.IntegratorType = bolt.IntegratorType.EULER_ADAPTIVE
    """ Integrator type (EULER_FIXED, RK4_FIXED, EULER_ADAPTIVE, RK_MERSON_ADAPTIVE) """
    integrator_use_inf_norm: bool = False
    """ For adaptive integrator, whether to use inf norm or L2-norm error calculation """
    integrator_accuracy: float = 0.1
    """ For adaptive integrator, overall accuracy/tolerance. Lower = more accurate but slower """

    # --- Model articulation properties ---
    model_root_free: bool = True
    """ Whether the model root is free (floating base) """
    model_path: str = ""
    """ OpenSim model file path """
    enable_drag: bool = False
    """ Whether to enable drag forces """
    use_specified_contact_params: bool = True
    """ Whether to use contact parameters defined in contact_params_path """
    contact_params_path: str = ""
    """ Contact parameters file path (YAML). NOTE: this overrides contact parameters defined in the model file """
    armature: float = 0.0
    """ Additional armature to add to joints (decreases realism but increases performance) """
    use_implicit_damping: bool = False
    """ Whether to integrate joint damping implicitly """
    use_linear_stop: bool = False
    """ Whether to use LinearStop force instead of CoordinateLimitForce """

    # --- Model muscle properties ---
    muscle_multiplier: float = 1.0
    """ Multiplier to max isometric force """
    muscle_activation_dynamics: bolt.ActivationType = bolt.ActivationType.MILLARD
    """ Muscle activation dynamics type (DGF, MILLARD) """
    muscle_activation_time_const: float = 0.010
    """ Muscle activation time constant. Default 15ms for DGF, 10ms for Millard """
    muscle_deactivation_time_const: float = 0.040
    """ Muscle deactivation time constant. Default 60ms for DGF, 40ms for Millard """
    muscle_activation_dynamics_smoothing: float = 10.0
    """ Muscle activation dynamics smoothing factor for DGF activation dynamics """
    muscle_min_activation: float = 0.01
    """ Minimum muscle activation. Use non-zero for undamped muscle """
    muscle_max_activation: float = 1.0
    """ Maximum muscle activation """
    muscle_contraction_dynamics: bolt.ContractionType = bolt.ContractionType.DGF
    """ Muscle contraction dynamics: which force curves to use (DGF, MILLARD) """
    muscle_active_force_width_scale: float = 1.0
    """ For DGF active force length curve, the scaling factor of the width """
    muscle_fiber_damping: float = 0.1
    """ Fiber damping (0.0 = undamped) """
    muscle_v_max: float = 10.0
    """ Maximum contraction velocity (in optimal fiber lengths per second) """
    muscle_dynamics_substeps: int = 0
    """ Number of substeps for muscle dynamics integration (can improve stability) """
    use_function_based_path: bool = True
    """ Whether to use function-based path (or geometry path)"""
    muscle_function_path: str = ""
    """ Function-based path data file """
    ignore_short_elastic_tendons: bool = False
    """ Ignore tendon dynamics for muscles where tendon slack length < optimal fiber length"""
    custom_muscle_multipliers: dict[str, float] = field(default_factory=dict)
    """ Custom muscle force multiplier (overrides muscle_multiplier) """

    # Starting pose (starting_pose and noise is ignored for IMITATE variant)
    starting_pose_path: str = ""
    """ Starting pose file path (YAML) """
    target_pose_path: str = ""
    """ Target pose file path for reach pose env (YAML) """
    noise_start: bool = True
    """ Whether to add noise to starting state """
    q_noise: float = 0.03
    """ std of starting joint position noise"""
    qv_noise: float = 0.1
    """ std of starting joint velocity noise"""
    noise_root: bool = True
    """ Whether to add noise to root position/rotation """
    swap_lr: bool = True
    """ Whether to swap left/right sides when adding noise to starting state """
    enforce_ground_contact: bool = True
    """ Whether to enforce contact with ground at start (for free body only) """
    motion_name: str = ""
    """ motion file name for IMITATE environments (or variants) """
    default_activation: float = -1.0
    """ Default activation value when prescribed activations are not used. -1.0 is random. """

    # Perturbations
    apply_perturbations: bool = False
    """ Whether to apply random push forces in the environment """
    perturbation_duration: tuple = (0.05, 0.5)
    """ How long each push lasts """
    perturbation_frequency: tuple = (0.5, 2.0)
    """ How long to wait between pushes """
    force_std: float = 300.0
    """ Standard deviation of perturbation force magnitude to apply """

    # Miscellaneous
    ground_rotation: tuple = (0.0, 0.0, 0.0, 1.0)
    """ Ground rotation quaternion (x,y,z,w) """
    gravity: float = -9.80665
    """ Gravity """

    # Rewards for specific env variants: The following need to be set
    reward_lambdas: dict = field(default_factory=dict)
    """ Reward weights """
    imitation_weights: dict = field(default_factory=dict)
    """ Imitation reward weights """
    extra_rewarded_joints: list = field(default_factory=list)
    """ List of body names to give extra reward for tracking (for debug) """
    lambda_extra_rewarded_joints: float = 0.0
    """ Lambda for extra rewarded joints """
    extra_rewarded_dofs: list = field(default_factory=list)
    """ List of DOFs to give extra reward for tracking (for debug) """
    lambda_extra_rewarded_dofs: float = 0.0
    """ Lambda for extra rewarded DOFs """
    target_position: list = field(default_factory=lambda: [0.0, 1.0, 0.0])
    """ Target position for STATIC env variant """

    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_json_file(cls, file_path):
        with open(file_path, 'r') as f:
            json_str = f.read()
        return cls.from_json(json_str)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def pretty_print(env_config):
        print("Environment Configuration:")
        for key, value in env_config.to_dict().items():
            print(f"  {key}: {value}")


@dataclass
class EnvConfigSprinter(EnvConfig):
    model_path: str = "../msk_models/sprinter/sprinter_model.osim"
    muscle_function_path: str = "../msk_models/sprinter/sprinter_model_fn.xml"
    contact_params_path: str = "../msk_models/sprinter/contact_params/contact_params.yaml"
    starting_pose_path: str = "../msk_models/sprinter/poses/starting_pose_stand.yaml"
    ignore_short_elastic_tendons: bool = True

    muscle_multiplier: float = 2.0
    muscle_v_max: float = 12.0
    armature: float = 1e-3
    integrator_accuracy: float = 1.0


@dataclass
class EnvConfigAthlete(EnvConfig):
    model_path: str = "../msk_models/athlete/athlete.osim"
    muscle_function_path: str = "../msk_models/athlete/athlete_fn.xml"
    contact_params_path: str = "../msk_models/athlete/contact_params/contact_params.yaml"
    starting_pose_path: str = "../msk_models/athlete/poses/starting_pose_run.yaml"
    ignore_short_elastic_tendons: bool = True

    # Tuned passive forces using Millard curves
    muscle_contraction_dynamics: bolt.ContractionType = bolt.ContractionType.DGF_MILLARD_PASSIVE

    muscle_multiplier: float = 2.0
    muscle_v_max: float = 12.0
    armature: float = 1e-3
    integrator_accuracy: float = 1.0


EnvConfigUnion = Union[
    Annotated[EnvConfigSprinter, tyro.conf.subcommand(name="sprinter")],
    Annotated[EnvConfigAthlete, tyro.conf.subcommand(name="athlete")],
]
