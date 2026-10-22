import os
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Union

import bolt
import tyro
from typing_extensions import Annotated

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "msk_models")
""" Model file paths in EnvConfig are relative to this directory (absolute paths are used as-is) """


def resolve_model_path(path: str) -> str:
    return os.path.join(MODELS_DIR, path)


class DerivedEnv(Enum):
    """ Task variants; see registry.ENV_CLASSES for the env class of each """
    SPRINT = 0
    SPRINT_BLOCK_START = 1
    BACKPEDAL = 2
    SIDE_SHUFFLE = 3
    HURDLES = 4
    RUN_THE_BEND = 5
    HOP = 6
    CARIOCA = 7
    VERTICAL = 8
    LOCOMOTION = 9


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
    integrator_min_step_size: float = 1e-5
    """ Minimum step size for adaptive integrator """
    integrator_step_estimator: bolt.StepEstimator = bolt.StepEstimator.PI_CONTROL
    """ Step estimator (INTEGRAL_CONTROL, PI_CONTROL) """

    # --- Model articulation properties ---
    model_path: str = ""
    """ OpenSim model file path (relative to msk_models/) """
    use_specified_contact_params: bool = True
    """ Whether to use contact parameters defined in contact_params_path """
    contact_params_path: str = ""
    """ Contact parameters file path (YAML). NOTE: this overrides contact parameters defined in the model file """
    armature: float = 0.0
    """ Additional armature to add to joints (decreases realism but increases performance) """
    use_implicit_damping: bool = False
    """ Whether to integrate joint damping implicitly """
    use_implicit_friction_damping: bool = False
    """ Whether to integrate friction damping implicitly (can speed up Hunt-Crossley contacts) """
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
    use_function_based_path: bool = True
    """ Whether to use function-based path (or geometry path)"""
    muscle_function_path: str = ""
    """ Function-based path data file """
    ignore_short_elastic_tendons: bool = False
    """ Ignore tendon dynamics for muscles where tendon slack length < optimal fiber length"""
    force_rigid_tendons: bool = False
    """ Force all tendons to be rigid"""

    # Starting pose
    starting_pose_path: str = ""
    """ Starting pose file path (YAML) """
    apply_start_noise: bool = True
    """ Whether to add noise to starting state """
    q_noise: float = 0.03
    """ std of starting joint position noise"""
    qv_noise: float = 0.1
    """ std of starting joint velocity noise"""
    apply_swap_lr: bool = True
    """ Whether to swap left/right sides when adding noise to starting state """
    enforce_ground_contact: bool = True
    """ Whether to enforce contact with ground at start (for free body only) """
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

    # Set by the training task config
    reward_lambdas: dict = field(default_factory=dict)
    """ Reward weights, keyed lambda_<name> for each reward term rew_<name> """

    def pretty_print(self):
        print("Environment Configuration:")
        for f in fields(self):
            print(f"  {f.name}: {getattr(self, f.name)}")


@dataclass
class EnvConfigSprinter(EnvConfig):  # Setup used for publication
    model_path: str = "sprinter/sprinter_model.osim"
    muscle_function_path: str = "sprinter/sprinter_model_fn.xml"
    contact_params_path: str = "sprinter/contact_params/contact_params.yaml"
    starting_pose_path: str = "sprinter/poses/starting_pose_stand.yaml"
    ignore_short_elastic_tendons: bool = True

    use_implicit_damping: bool = False
    use_implicit_friction_damping: bool = True

    muscle_multiplier: float = 2.0
    muscle_v_max: float = 12.0
    armature: float = 1e-3
    integrator_accuracy: float = 1.0


@dataclass
class EnvConfigSprinterExp(EnvConfig):
    model_path: str = "sprinterexp/sprinter_model_exp.osim"
    muscle_function_path: str = "sprinterexp/sprinter_model_fn.xml"
    contact_params_path: str = "sprinterexp/contact_params/contact_params.yaml"
    starting_pose_path: str = "sprinterexp/poses/starting_pose_stand.yaml"
    ignore_short_elastic_tendons: bool = False
    enforce_ground_contact: bool = False

    use_implicit_damping: bool = False
    use_implicit_friction_damping: bool = False

    muscle_multiplier: float = 2.0
    muscle_v_max: float = 12.0
    armature: float = 1e-3
    integrator_accuracy: float = 0.01


EnvConfigUnion = Union[
    Annotated[EnvConfigSprinter, tyro.conf.subcommand(name="sprinter")],
    Annotated[EnvConfigSprinterExp, tyro.conf.subcommand(name="sprinterexp")],
]
