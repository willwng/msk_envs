import json
import msk_warp
from dataclasses import dataclass

from .env_variants import DerivedEnv


@dataclass
class EnvConfig:
    env_variant: DerivedEnv = DerivedEnv.SPRINT
    """ Environment type """

    delta_t: float = 1.0 / 500.0
    """ Control/policy step size """
    delta_t_sim: float = 1.0 / 10000.0
    """ Simulator/physics step size """

    max_episode_duration: float = 12.0
    """ Max episode duration in seconds """

    # Model properties
    model_path: str = "../msk_models/model_motor_arms_foot_contact.osim"
    """ OpenSim model file path """
    joint_damping: float = 0.1
    """ Joint damping applied to all joints """
    joint_armature: float = 0.001
    """ Armature added to all joints (improves stability) """
    torso_damping: float = 1.0
    """ Damping specifically for torso joint """
    toes_stiffness: float = 65.0
    """ Toes joint stiffness """
    toes_damping: float = 0.1
    """ Toes joint damping """
    use_default_joint_limits: bool = False
    """ Whether to use joint limits defined in the model file """
    joint_limits_path: str = "../msk_models/joint_limits_sprinting.yaml"
    """ Joint limits file path (YAML). NOTE: this overrides limits defined in the model file """
    enable_drag: bool = True
    """ Whether to enable drag forces """

    # Constraint properties
    contact_type: msk_warp.ContactType = msk_warp.ContactType.HUNT_CROSSLEY
    """ Contact model type (HUNT_CROSSLEY, HUNT_CROSSLEY_SMOOTH, MUJOCO) """
    limit_type: msk_warp.LimitType = msk_warp.LimitType.EXPONENTIAL
    """ Joint limit model type (MUJOCO, EXPONENTIAL) """
    limit_force_curves_path: str = "../msk_models/limit_force_curves.yaml"
    """ Exponential limit force curves file path (if using Exponential limits) """
    solref: tuple[float, float] = (0.02, 1.0)
    """ MuJoCo limit/contact parameters (if using MuJoCo limits/contacts) """

    # Integrator properties
    integrator: msk_warp.IntegratorType = msk_warp.IntegratorType.RK4_FIXED
    """ Integrator type (EULER_FIXED, RK4_FIXED) """

    # Muscle properties
    muscle_multiplier: float = 2.0
    """ Multiplier to max isometric force """
    muscle_fiber_damping: float = 0.01
    """ Fiber damping (0.0 = undamped) """
    muscle_min_activation: float = 0.0
    """ Minimum muscle activation. Use non-zero for undamped muscle """
    muscle_max_activation: float = 1.0
    """ Maximum muscle activation """
    muscle_v_max: float = 12.0
    """ Maximum contraction velocity (in optimal fiber lengths per second) """
    muscle_dynamics_substeps: int = 0
    """ Number of substeps for muscle dynamics integration (can improve stability) """
    use_function_based_path: bool = True
    """ Whether to use function-based path (or geometry path)"""
    muscle_function_path: str = "../msk_models/muscle_fn_path_info.json"

    # Starting pose (starting_pose and noise is ignored for IMITATE variant)
    starting_pose: str = "../msk_models/starting_pose_run.yaml"
    """ Starting pose file path (YAML) """
    noise_start: bool = True
    """ Whether to add noise to starting state """
    q_noise: float = 0.05
    """ std of starting joint position noise"""
    qv_noise: float = 0.1
    """ std of starting joint velocity noise"""
    swap_lr: bool = True
    """ Whether to swap left/right sides when adding noise to starting state """
    motion_name: str = "../motions/pred_sprint_two_step"
    """ motion file name (without .mot extension) for IMITATE variant """
    use_prescribed_starting_activations: bool = True
    """ Whether to use prescribed starting activations from file """
    starting_activations: str = "../msk_models/starting_activations.yaml"
    """ Starting activations file path (YAML) """
    default_activation: float = 0.05
    """ Default activation value when prescribed activations are not used """

    reward_lambdas: dict = None
    """ Reward weights """
    imitation_weights: dict = None
    """ Imitation reward weights """

    extra_rewarded_joints: list = None
    """ List of body names to give extra reward for tracking (for debug) """
    lambda_extra_rewarded_joints: float = 0.0
    """ Lambda for extra rewarded joints """
    extra_rewarded_dofs: list = None
    """ List of DOFs to give extra reward for tracking (for debug) """
    lambda_extra_rewarded_dofs: float = 0.0
    """ Lambda for extra rewarded DOFs """

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
