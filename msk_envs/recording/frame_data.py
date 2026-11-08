"""
Per-frame data. These dataclasses define the JSON formats written to disk:
- frame data (FrameData.to_dict): everything below, used for analytics
- animation (see animation.py): the subset needed by the dashboard and Blender, with to_anim_dict()
"""
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional

from msk_envs.envs.utils.scene_settings import SceneSettings
from msk_envs.trajectory_io import MUSCLE_FRAME_FIELDS


def to_json_value(value):
    """ Convert dataclasses (recursively) and sequences to JSON-ready dicts and lists """
    if is_dataclass(value):
        return value.to_dict() if hasattr(value, "to_dict") else {
            f.name: to_json_value(getattr(value, f.name)) for f in fields(value)
        }
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__iter__"):
        return value
    return [to_json_value(v) for v in value]  # lists, tuples, and vectors (e.g. warp vec3)


class Serializable:
    """ Mixin: to_dict() includes every field, in order """

    def to_dict(self) -> dict:
        return {f.name: to_json_value(getattr(self, f.name)) for f in fields(self)}


@dataclass
class PointData(Serializable):
    name: str
    pos: list[float]


@dataclass
class ColliderData(Serializable):
    name: str
    geom_type: int
    """ bolt.GeomType value """
    pos: list[float]
    rot: list[float]
    scale: list[float]
    contact_force: float


@dataclass
class VisualData(Serializable):
    mesh_file: str
    pos: list[float]
    rot: list[float]
    scale: list[float]
    opacity: float = 1.0


@dataclass
class MuscleData(Serializable):
    name: str
    points: list
    """ Path polyline, [n_points][3]: sampled over wrapping surfaces, with inactive conditional points collapsed onto
    the previous point, so every frame has the same number of points (see muscle_paths.py) """

    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float

    activation: float
    excitation: float

    actuation: float
    length_multiplier: float
    velocity_multiplier: float
    passive_multiplier: float
    damping_multiplier: float

    path_length: float
    path_velocity: float
    fiber_length: float
    fiber_velocity: float
    tendon_length: float
    pennation_angle: float

    moment_arm: list[float]
    """ Per qpos """

    ANIM_FIELDS = ("name", "max_isometric_force", "optimal_fiber_length", "tendon_slack_length") + MUSCLE_FRAME_FIELDS
    """ The animation's muscle fields (trajectory_io), besides points """

    def to_anim_dict(self) -> dict:
        # points are already lists of [x, y, z] (frame_parser.parse_muscle_data)
        return {"points": self.points} | {name: to_json_value(getattr(self, name)) for name in self.ANIM_FIELDS}


@dataclass
class ActuatorData(Serializable):
    name: str

    optimal_force: float
    activation: float
    excitation: float


@dataclass
class KineticData(Serializable):
    com: tuple
    foot_pos_l: tuple
    foot_pos_r: tuple
    grf: tuple
    total_mass: float
    gravity: float


@dataclass
class NamedValue(Serializable):
    name: str
    value: float
    limits: Optional[tuple[float, float]] = None

    def to_dict(self) -> dict:
        ret = {"name": self.name, "value": self.value}
        if self.limits is not None:
            ret["limits"] = list(self.limits)
        return ret


@dataclass
class JointMoment(Serializable):
    name: str
    net_moment: float
    spring: float
    damping: float
    muscle: float
    muscle_passive: float
    actuator: float
    limit: float


@dataclass
class JointMuscleBreakdown(Serializable):
    name: str
    passive_breakdown: list[float]
    """ Per muscle """
    active_breakdown: list[float]
    """ Per muscle """


@dataclass
class BodyForces(Serializable):
    name: str
    gravity: tuple
    contact: tuple
    muscle: tuple


@dataclass
class Arrow(Serializable):
    start: list[float]
    direction: list[float]


@dataclass
class WrapSurfaceData(Serializable):
    """ A muscle wrapping surface (see muscle_paths.WrapSurface), in its surface frame """
    name: str
    wrap_type: int
    """ bolt.WrapType value: 0 sphere, 1 cylinder (along z), 2 ellipsoid, 3 torus (around z) """
    size: list[float]
    half_length: float
    """ Drawn half length of cylinders """
    pos: list[float]
    rot: list[float]


@dataclass
class TargetData(Serializable):
    pos: list[float]
    rot: list[float]
    radius: float
    active: bool


@dataclass
class FrameData(Serializable):
    time: float
    visuals: list[VisualData]
    colliders: list[ColliderData]
    joint_angles: list[NamedValue]
    joint_velocities: list[NamedValue]
    joint_moments: list[JointMoment]
    joint_muscle_breakdown: list[JointMuscleBreakdown]
    body_forces: list[BodyForces]
    muscles: list[MuscleData]
    actuators: list[ActuatorData]
    kinetic_data: KineticData
    arrows: list[Arrow]
    targets: list[TargetData]
    beam_points: list[PointData]
    scene_settings: SceneSettings
    wrap_surfaces: list[WrapSurfaceData] = field(default_factory=list)
