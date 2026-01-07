import msk_warp
import os
import torch

from dataclasses import dataclass
from typing import Optional


@dataclass
class ColliderData:
    geom_type: int
    pos: list[float]
    rot: list[float]
    scale: list[float]

    def to_dict(self):
        return {
            "geom_type": int(self.geom_type),
            "pos": list(self.pos),
            "rot": list(self.rot),
            "scale": list(self.scale),
        }

    @staticmethod
    def from_dict(data: dict) -> 'ColliderData':
        return ColliderData(
            geom_type=data["geom_type"],
            pos=data["pos"],
            rot=data["rot"],
            scale=data["scale"],
        )


@dataclass
class VisualData:
    mesh_file: str
    pos: list[float]
    rot: list[float]
    scale: list[float]
    opacity: float = 1.0

    def to_dict(self):
        # Remove .vtp if it exists and replace with .obj
        mesh_obj_file = self.mesh_file
        if mesh_obj_file.endswith('.vtp'):
            mesh_obj_file = self.mesh_file[:-4] + '.obj'
        mesh_obj_file = os.path.join("assets", "geometry", "obj", mesh_obj_file)

        return {
            "mesh_file": mesh_obj_file,
            "pos": list(self.pos),
            "rot": list(self.rot),
            "scale": list(self.scale),
            "opacity": self.opacity,
        }

    @staticmethod
    def from_dict(data: dict) -> 'VisualData':
        return VisualData(
            mesh_file=data["mesh_file"],
            pos=data["pos"],
            rot=data["rot"],
            scale=data["scale"],
            opacity=data.get("opacity", 1.0),
        )


@dataclass
class MuscleData:
    name: str
    points: list

    max_isometric_force: float
    activation: float
    excitation: float
    actuation: float

    path_length: float
    path_velocity: float
    fiber_length: float
    fiber_velocity: float
    tendon_length: float
    pennation_angle: float

    def to_dict(self):
        return {
            "name": self.name,
            "points": self.points,
            "max_isometric_force": self.max_isometric_force,
            "activation": self.activation,
            "excitation": self.excitation,
            "actuation": self.actuation,
            "path_length": self.path_length,
            "path_velocity": self.path_velocity,
            "fiber_length": self.fiber_length,
            "fiber_velocity": self.fiber_velocity,
            "tendon_length": self.tendon_length,
            "pennation_angle": self.pennation_angle,
        }

    @staticmethod
    def from_dict(data: dict) -> 'MuscleData':
        return MuscleData(
            name=data["name"],
            points=data["points"],
            max_isometric_force=data["max_isometric_force"],
            activation=data["activation"],
            excitation=data["excitation"],
            actuation=data["actuation"],
            path_length=data["path_length"],
            path_velocity=data["path_velocity"],
            fiber_length=data["fiber_length"],
            fiber_velocity=data["fiber_velocity"],
            tendon_length=data["tendon_length"],
            pennation_angle=data["pennation_angle"],
        )


@dataclass
class ActuatorData:
    name: str

    optimal_force: float
    activation: float
    excitation: float

    def to_dict(self):
        return {
            "name": self.name,
            "optimal_force": self.optimal_force,
            "activation": self.activation,
            "excitation": self.excitation,
        }

    @staticmethod
    def from_dict(data: dict) -> 'ActuatorData':
        return ActuatorData(
            name=data["name"],
            optimal_force=data["optimal_force"],
            activation=data["activation"],
            excitation=data["excitation"],
        )


@dataclass
class KineticData:
    com: tuple
    grf: tuple
    total_mass: float
    gravity: float

    def to_dict(self):
        return {
            "com": list(self.com),
            "grf": list(self.grf),
            "total_mass": self.total_mass,
            "gravity": self.gravity,
        }

    @staticmethod
    def from_dict(data: dict) -> 'KineticData':
        return KineticData(
            com=tuple(data["com"]),
            grf=tuple(data["grf"]),
            total_mass=data["total_mass"],
            gravity=data["gravity"],
        )


@dataclass
class NamedValue:
    name: str
    value: float
    reference: Optional[float] = None
    limits: Optional[tuple[float, float]] = None

    def has_reference(self) -> bool:
        return self.reference is not None

    def has_limits(self) -> bool:
        return self.limits is not None

    def to_dict(self):
        ret = {
            "name": self.name,
            "value": self.value,
        }
        if self.reference is not None:
            ret["reference"] = self.reference
        if self.limits is not None:
            ret["limits"] = list(self.limits)
        return ret

    @staticmethod
    def from_dict(data: dict) -> 'NamedValue':
        return NamedValue(
            name=data["name"],
            value=data["value"],
            reference=data.get("reference"),
            limits=tuple(data["limits"]) if "limits" in data else None,
        )


@dataclass
class FrameData:
    time: float
    visuals: list[VisualData]
    colliders: list[ColliderData]
    joint_angles: list[NamedValue]
    joint_velocities: list[NamedValue]
    joint_moments: list[NamedValue]
    muscles: list[MuscleData]
    actuators: list[ActuatorData]
    kinetic_data: KineticData
    reward_data: dict

    def to_dict(self):
        return {
            "time": self.time,
            "visuals": [obj.to_dict() for obj in self.visuals],
            "colliders": [obj.to_dict() for obj in self.colliders],
            "joint_angles": [angle.to_dict() for angle in self.joint_angles],
            "joint_velocities": [vel.to_dict() for vel in self.joint_velocities],
            "joint_moments": [moment.to_dict() for moment in self.joint_moments],
            "muscles": [muscle.to_dict() for muscle in self.muscles],
            "actuators": [actuator.to_dict() for actuator in self.actuators],
            "kinetic_data": self.kinetic_data.to_dict(),
            "reward_data": self.reward_data,
        }

    @staticmethod
    def from_dict(data: dict) -> 'FrameData':
        return FrameData(
            time=data["time"],
            visuals=[VisualData.from_dict(obj) for obj in data["visuals"]],
            colliders=[ColliderData.from_dict(obj) for obj in data["colliders"]],
            joint_angles=[NamedValue.from_dict(angle) for angle in data["joint_angles"]],
            joint_velocities=[NamedValue.from_dict(vel) for vel in data["joint_velocities"]],
            joint_moments=[NamedValue.from_dict(moment) for moment in data["joint_moments"]],
            muscles=[MuscleData.from_dict(muscle) for muscle in data["muscles"]],
            actuators=[ActuatorData.from_dict(actuator) for actuator in data["actuators"]],
            kinetic_data=KineticData.from_dict(data["kinetic_data"]),
            reward_data=data["reward_data"],
        )


def parse_visual_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        visual_load_results: list[msk_warp.types.MeshLoadResult],
        world_id: int
) -> list[VisualData]:
    visual_positions = msk_warp.get_visual_positions(d)
    visual_rotations = msk_warp.get_visual_rotations(d)

    visuals = []
    for i in range(msk_warp.get_num_visuals(m)):
        visual_load = visual_load_results[i]
        visual_data = VisualData(
            mesh_file=visual_load.file,
            pos=visual_positions[world_id][i].tolist(),
            rot=visual_rotations[world_id][i].tolist(),
            scale=visual_load.scale,
        )
        visuals.append(visual_data)
    return visuals


def parse_collider_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        world_id: int
) -> list[ColliderData]:
    collider_types = msk_warp.get_collider_types(m)
    collider_scales = msk_warp.get_collider_sizes(m)
    collider_positions = msk_warp.get_collider_positions(d)
    collider_rotations = msk_warp.get_collider_rotations(d)

    colliders = []
    for i in range(msk_warp.get_num_colliders(m)):
        collider_data = ColliderData(
            geom_type=int(collider_types[i]),
            pos=collider_positions[world_id][i].tolist(),
            rot=collider_rotations[world_id][i].tolist(),
            scale=collider_scales[i].tolist(),
        )
        colliders.append(collider_data)
    return colliders


def parse_muscle_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        muscle_idx_to_name: dict[int, str],
        world_id: int
) -> list[MuscleData]:
    muscle_activations = msk_warp.muscle_activations(d)
    muscle_excitations = msk_warp.muscle_excitations(d)
    muscle_actuations = msk_warp.muscle_actuations(d)
    muscle_path_lengths = msk_warp.muscle_path_lengths(d)
    muscle_path_velocities = msk_warp.muscle_path_velocities(d)
    muscle_fiber_lengths = msk_warp.muscle_fiber_lengths(d)
    muscle_fiber_velocities = msk_warp.muscle_fiber_velocities(d)

    muscle_metadata = msk_warp.muscle_metadata_np(m)
    muscle_length_info = msk_warp.muscle_length_info_np(d)

    site_positions = msk_warp.site_positions(d)
    muscle_site_adr = msk_warp.muscle_site_adr(m)
    muscle_site_num = msk_warp.muscle_site_num(m)

    muscles = []
    for i in range(msk_warp.get_num_muscles(m)):
        pt_adr = muscle_site_adr[i]
        n_pts = muscle_site_num[i]
        muscle_data = MuscleData(
            name=muscle_idx_to_name[i],
            points=site_positions[world_id][pt_adr:pt_adr + n_pts].tolist(),
            max_isometric_force=float(muscle_metadata["max_isometric_force"][i]),
            activation=float(muscle_activations[world_id][i].item()),
            excitation=float(muscle_excitations[world_id][i].item()),
            actuation=float(muscle_actuations[world_id][i].item()),
            path_length=float(muscle_path_lengths[world_id][i].item()),
            path_velocity=float(muscle_path_velocities[world_id][i].item()),
            fiber_length=float(muscle_fiber_lengths[world_id][i].item()),
            fiber_velocity=float(muscle_fiber_velocities[world_id][i].item()),
            tendon_length=float(muscle_length_info["tendon_length"][world_id][i].item()),
            pennation_angle=float(muscle_length_info["pennation_angle"][world_id][i].item()),
        )
        muscles.append(muscle_data)
    return muscles


def parse_actuator_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        actuation_idx_to_name: dict[int, str],
        world_id: int
) -> list[ActuatorData]:
    actuator_activations = msk_warp.actuator_activations(d)
    actuator_excitations = msk_warp.actuator_excitations(d)
    actuator_metadata = msk_warp.actuator_metadata_np(m)

    actuators = []
    for i in range(msk_warp.get_num_actuators(m)):
        actuator_data = ActuatorData(
            name=actuation_idx_to_name[i],
            optimal_force=float(actuator_metadata["optimal_force"][i]),
            activation=float(actuator_activations[world_id][i].item()),
            excitation=float(actuator_excitations[world_id][i].item()),
        )
        actuators.append(actuator_data)
    return actuators


def parse_kinetic_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        world_id: int
) -> KineticData:
    com = msk_warp.subtree_com_positions(d)[world_id][1].tolist()  # why am I hardcoded!
    grf = msk_warp.grf(d)[world_id].tolist()
    mass = msk_warp.subtree_mass(m)[1]
    gravity = msk_warp.gravity(m)
    kinetic_data = KineticData(
        com=tuple(com),
        grf=tuple(grf),
        total_mass=float(mass),
        gravity=gravity,
    )
    return kinetic_data


def find_index_1d(tensor, x):
    idx = torch.where(tensor == x)[0]
    return idx[0].item() if idx.numel() > 0 else None


def parse_joint_angles(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        qpos_idx_to_name: dict[int, str],
        world_id: int,
        ref_joint_angles: torch.Tensor | None
) -> list[NamedValue]:
    joint_angles = msk_warp.joint_positions(d)
    joint_limit_ranges = msk_warp.joint_limit_ranges(m)
    joint_limit_qadr = list(msk_warp.joint_limit_qadr(m))
    angles = []

    for i in range(msk_warp.get_num_qpos(m)):
        reference = None if ref_joint_angles is None else float(ref_joint_angles[world_id][i].item())
        limits = None
        limit_id = find_index_1d(torch.tensor(joint_limit_qadr), i)
        if limit_id is not None:
            limits = (
                float(joint_limit_ranges[limit_id, 0]),
                float(joint_limit_ranges[limit_id, 1]),
            )

        angle = NamedValue(
            name=qpos_idx_to_name[i],
            value=float(joint_angles[world_id][i].item()),
            reference=reference,
            limits=limits,
        )
        angles.append(angle)
    return angles


def parse_joint_velocities(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        world_id: int
) -> list[NamedValue]:
    joint_velocities = msk_warp.joint_velocities(d)
    velocities = []  # todo
    return velocities


def parse_joint_moments(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        dof_idx_to_name: dict[int, str],
        world_id: int
) -> list[NamedValue]:
    joint_moments = msk_warp.joint_moments(d)
    moments = []

    for i in range(msk_warp.get_num_dofs(m)):
        angle = NamedValue(
            name=dof_idx_to_name[i],
            value=float(joint_moments[world_id][i].item()),
        )
        moments.append(angle)
    return moments


def parse_frame(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        qpos_idx_to_name: dict[int, str],
        dof_idx_to_name: dict[int, str],
        muscle_idx_to_name: dict[int, str],
        actuation_idx_to_name: dict[int, str],
        visual_load_results: list[msk_warp.types.MeshLoadResult],
        world_id: int,
        frame_time: float,
        reward_data: dict,
        ref_joint_angles=None,
) -> FrameData:
    visuals = parse_visual_data(m, d, visual_load_results, world_id)
    colliders = parse_collider_data(m, d, world_id)
    muscles = parse_muscle_data(m, d, muscle_idx_to_name, world_id)
    actuators = parse_actuator_data(m, d, actuation_idx_to_name, world_id)
    kinetic_data = parse_kinetic_data(m, d, world_id)
    joint_angles = parse_joint_angles(m, d, qpos_idx_to_name, world_id, ref_joint_angles)
    joint_velocities = parse_joint_velocities(m, d, world_id)
    joint_moments = parse_joint_moments(m, d, dof_idx_to_name, world_id)

    frame_visuals = FrameData(
        time=frame_time,
        visuals=visuals,
        colliders=colliders,
        joint_angles=joint_angles,
        joint_velocities=joint_velocities,
        joint_moments=joint_moments,
        muscles=muscles,
        actuators=actuators,
        kinetic_data=kinetic_data,
        reward_data=reward_data
    )

    return frame_visuals


def add_reference_visuals(
        frame: FrameData,
        ref_visuals_pos: list,
        ref_visuals_rot: list,
):
    for i in range(len(frame.visuals)):
        ref_pos = ref_visuals_pos[i].tolist()
        ref_rot = ref_visuals_rot[i].tolist()
        ref_visual = VisualData(
            mesh_file=frame.visuals[i].mesh_file,
            pos=ref_pos,
            rot=ref_rot,
            scale=frame.visuals[i].scale,
            opacity=0.3
        )
        frame.visuals.append(ref_visual)
