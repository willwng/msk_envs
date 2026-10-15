"""
Read one world's simulator state into a FrameData. Each tensor is copied to Python once per frame (instead of one
.item() per value), since on the GPU every copy is a device sync.
"""
import bolt
import torch

from msk_envs.envs.utils.scene_settings import SceneSettings
from msk_envs.geometry import get_position_from_transform, get_rotation_from_transform
from .frame_data import (ActuatorData, Arrow, BodyForces, ColliderData, FrameData, JointMoment, JointMuscleBreakdown,
                         KineticData, MuscleData, NamedValue, PointData, VisualData)


def _to_list(values) -> list:
    return values.tolist() if hasattr(values, "tolist") else list(values)


def parse_visual_data(
        m: bolt.Model,
        d: bolt.Data,
        visual_load_results: list[bolt.MeshLoadResult],
        world_id: int
) -> list[VisualData]:
    visual_transforms = bolt.get_visual_transforms(d)
    positions = get_position_from_transform(visual_transforms)[world_id].tolist()
    rotations = get_rotation_from_transform(visual_transforms)[world_id].tolist()

    return [
        VisualData(
            mesh_file=visual_load_results[i].file,
            pos=positions[i],
            rot=rotations[i],
            scale=visual_load_results[i].scale,
        )
        for i in range(bolt.get_num_visuals(m))
    ]


def parse_collider_data(
        m: bolt.Model,
        d: bolt.Data,
        collider_idx_to_name: dict[int, str],
        world_id: int
) -> list[ColliderData]:
    collider_types = _to_list(bolt.get_collider_types(m))
    collider_scales = _to_list(bolt.get_collider_sizes(m))
    collider_transforms = bolt.get_collider_transforms(d)
    positions = get_position_from_transform(collider_transforms)[world_id].tolist()
    rotations = get_rotation_from_transform(collider_transforms)[world_id].tolist()
    contact_forces = torch.stack([torch.norm(force, dim=0) for force in bolt.collider_forces(d)[world_id]]).tolist()

    return [
        ColliderData(
            name=collider_idx_to_name[i],
            geom_type=int(collider_types[i]),
            pos=positions[i],
            rot=rotations[i],
            scale=collider_scales[i],
            contact_force=contact_forces[i],
        )
        for i in range(bolt.get_num_colliders(m))
    ]


def parse_muscle_data(
        m: bolt.Model,
        d: bolt.Data,
        muscle_idx_to_name: dict[int, str],
        world_id: int
) -> list[MuscleData]:
    activations = bolt.muscle_activations(d)[world_id].tolist()
    excitations = bolt.muscle_excitations(d)[world_id].tolist()
    actuations = bolt.muscle_actuations(d)[world_id].tolist()
    path_lengths = bolt.muscle_path_lengths(d)[world_id].tolist()
    path_velocities = bolt.muscle_path_velocities(d)[world_id].tolist()
    moment_arms = bolt.muscle_moment_arms(d)[world_id].tolist()

    metadata = bolt.muscle_metadata_np(m)
    length_info_np = bolt.muscle_length_info_np(d)
    length_info = {
        k: length_info_np[k][world_id].tolist()
        for k in ["fiber_active_force_length_multiplier", "fiber_passive_force_length_multiplier", "fiber_length",
                  "tendon_length", "pennation_angle"]
    }
    velocity_info_np = bolt.muscle_velocity_info_np(d)
    velocity_info = {
        k: velocity_info_np[k][world_id].tolist()
        for k in ["fiber_force_velocity_multiplier", "fiber_damping_force_multiplier", "fiber_velocity"]
    }

    site_positions = bolt.site_positions(d)[world_id].tolist()
    muscle_site_adr = bolt.muscle_site_adr(m)
    muscle_site_num = bolt.muscle_site_num(m)

    muscles = []
    for i in range(bolt.get_num_muscles(m)):
        pt_adr = muscle_site_adr[i]
        n_pts = muscle_site_num[i]
        muscles.append(MuscleData(
            name=muscle_idx_to_name[i],
            points=site_positions[pt_adr:pt_adr + n_pts],
            max_isometric_force=float(metadata["max_isometric_force"][i]),
            optimal_fiber_length=float(metadata["optimal_fiber_length"][i]),
            tendon_slack_length=float(metadata["tendon_slack_length"][i]),
            activation=activations[i],
            excitation=excitations[i],
            actuation=actuations[i],
            length_multiplier=length_info["fiber_active_force_length_multiplier"][i],
            velocity_multiplier=velocity_info["fiber_force_velocity_multiplier"][i],
            passive_multiplier=length_info["fiber_passive_force_length_multiplier"][i],
            damping_multiplier=velocity_info["fiber_damping_force_multiplier"][i],
            path_length=path_lengths[i],
            path_velocity=path_velocities[i],
            fiber_length=length_info["fiber_length"][i],
            fiber_velocity=velocity_info["fiber_velocity"][i],
            tendon_length=length_info["tendon_length"][i],
            pennation_angle=length_info["pennation_angle"][i],
            moment_arm=moment_arms[i],
        ))
    return muscles


def parse_actuator_data(
        m: bolt.Model,
        d: bolt.Data,
        actuation_idx_to_name: dict[int, str],
        world_id: int
) -> list[ActuatorData]:
    activations = bolt.actuator_activations(d)[world_id].tolist()
    excitations = bolt.actuator_excitations(d)[world_id].tolist()
    metadata = bolt.actuator_metadata_np(m)

    return [
        ActuatorData(
            name=actuation_idx_to_name[i],
            optimal_force=float(metadata["optimal_force"][i]),
            activation=activations[i],
            excitation=excitations[i],
        )
        for i in range(bolt.get_num_actuators(m))
    ]


def parse_kinetic_data(
        m: bolt.Model,
        d: bolt.Data,
        body_name_to_idx: dict[str, int],
        world_id: int
) -> KineticData:
    body_com_positions = bolt.body_com_positions(d)[world_id]
    if "talus_l" in body_name_to_idx and "talus_r" in body_name_to_idx:
        foot_pos_l = tuple(body_com_positions[body_name_to_idx["talus_l"]].tolist())
        foot_pos_r = tuple(body_com_positions[body_name_to_idx["talus_r"]].tolist())
    else:
        foot_pos_l = (0, 0, 0)
        foot_pos_r = (0, 0, 0)

    return KineticData(
        com=tuple(bolt.body_subtree_com_positions(d)[world_id, 0].tolist()),
        foot_pos_l=foot_pos_l,
        foot_pos_r=foot_pos_r,
        grf=tuple(bolt.grf(d)[world_id].tolist()),
        total_mass=float(bolt.body_mass(m).sum()),
        gravity=bolt.gravity(m).y,
    )


def parse_joint_angles(
        m: bolt.Model,
        d: bolt.Data,
        qpos_idx_to_name: dict[int, str],
        limit_id_lookup: dict[str, tuple[float, float]],
        world_id: int,
) -> list[NamedValue]:
    joint_angles = bolt.joint_positions(d)[world_id].tolist()
    return [
        NamedValue(
            name=qpos_idx_to_name[i],
            value=joint_angles[i],
            limits=limit_id_lookup[qpos_idx_to_name[i]],
        )
        for i in range(bolt.get_num_qpos(m))
    ]


def parse_joint_moments(
        m: bolt.Model,
        d: bolt.Data,
        dof_idx_to_name: dict[int, str],
        world_id: int
) -> list[JointMoment]:
    net_moment = bolt.joint_moments(d)[world_id].tolist()
    spring = bolt.ufrc_spring(d)[world_id].tolist()
    damper = bolt.ufrc_damper(d)[world_id].tolist()
    muscle = bolt.ufrc_muscle(d)[world_id].tolist()
    muscle_passive = bolt.ufrc_muscle_passive(d)[world_id].tolist()
    actuator = bolt.ufrc_actuator(d)[world_id].tolist()
    limit = bolt.ufrc_limit(d)[world_id].tolist()

    return [
        JointMoment(
            name=dof_idx_to_name[i],
            net_moment=net_moment[i],
            spring=spring[i],
            damping=damper[i],
            muscle=muscle[i],
            muscle_passive=muscle_passive[i],
            actuator=actuator[i],
            limit=limit[i],
        )
        for i in range(bolt.get_num_dofs(m))
    ]


def parse_joint_muscle_breakdown(
        m: bolt.Model,
        d: bolt.Data,
        qpos_idx_to_name: dict[int, str],
        world_id: int
) -> list[JointMuscleBreakdown]:
    """ Per qpos, the passive and active generalized force of each muscle """
    num_muscles = bolt.get_num_muscles(m)
    passive = bolt.qfrc_muscle_passive_breakdown(d)[world_id].tolist()
    active = bolt.qfrc_muscle_active_breakdown(d)[world_id].tolist()

    return [
        JointMuscleBreakdown(
            name=qpos_idx_to_name[i],
            passive_breakdown=passive[i][:num_muscles],
            active_breakdown=active[i][:num_muscles],
        )
        for i in range(bolt.get_num_qpos(m))
    ]


def parse_body_forces(
        m: bolt.Model,
        d: bolt.Data,
        body_idx_to_name: dict[int, str],
        world_id: int
) -> list[BodyForces]:
    gravity = bolt.body_force_gravity(d)[world_id].tolist()
    contact = bolt.body_force_contact(d)[world_id].tolist()
    muscle = bolt.body_force_muscle(d)[world_id].tolist()

    return [
        BodyForces(
            name=body_idx_to_name[i],
            gravity=tuple(gravity[i]),
            contact=tuple(contact[i]),
            muscle=tuple(muscle[i]),
        )
        for i in range(bolt.get_num_bodies(m))
    ]


def parse_beam_points(
        m: bolt.Model,
        d: bolt.Data,
        world_id: int
) -> list[PointData]:
    beam_point_pos = bolt.get_beam_visual_positions(d)[world_id].reshape(-1, 3).tolist()
    return [PointData(name=f"beam_point_{i}", pos=pos) for i, pos in enumerate(beam_point_pos)]


def parse_ext_force_arrows(
        m: bolt.Model,
        d: bolt.Data,
        world_id: int
) -> list[Arrow]:
    """ Arrows for the external (user) forces on each body, excluding ground """
    body_positions = get_position_from_transform(bolt.body_transforms(d))[world_id].tolist()
    ext_forces = bolt.body_user_forces(d)[world_id].tolist()
    return [
        Arrow(start=body_positions[i], direction=ext_forces[i][3:6])
        for i in range(1, bolt.get_num_bodies(m))
    ]


def parse_frame(
        m: bolt.Model,
        d: bolt.Data,
        body_name_to_idx: dict[str, int],
        qpos_idx_to_name: dict[int, str],
        dof_idx_to_name: dict[int, str],
        limit_id_lookup: dict[str, tuple[float, float]],
        muscle_idx_to_name: dict[int, str],
        actuation_idx_to_name: dict[int, str],
        collider_idx_to_name: dict[int, str],
        visual_load_results: list[bolt.MeshLoadResult],
        world_id: int,
        frame_time: float,
        scene_settings: SceneSettings,
) -> FrameData:
    body_idx_to_name = {v: k for k, v in body_name_to_idx.items()}
    return FrameData(
        time=frame_time,
        scene_settings=scene_settings,
        visuals=parse_visual_data(m, d, visual_load_results, world_id),
        colliders=parse_collider_data(m, d, collider_idx_to_name, world_id),
        joint_angles=parse_joint_angles(m, d, qpos_idx_to_name, limit_id_lookup, world_id),
        joint_velocities=[],  # not recorded yet
        joint_moments=parse_joint_moments(m, d, dof_idx_to_name, world_id),
        joint_muscle_breakdown=parse_joint_muscle_breakdown(m, d, qpos_idx_to_name, world_id),
        body_forces=parse_body_forces(m, d, body_idx_to_name, world_id),
        muscles=parse_muscle_data(m, d, muscle_idx_to_name, world_id),
        actuators=parse_actuator_data(m, d, actuation_idx_to_name, world_id),
        kinetic_data=parse_kinetic_data(m, d, body_name_to_idx, world_id),
        beam_points=parse_beam_points(m, d, world_id),
        arrows=parse_ext_force_arrows(m, d, world_id),
        targets=[],
    )
