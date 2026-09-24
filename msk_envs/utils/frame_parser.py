import bolt
import torch

from msk_envs.utils.frame_data import *
from msk_envs.utils.transforms import get_position_from_transform, get_rotation_from_transform


def parse_visual_data(
        m: bolt.Model,
        d: bolt.Data,
        visual_load_results: list[bolt.MeshLoadResult],
        world_id: int
) -> list[VisualData]:
    visual_transforms = bolt.get_visual_transforms(d)
    visuals_positions = get_position_from_transform(visual_transforms)
    visual_rotations = get_rotation_from_transform(visual_transforms)

    visuals = []
    for i in range(bolt.get_num_visuals(m)):
        visual_load = visual_load_results[i]
        visual_data = VisualData(
            mesh_file=visual_load.file,
            pos=visuals_positions[world_id][i].tolist(),
            rot=visual_rotations[world_id][i].tolist(),
            scale=visual_load.scale,
        )
        visuals.append(visual_data)
    return visuals


def parse_collider_data(
        m: bolt.Model,
        d: bolt.Data,
        collider_idx_to_name: dict[int, str],
        world_id: int
) -> list[ColliderData]:
    collider_types = bolt.get_collider_types(m)
    collider_scales = bolt.get_collider_sizes(m)
    collider_transforms = bolt.get_collider_transforms(d)
    collider_positions = get_position_from_transform(collider_transforms)
    collider_rotations = get_rotation_from_transform(collider_transforms)
    collider_forces = bolt.collider_forces(d)

    colliders = []
    for i in range(bolt.get_num_colliders(m)):
        collider_data = ColliderData(
            name=collider_idx_to_name[i],
            geom_type=int(collider_types[i]),
            pos=collider_positions[world_id][i].tolist(),
            rot=collider_rotations[world_id][i].tolist(),
            scale=collider_scales[i].tolist(),
            contact_force=torch.norm(collider_forces[world_id][i], dim=0).item(),
        )
        colliders.append(collider_data)
    return colliders


def parse_muscle_data(
        m: bolt.Model,
        d: bolt.Data,
        muscle_idx_to_name: dict[int, str],
        world_id: int
) -> list[MuscleData]:
    muscle_activations = bolt.muscle_activations(d)
    muscle_excitations = bolt.muscle_excitations(d)
    muscle_actuations = bolt.muscle_actuations(d)
    muscle_path_lengths = bolt.muscle_path_lengths(d)
    muscle_path_velocities = bolt.muscle_path_velocities(d)
    muscle_moment_arms = bolt.muscle_moment_arms(d)

    muscle_metadata = bolt.muscle_metadata_np(m)
    muscle_length_info = bolt.muscle_length_info_np(d)
    # # sort the muscles based on fiber_passive_force_length_multiplier
    # import numpy as np
    # nm = bolt.get_num_muscles(m)
    # # Muscle names
    # muscle_names = [muscle_idx_to_name[i] for i in range(nm)]
    # # Passive multiplier
    # passive_flm = [float(muscle_length_info["fiber_passive_force_length_multiplier"][world_id][i].item()) for i in
    #                range(nm)]
    # # Muscle path length
    # muscle_lengths = [float(muscle_path_lengths[world_id][i].item()) for i in range(nm)]
    # # "Optimal/Default" length
    # optimal_fl = [float(muscle_metadata["optimal_fiber_length"][i]) for i in range(nm)]
    # tsl = [float(muscle_metadata["tendon_slack_length"][i]) for i in range(nm)]
    # optimal_pennation = [float(muscle_metadata["optimal_pennation_angle"][i]) for i in range(nm)]
    # default_lengths = []
    # for i in range(nm):
    #     default_length = optimal_fl[i] * np.cos(optimal_pennation[i]) + tsl[i]
    #     default_lengths.append(default_length)
    #
    # muscle_names = np.array(muscle_names)
    # passive_flm = np.array(passive_flm)
    # default_lengths = np.array(default_lengths)
    # muscle_lengths = np.array(muscle_lengths)
    # multiplier = muscle_lengths / default_lengths
    #
    # ind_sort = np.argsort(multiplier)[::-1]
    # passive_flm = passive_flm[ind_sort]
    # muscle_names = muscle_names[ind_sort]
    # default_lengths = default_lengths[ind_sort]
    # muscle_lengths = muscle_lengths[ind_sort]
    # multiplier = multiplier[ind_sort]
    #
    # for i in range(len(ind_sort)):
    #     print(f"{muscle_names[i]}. multiplier: {multiplier[i]}")
    #
    # quit()

    muscle_velocity_info = bolt.muscle_velocity_info_np(d)

    site_positions = bolt.site_positions(d)
    muscle_site_adr = bolt.muscle_site_adr(m)
    muscle_site_num = bolt.muscle_site_num(m)

    muscles = []
    for i in range(bolt.get_num_muscles(m)):
        pt_adr = muscle_site_adr[i]
        n_pts = muscle_site_num[i]
        muscle_data = MuscleData(
            name=muscle_idx_to_name[i],
            points=site_positions[world_id][pt_adr:pt_adr + n_pts].tolist(),
            max_isometric_force=float(muscle_metadata["max_isometric_force"][i]),
            optimal_fiber_length=float(muscle_metadata["optimal_fiber_length"][i]),
            tendon_slack_length=float(muscle_metadata["tendon_slack_length"][i]),
            activation=float(muscle_activations[world_id][i].item()),
            excitation=float(muscle_excitations[world_id][i].item()),
            actuation=float(muscle_actuations[world_id][i].item()),
            length_multiplier=float(muscle_length_info["fiber_active_force_length_multiplier"][world_id][i].item()),
            velocity_multiplier=float(muscle_velocity_info["fiber_force_velocity_multiplier"][world_id][i].item()),
            passive_multiplier=float(muscle_length_info["fiber_passive_force_length_multiplier"][world_id][i].item()),
            damping_multiplier=float(muscle_velocity_info["fiber_damping_force_multiplier"][world_id][i].item()),
            path_length=float(muscle_path_lengths[world_id][i].item()),
            path_velocity=float(muscle_path_velocities[world_id][i].item()),
            fiber_length=float(muscle_length_info["fiber_length"][world_id][i].item()),
            fiber_velocity=float(muscle_velocity_info["fiber_velocity"][world_id][i].item()),
            tendon_length=float(muscle_length_info["tendon_length"][world_id][i].item()),
            pennation_angle=float(muscle_length_info["pennation_angle"][world_id][i].item()),
            moment_arm=muscle_moment_arms[world_id][i].tolist(),
        )
        muscles.append(muscle_data)
    return muscles


def parse_actuator_data(
        m: bolt.Model,
        d: bolt.Data,
        actuation_idx_to_name: dict[int, str],
        world_id: int
) -> list[ActuatorData]:
    actuator_activations = bolt.actuator_activations(d)
    actuator_excitations = bolt.actuator_excitations(d)
    actuator_metadata = bolt.actuator_metadata_np(m)

    actuators = []
    for i in range(bolt.get_num_actuators(m)):
        actuator_data = ActuatorData(
            name=actuation_idx_to_name[i],
            optimal_force=float(actuator_metadata["optimal_force"][i]),
            activation=float(actuator_activations[world_id][i].item()),
            excitation=float(actuator_excitations[world_id][i].item()),
        )
        actuators.append(actuator_data)
    return actuators


def parse_kinetic_data(
        m: bolt.Model,
        d: bolt.Data,
        body_name_to_idx: dict[str, int],
        world_id: int
) -> KineticData:
    # com = bolt.subtree_com_positions(d)[world_id][1].tolist()  # why am I hardcoded!
    body_com_positions = bolt.body_com_positions(d)[world_id]
    grf = bolt.grf(d)[world_id].tolist()
    gravity = bolt.gravity(m).y
    com = bolt.body_subtree_com_positions(d)[world_id, 0].tolist()
    mass = bolt.body_mass(m).sum()

    if "talus_l" in body_name_to_idx and "talus_r" in body_name_to_idx:
        foot_pos_l = tuple(body_com_positions[body_name_to_idx["talus_l"]].tolist())
        foot_pos_r = tuple(body_com_positions[body_name_to_idx["talus_r"]].tolist())
    else:
        foot_pos_l = (0, 0, 0)
        foot_pos_r = (0, 0, 0)

    kinetic_data = KineticData(
        com=tuple(com),
        foot_pos_l=foot_pos_l,
        foot_pos_r=foot_pos_r,
        grf=tuple(grf),
        total_mass=float(mass),
        gravity=gravity,
    )
    return kinetic_data


def find_index_1d(tensor, x):
    idx = torch.where(tensor == x)[0]
    return idx[0].item() if idx.numel() > 0 else None


def parse_joint_angles(
        m: bolt.Model,
        d: bolt.Data,
        qpos_idx_to_name: dict[int, str],
        limit_id_lookup: dict[str, tuple[float, float]],
        world_id: int,
        ref_joint_angles: torch.Tensor | None
) -> list[NamedValue]:
    joint_angles = bolt.joint_positions(d)
    # joint_limit_ranges = bolt.joint_limit_ranges(m)
    # joint_limit_qadr = list(bolt.joint_limit_qadr(m))
    angles = []

    for i in range(bolt.get_num_qpos(m)):
        qpos_name = qpos_idx_to_name[i]
        reference = None if ref_joint_angles is None else float(ref_joint_angles[world_id][i].item())
        limits = limit_id_lookup[qpos_name]

        angle = NamedValue(
            name=qpos_name,
            value=float(joint_angles[world_id][i].item()),
            reference=reference,
            limits=limits,
        )
        angles.append(angle)
    return angles


def parse_joint_velocities(
        m: bolt.Model,
        d: bolt.Data,
        world_id: int
) -> list[NamedValue]:
    joint_velocities = bolt.joint_velocities(d)
    velocities = []  # todo
    return velocities


def parse_joint_moments(
        m: bolt.Model,
        d: bolt.Data,
        dof_idx_to_name: dict[int, str],
        world_id: int
) -> list[JointMoment]:
    net_moment = bolt.joint_moments(d)
    ufrc_spring = bolt.ufrc_spring(d)
    ufrc_damper = bolt.ufrc_damper(d)
    ufrc_muscle = bolt.ufrc_muscle(d)
    ufrc_muscle_passive = bolt.ufrc_muscle_passive(d)
    ufrc_actuator = bolt.ufrc_actuator(d)
    ufrc_limit = bolt.ufrc_limit(d)

    moments = []
    for i in range(bolt.get_num_dofs(m)):
        dof_name = dof_idx_to_name[i]

        angle = JointMoment(
            name=dof_name,
            net_moment=float(net_moment[world_id][i].item()),
            spring=float(ufrc_spring[world_id][i].item()),
            damping=float(ufrc_damper[world_id][i].item()),
            muscle=float(ufrc_muscle[world_id][i].item()),
            muscle_passive=float(ufrc_muscle_passive[world_id][i].item()),
            actuator=float(ufrc_actuator[world_id][i].item()),
            limit=float(ufrc_limit[world_id][i].item()),
        )
        moments.append(angle)
    return moments


def parse_joint_passive_moments(
        m: bolt.Model,
        d: bolt.Data,
        qpos_idx_to_name: dict[int, str],
        world_id: int
) -> list[JointMuscleBreakdown]:
    qfrc_muscle_passive_breakdown = bolt.qfrc_muscle_passive_breakdown(d)
    qfrc_muscle_active_breakdown = bolt.qfrc_muscle_active_breakdown(d)

    moments = []
    for i in range(bolt.get_num_qpos(m)):
        qpos_name = qpos_idx_to_name[i]

        # Build passive muscle breakdown
        passive_breakdown = []
        active_breakdown = []
        for j in range(bolt.get_num_muscles(m)):
            passive_breakdown.append(
                float(qfrc_muscle_passive_breakdown[world_id][i][j].item())
            )
            active_breakdown.append(
                float(qfrc_muscle_active_breakdown[world_id][i][j].item())
            )

        pm = JointMuscleBreakdown(
            name=qpos_name,
            passive_breakdown=passive_breakdown,
            active_breakdown=active_breakdown,
        )
        moments.append(pm)
    return moments


def parse_body_forces(
        m: bolt.Model,
        d: bolt.Data,
        body_idx_to_name: dict[int, str],
        world_id: int
) -> list[BodyForces]:
    body_force_gravity = bolt.body_force_gravity(d)
    body_force_contact = bolt.body_force_contact(d)
    body_force_muscle = bolt.body_force_muscle(d)

    body_forces = []
    for i in range(bolt.get_num_bodies(m)):
        angle = BodyForces(
            name=body_idx_to_name[i],
            gravity=tuple(body_force_gravity[world_id][i].tolist()),
            contact=tuple(body_force_contact[world_id][i].tolist()),
            muscle=tuple(body_force_muscle[world_id][i].tolist()),
        )
        body_forces.append(angle)
    return body_forces


def parse_beam_points(
        m: bolt.Model,
        d: bolt.Data,
        world_id: int
) -> list[PointData]:
    points = []
    beam_point_pos = bolt.get_beam_visual_positions(d)[world_id]
    beam_point_pos = beam_point_pos.reshape(-1, 3)
    for i in range(beam_point_pos.shape[0]):
        point = PointData(
            name=f"beam_point_{i}",
            pos=beam_point_pos[i].tolist(),
        )
        points.append(point)
    return points


def parse_frame(
        m: bolt.Model,
        d: bolt.Data,
        body_name_to_idx: dict[str, int],
        qpos_idx_to_name: dict[int, str],
        qpos_name_to_idx: dict[str, int],
        dof_idx_to_name: dict[int, str],
        limit_id_lookup: dict[str, tuple[float, float]],
        muscle_idx_to_name: dict[int, str],
        actuation_idx_to_name: dict[int, str],
        collider_idx_to_name: dict[int, str],
        visual_load_results: list[bolt.MeshLoadResult],
        world_id: int,
        frame_time: float,
        scene_settings: SceneSettings,
        ref_joint_angles=None,
) -> FrameData:
    body_idx_to_name = {v: k for k, v in body_name_to_idx.items()}
    visuals = parse_visual_data(m, d, visual_load_results, world_id)
    colliders = parse_collider_data(m, d, collider_idx_to_name, world_id)
    muscles = parse_muscle_data(m, d, muscle_idx_to_name, world_id)
    actuators = parse_actuator_data(m, d, actuation_idx_to_name, world_id)
    kinetic_data = parse_kinetic_data(m, d, body_name_to_idx, world_id)
    joint_angles = parse_joint_angles(m, d, qpos_idx_to_name, limit_id_lookup, world_id, ref_joint_angles)
    joint_velocities = parse_joint_velocities(m, d, world_id)
    joint_moments = parse_joint_moments(m, d, dof_idx_to_name, world_id)
    joint_passive_moments = parse_joint_passive_moments(m, d, qpos_idx_to_name, world_id)
    body_forces = parse_body_forces(m, d, body_idx_to_name, world_id)
    beam_points = parse_beam_points(m, d, world_id)

    frame_visuals = FrameData(
        time=frame_time,
        scene_settings=scene_settings,
        visuals=visuals,
        colliders=colliders,
        joint_angles=joint_angles,
        joint_velocities=joint_velocities,
        joint_moments=joint_moments,
        joint_muscle_breakdown=joint_passive_moments,
        body_forces=body_forces,
        muscles=muscles,
        actuators=actuators,
        kinetic_data=kinetic_data,
        beam_points=beam_points,
        arrows=[],
        targets=[],
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


def add_target(
        frame: FrameData,
        pos: list,
        radius: float,
        active: bool
):
    target = TargetData(
        pos=pos,
        rot=[1.0, 0.0, 0.0, 0.0],
        radius=radius,
        active=active,
    )
    frame.targets.append(target)


def add_ext_forces_to_frame(
        frame: FrameData,
        m: bolt.Model,
        d: bolt.Data,
        idx_world: int
):
    num_bodies = bolt.get_num_bodies(m)
    body_transforms = bolt.body_transforms(d)
    body_positions = get_position_from_transform(body_transforms)
    ext_forces = bolt.body_user_forces(d)
    for i in range(1, num_bodies):
        force = ext_forces[idx_world][i][3:6].tolist()
        point = body_positions[idx_world][i].tolist()
        arrow = Arrow(
            start=point,
            direction=force
        )
        frame.arrows.append(arrow)
