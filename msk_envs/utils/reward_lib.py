import torch

from msk_envs.utils.global_params import UP_IDX, SIDE_IDX, MIN_ROOT_HEIGHT
from msk_envs.utils.quat import rotate_vec


def _get_velocity(body_velocities, body_id: int, linear: bool, idx: int):
    """ Returns the [idx]-velocity of body [body_id] """
    return body_velocities[:, body_id, idx + 3] if linear else body_velocities[:, body_id, idx]


def velocity_reward(body_velocities, body_id: int, coord_idx: int, linear: bool):
    """ Velocity reward """
    root_velocity = _get_velocity(body_velocities, body_id, linear, coord_idx)
    return root_velocity


def mid_lane_reward(root_position: torch.Tensor, weight: float = 2.5):
    """Reward for being in the center of the lane"""
    dist_from_center = torch.abs(root_position[:, SIDE_IDX])
    return torch.exp(-weight * dist_from_center.pow(2))


def joint_penalty(limit_torques: torch.Tensor, squared: bool = False):
    """Joint limit penalty based on sum of absolute limit torques"""
    if squared:
        sq_limit_torque = torch.pow(limit_torques, 2)
        limit_torque_sum = torch.sum(sq_limit_torque, dim=1)
    else:
        abs_limit_torque = torch.abs(limit_torques)
        limit_torque_sum = torch.sum(abs_limit_torque, dim=1)

    return limit_torque_sum


def metabolic_penalty(muscle_powers, num_muscles, squared: bool = False):
    """Metabolic penalty based on total muscle power"""
    if squared:
        muscle_powers_sq = torch.pow(muscle_powers, 2)
        total_power = torch.sum(muscle_powers_sq, dim=1)
    else:
        total_power = torch.sum(muscle_powers, dim=1)

    if num_muscles == 0:
        return torch.zeros_like(total_power)
    return total_power / num_muscles


def activation_square_penalty(muscle_activations):
    """Fatigue penalty based on squared muscle activations"""
    activations_sq = torch.pow(muscle_activations, 2)
    total_fatigue = torch.sum(activations_sq, dim=1)
    return total_fatigue


def self_collision_penalty(collision_forces, force_bound):
    clamped_collision_forces = torch.clamp(collision_forces, min=-force_bound, max=force_bound)
    norm_collision_forces = torch.abs(clamped_collision_forces) / force_bound
    return norm_collision_forces.sum(dim=1)


def compute_ground_vertical(position: torch.Tensor, ground_rotation: torch.Tensor) -> torch.Tensor:
    """ At the given positions, find the ground height at that given (x, z) """
    ground_normal = torch.zeros_like(position)
    ground_normal[:, UP_IDX] = 1.0
    ground_normal = rotate_vec(rot=ground_rotation[torch.newaxis, :], v=ground_normal)

    # zero out the vertical of the position
    pos_horizontal = position.clone()
    pos_horizontal[:, UP_IDX] = 0.0

    # use plane eqn passing through origin: dot(normal, point_on_plane) = 0
    dot_product = torch.sum(ground_normal * pos_horizontal, dim=-1)
    normal_up = ground_normal[:, UP_IDX]
    ground_vertical = -dot_product / (normal_up + 1e-8)
    return ground_vertical


def has_fallen(root_pos: torch.Tensor, ground_rotation: torch.Tensor, min_root=MIN_ROOT_HEIGHT):
    # Find height of root wrst ground
    ground_vertical = compute_ground_vertical(root_pos, ground_rotation)
    root_height = root_pos[:, UP_IDX] - ground_vertical
    fallen = (root_height < min_root)
    return fallen.detach()


def _sigmoids(x, value_at_1):
    """ Just a linear 'sigmoid' for now """
    scale = 1.0 - value_at_1
    scaled_x = x * scale
    return torch.where(
        torch.abs(scaled_x) < 1.0,
        1.0 - scaled_x,
        torch.tensor(0.0, dtype=x.dtype, device=x.device)
    )


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
    """ Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise. """
    lower, upper = bounds
    in_bounds = torch.logical_and(lower <= x, x <= upper)

    if margin == 0:
        value = torch.where(in_bounds,
                            torch.tensor(1.0, dtype=x.dtype, device=x.device),
                            torch.tensor(0.0, dtype=x.dtype, device=x.device))
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        sig_val = _sigmoids(d, value_at_margin)
        value = torch.where(in_bounds, torch.tensor(1.0, dtype=x.dtype, device=x.device), sig_val)
    return value
