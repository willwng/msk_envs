import torch

from msk_envs.geometry import UP_IDX, SIDE_IDX, compute_ground_height

MIN_ROOT_HEIGHT = 0.6
""" Root height (above the ground) below which an episode terminates """


def _get_velocity(body_velocities, body_id: int, linear: bool, idx: int):
    """ Returns the [idx]-velocity of body [body_id] """
    return body_velocities[:, body_id, idx + 3] if linear else body_velocities[:, body_id, idx]


def velocity_reward(body_velocities, body_id: int, coord_idx: int, linear: bool):
    """ Velocity reward (a copy, so it keeps this step's value after the simulator state changes) """
    return _get_velocity(body_velocities, body_id, linear, coord_idx).clone()


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


def muscle_passive_penalty(muscle_passive_multipliers: torch.Tensor, threshold: float, squared: bool = False):
    """ Compute sum of passive force multipliers above [threshold], ignores multipliers below [threshold] """
    passive_force_excess = torch.clamp(muscle_passive_multipliers - threshold, min=0.0)
    excess_penalty = torch.pow(passive_force_excess, 2) if squared else torch.abs(passive_force_excess)
    return excess_penalty.sum(dim=1)


def has_fallen(root_pos: torch.Tensor, ground_rotation: torch.Tensor, min_root=MIN_ROOT_HEIGHT):
    # Find height of root wrst ground
    ground_vertical = compute_ground_height(root_pos, ground_rotation)
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
