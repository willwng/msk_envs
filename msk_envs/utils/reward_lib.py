import torch
from msk_envs.utils.global_params import UP_IDX
from msk_envs.utils.quat import quat_diff_angle


def velocity_reward(body_velocities, body_id: int, coordinate: int, linear: bool):
    """Forward velocity reward (x-velocity of root body)"""
    root_velocity = body_velocities[:, body_id, :]
    return root_velocity[:, coordinate + 3] if linear else root_velocity[:, coordinate]


def joint_limit_penalty(limit_torques):
    """Joint limit penalty based on sum of absolute limit torques"""
    num_limits = limit_torques.shape[1]
    abs_limit_torque = torch.abs(limit_torques)
    abs_limit_torque_sum = torch.sum(abs_limit_torque, dim=1)
    if num_limits == 0:
        return torch.zeros_like(abs_limit_torque_sum)
    return abs_limit_torque_sum / num_limits


def actuator_penalty(actuator_activations, num_actuators):
    """Actuator penalty based on squared activation deviation from 0.5"""
    actuator_act = (actuator_activations - 0.5) * 2.0
    squared_act = torch.pow(actuator_act, 2)
    mean_squared_act = torch.sum(squared_act, dim=1) / num_actuators
    if num_actuators == 0:
        return torch.zeros_like(mean_squared_act)
    return mean_squared_act


def max_vertical_reward(body_positions):
    """Maximum vertical height achieved (current max across all bodies)"""
    current_max_height = torch.max(body_positions[:, :, UP_IDX], dim=1)[0]
    return current_max_height


def joint_angle_track_reward(joint_positions, target_joint_positions, qpos_adr, weight):
    """Imitation reward for joint tracking"""
    # If qpos_adr can be either a range or an integer
    if isinstance(qpos_adr, range):
        _range = qpos_adr
    else:
        _range = range(qpos_adr, qpos_adr + 1)

    joint_values = joint_positions[:, _range]
    target_values = target_joint_positions[:, _range]

    mse = (joint_values - target_values).pow(2).mean(dim=1)
    reward = torch.exp(-weight * mse)
    return reward


def body_pos_track_reward(body_positions, target_body_positions, body_id, weight):
    """Imitation reward for body position tracking"""
    # If body_id can be either a range or an integer
    if isinstance(body_id, range) or isinstance(body_id, list):
        _range = body_id
    else:
        _range = range(body_id, body_id + 1)
    
    body_pos_values = body_positions[:, _range, :]
    target_pos_values = target_body_positions[:, _range, :]
    mse = (body_pos_values - target_pos_values).pow(2).mean(dim=(1, 2))
    reward = torch.exp(-weight * mse)
    return reward

def body_rot_track_reward(body_rotations, target_body_rotations, body_id, weight):
    # If body_id can be either a range or an integer
    if isinstance(body_id, range) or isinstance(body_id, list):
        _range = body_id
    else:
        _range = range(body_id, body_id + 1)
    
    body_rot_values = body_rotations[:, _range, :]
    target_rot_values = target_body_rotations[:, _range, :]
    rot_diff_angle = quat_diff_angle(body_rot_values, target_rot_values)
    mse = rot_diff_angle.pow(2).mean(dim=1)
    reward = torch.exp(-weight * mse)
    return reward

def update_dict(reward_dict, key, value):
    if key in reward_dict:
        reward_dict[key] += value.detach()
    else:
        reward_dict[key] = value.detach()