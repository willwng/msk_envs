"""
World axes, quaternions and transforms.

Quaternions are in (x, y, z, w) format. Transforms are (position, quaternion), i.e. [..., 7].
"""
import torch

FWD_IDX = 0  # x-axis is the lane direction
UP_IDX = 1  # y-axis is the up direction
SIDE_IDX = 2  # z-axis is orthogonal to the lane direction


def build_axis(axis: int, scale: float) -> list[float]:
    a = [0.0, 0.0, 0.0]
    a[axis] = scale
    return a


def rotate_vec(rot: torch.Tensor, v: torch.Tensor):
    pure = rot[..., :3]
    scalar = rot[..., 3:]

    pure_x_v = torch.cross(pure, v, dim=-1)
    pure_x_pure_x_v = torch.cross(pure, pure_x_v, dim=-1)
    return v + 2.0 * (pure_x_v * scalar + pure_x_pure_x_v)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(q, dim=-1, keepdim=True)
    return q / norm


def get_position_from_transform(transform: torch.Tensor) -> torch.Tensor:
    return transform[..., :3]


def get_rotation_from_transform(transform: torch.Tensor) -> torch.Tensor:
    return transform[..., 3:]


def compute_ground_height(position: torch.Tensor, ground_rotation: torch.Tensor) -> torch.Tensor:
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
