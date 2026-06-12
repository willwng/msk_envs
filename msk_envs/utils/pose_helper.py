from dataclasses import dataclass

import torch
import yaml

from msk_envs.utils.global_params import UP_IDX
from msk_envs.utils.reward_lib import compute_ground_height


@dataclass
class SwapPair:
    """ Represents a pair of left/right dofs that can be swapped under symmetry """
    qpos_r: int
    qpos_l: int
    dof_r: int
    dof_l: int


def parse_starting_pose(
        file_path,
        qpos_id_lookup: dict[str, int],
        dof_id_lookup: dict[str, int],
        num_qpos: int,
        num_dofs: int,
):
    """ Parse starting pose from YAML file"""
    start_q, start_qv = [0.0] * num_qpos, [0.0] * num_dofs

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    for coord_name, values in data.items():
        if coord_name not in qpos_id_lookup:
            print(f"Warning: Coordinate '{coord_name}' not found in model's coordinates, skipping.")
        else:
            qpos_adr = qpos_id_lookup[coord_name]
            qpos = values["q"]
            start_q[qpos_adr] = qpos

        if coord_name not in dof_id_lookup:
            print(f"Warning: Coordinate '{coord_name}' not found in model's speeds, skipping.")
        else:
            dof_adr = dof_id_lookup[coord_name]
            qvel = values["v"]
            start_qv[dof_adr] = qvel

    return start_q, start_qv


def get_swap_left_right_data(
        qpos_id_lookup: dict[str, int],
        dof_id_lookup: dict[str, int],
) -> list[SwapPair]:
    """ Returns all pairs of dofs/qpos that can be swapped under L/R symmetry """
    all_coords = list(qpos_id_lookup.keys())
    # remove all "_r" and "_l" suffixes to get unique qpos names
    swappable_coords = [body_name[:-2] for body_name in all_coords if body_name.endswith(("_r", "_l"))]
    swappable_coords = list(set(swappable_coords))

    swap_data = []
    for coord in swappable_coords:
        swap_data.append(
            SwapPair(
                qpos_l=qpos_id_lookup[f"{coord}_l"],
                qpos_r=qpos_id_lookup[f"{coord}_r"],
                dof_l=dof_id_lookup[f"{coord}_l"],
                dof_r=dof_id_lookup[f"{coord}_r"],
            )
        )
    return swap_data


class StartingStateHelper:
    def __init__(
            self,
            starting_pose_path: str,
            qpos_id_lookup: dict[str, int],
            dof_id_lookup: dict[str, int],
            num_qpos: int,
            num_dofs: int,
            num_muscles: int,
            num_envs: int,
            apply_start_noise: bool,
            apply_swap_lr: bool,
            q_noise: float,
            qv_noise: float,
            default_activation: float,
            device: torch.device,
    ):
        self.num_qpos = num_qpos
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles
        self.num_envs = num_envs
        self.device = device
        # Noise settings
        self.apply_start_noise = apply_start_noise
        self.apply_swap_lr = apply_swap_lr
        self.q_noise = q_noise
        self.qv_noise = qv_noise

        # Parse the initial given starting pose
        q, qv = parse_starting_pose(
            starting_pose_path, qpos_id_lookup, dof_id_lookup, self.num_qpos, self.num_dofs)
        assert len(q) == self.num_qpos and len(qv) == self.num_dofs
        q_torch = torch.tensor(q, dtype=torch.float32, device=device)
        qv_torch = torch.tensor(qv, dtype=torch.float32, device=device)
        self.start_pose_base = q_torch.unsqueeze(0)
        self.start_velocity_base = qv_torch.unsqueeze(0)

        # Collect the pairs of left/right coordinates that can be swapped
        if self.apply_swap_lr:
            self.swap_lr_data = get_swap_left_right_data(qpos_id_lookup, dof_id_lookup)
        else:
            self.swap_lr_data = []

        # Allocate space for the starting pose for each env
        self.start_pose = q_torch.unsqueeze(0).repeat(num_envs, 1)
        self.start_velocity = qv_torch.unsqueeze(0).repeat(num_envs, 1)

        # Starting muscle activations
        self.default_activation = default_activation
        if default_activation == -1.0:
            self.start_activations = torch.rand((num_envs, self.num_muscles), device=device)
        else:
            self.start_activations = torch.ones(
                (num_envs, self.num_muscles), device=device) * default_activation
        return

    def _apply_starting_pose_noise(self, q: torch.Tensor, qv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Apply noise to given starting pose (q) and velocity (qv) """
        if self.apply_start_noise:
            q += torch.randn_like(q) * self.q_noise
            qv += torch.randn_like(qv) * self.qv_noise
        return q, qv

    def _apply_swap_lr_noise(self, q: torch.Tensor, qv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Randomly swap left and right sides of starting pose (q) and velocity (qv) """
        if self.apply_swap_lr:
            # Randomly choose to swap left and right
            swap_mask = (torch.rand(self.num_envs, device=self.device) > 0.5)
            q_old, qv_old = q.clone(), qv.clone()
            for swap_pair in self.swap_lr_data:
                rq, lq = swap_pair.qpos_r, swap_pair.qpos_l
                q[swap_mask, rq] = q_old[swap_mask, lq]
                q[swap_mask, lq] = q_old[swap_mask, rq]

                rdq, ldq = swap_pair.dof_r, swap_pair.dof_l
                qv[swap_mask, rdq] = qv_old[swap_mask, ldq]
                qv[swap_mask, ldq] = qv_old[swap_mask, rdq]
        return q, qv

    def create_new_starting_poses(self, reset_mask: torch.Tensor):
        """ Create a new starting pose and velocity for envs where reset_mask is 1 """
        # Start with initial starting pose
        q = self.start_pose_base.repeat(self.num_envs, 1)
        qv = self.start_velocity_base.repeat(self.num_envs, 1)
        # Apply noise/swaps
        q, qv = self._apply_starting_pose_noise(q=q, qv=qv)
        q, qv = self._apply_swap_lr_noise(q=q, qv=qv)
        # Set the new starting pose
        self.start_pose[reset_mask, :] = q[reset_mask, :]
        self.start_velocity[reset_mask, :] = qv[reset_mask, :]

        # Noise starting muscle activations
        if self.default_activation == -1.0:
            random_acts = torch.rand((self.num_envs, self.num_muscles), device=self.device)
            self.start_activations[reset_mask, :] = random_acts[reset_mask, :]
        return

    def set_starting_state(
            self,
            time_out: torch.Tensor,
            q_out: torch.Tensor,
            qv_out: torch.Tensor,
            activations_out: torch.Tensor,
            actuator_activations_out: torch.Tensor,
            reset_mask: torch.Tensor,
    ):
        """ Apply new starting state for envs where reset_mask is 1 """
        time_out[reset_mask] = 0.0
        q_out[reset_mask, :] = self.start_pose[reset_mask, :]
        qv_out[reset_mask, :] = self.start_velocity[reset_mask, :]
        activations_out[reset_mask, :] = self.start_activations[reset_mask, :]
        actuator_activations_out[reset_mask, :] = 0.5
        return

    def adjust_for_ground_contact(
            self,
            root_pos: torch.Tensor,
            collider_sizes: torch.Tensor,
            collider_body_id: torch.Tensor,
            collider_positions: torch.Tensor,
            ground_rotation: torch.Tensor,
            reset_mask: torch.Tensor,
            root_height_qpos_id: int,
            ground_id: int,
    ):
        non_ground_collider_ids = torch.where(collider_body_id != ground_id)[0]

        # Need at least one non-root collider
        if non_ground_collider_ids.size(0) > 1:
            ground_height = compute_ground_height(position=root_pos, ground_rotation=ground_rotation)
            collider_heights = collider_positions[:, non_ground_collider_ids, UP_IDX]
            collider_heights -= collider_sizes[non_ground_collider_ids, 0]
            lowest_collider_height = collider_heights.min(dim=1).values
            adjustment = -lowest_collider_height + ground_height
            self.start_pose[reset_mask, root_height_qpos_id] += adjustment[reset_mask]
        return

    def apply(
            self,
            q_out: torch.Tensor,
            qv_out: torch.Tensor,
            activations_out: torch.Tensor,
            actuator_activations_out: torch.Tensor,
            reset_mask: torch.Tensor,
    ):
        """ Modify starting pose and velocity for envs where reset_mask is 1 """
        q_out[reset_mask, :] = self.start_pose[reset_mask, :]
        qv_out[reset_mask, :] = self.start_velocity[reset_mask, :]
        activations_out[reset_mask, :] = self.start_activations[reset_mask, :]
        actuator_activations_out[reset_mask, :] = 0.5
        return
