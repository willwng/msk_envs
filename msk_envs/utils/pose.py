import yaml
import bolt
from dataclasses import dataclass


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


def get_base_name(name: str) -> str:
    """ Get base name by removing _r/_l suffix """
    if name.endswith("_r") or name.endswith("_l"):
        return name[:-2]
    return name
