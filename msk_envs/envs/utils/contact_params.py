from dataclasses import dataclass

import yaml
import math


@dataclass
class ContactParam:
    geom_name: str
    stiffness: float
    dissipation: float
    priority: int

    static_friction: float
    dynamic_friction: float
    viscous_friction: float
    transition_velocity: float


def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def parse_contact_params(file_path: str) -> list[ContactParam]:
    """ Load Hunt-Crossley contact params from YAML """
    data = load_yaml(file_path)
    params = []
    for geom_name, geom_data in data.items():
        param = ContactParam(
            geom_name=geom_name,
            stiffness=math.pow(float(geom_data["stiffness"]), 2.0 / 3.0),
            dissipation=float(geom_data["dissipation"]),
            priority=int(geom_data["priority"]),
            static_friction=float(geom_data["static_friction"]),
            dynamic_friction=float(geom_data["dynamic_friction"]),
            viscous_friction=float(geom_data["viscous_friction"]),
            transition_velocity=float(geom_data["transition_velocity"]),
        )
        params.append(param)
    return params
