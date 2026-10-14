"""
A trajectory is a list of frames, each a dict with keys: time, scene_settings, visuals, etc.
    Positions are (x, y, z) with y up, and rotations are
    quaternions in (x, y, z, w) order
"""
import gzip
import json

GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid"}
""" Collider geom_type values (bolt.GeomType) that the viewers draw """


def _open(path: str, mode: str):
    return gzip.open(path, mode + "t") if str(path).endswith(".gz") else open(path, mode)


def load_trajectory(path: str) -> list[dict]:
    with _open(path, "r") as f:
        return json.load(f)


def save_trajectory(frames: list[dict], path: str) -> None:
    with _open(path, "w") as f:
        json.dump(frames, f, indent=2)


def trajectory_fps(frames: list[dict], default: float = 30.0) -> float:
    """ Playback rate matching the recording rate (frames are evenly spaced in time) """
    if len(frames) < 2 or frames[1]["time"] <= frames[0]["time"]:
        return default
    return 1.0 / (frames[1]["time"] - frames[0]["time"])
