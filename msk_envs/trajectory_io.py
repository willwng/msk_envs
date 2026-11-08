"""
Animation files read by the dashboard and Blender. Positions are (x, y, z) with y up, and rotations are quaternions in
(x, y, z, w) order. Files are JSON, gzipped when the name ends with .gz.

Version 2 (written by msk_envs.recording since 2026-11): what doesn't change during the episode is stored once,
and each frame holds flat arrays:

    {
      "format": "msk_animation", "version": 2, "fps": 100.0, "scene_settings": {...},
      "visuals":       [{"mesh_file", "scale", "opacity"}],
      "colliders":     [{"name", "geom_type", "size"}],              # geom_type: bolt.GeomType, see GEOM_TYPES
      "muscles":       [{"name", "max_isometric_force", "optimal_fiber_length", "tendon_slack_length",
                         "num_points"}],
      "wrap_surfaces": [{"name", "wrap_type", "size", "half_length"}],  # wrap_type: see WRAP_TYPES
      "frames": [{
          "time",
          "visuals":       [[px, py, pz, qx, qy, qz, qw]],           # per visual
          "colliders":     [[px, py, pz, qx, qy, qz, qw, contact force]],
          "wrap_surfaces": [[px, py, pz, qx, qy, qz, qw]],
          "muscle_points": [x, y, z, ...],     # every muscle's polyline in order, muscles[i].num_points points each
          "muscles": {"activation": [...], "excitation": [...], "actuation": [...], "path_length": [...],
                      "fiber_length": [...], "tendon_length": [...], "pennation_angle": [...]},  # per muscle
          "beam_points": [[x, y, z]], "arrows": [...], "targets": [...], "guides": [...],
          "cam_pos": [x, y, z], "foot_l_pos": [x, y, z], "foot_r_pos": [x, y, z]
      }]
    }

Version 1 files (a list of frames, each with lists of visual/collider/muscle dicts) are converted on load by
load_animation.
"""
import gzip
import json

import numpy as np

FORMAT = "msk_animation"
VERSION = 2

GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}
""" Collider geom_type values (bolt.GeomType) that the viewers draw """
WRAP_TYPES = {0: "sphere", 1: "cylinder", 2: "ellipsoid", 3: "torus"}
""" Wrapping surface wrap_type values (bolt.WrapType) """

MUSCLE_FRAME_FIELDS = ("activation", "excitation", "actuation", "path_length", "fiber_length", "tendon_length",
                       "pennation_angle")
""" Per-muscle values stored in each frame """

DECIMALS = 5
""" Of every stored value (10 um for positions) """


def _open(path: str, mode: str):
    return gzip.open(path, mode + "t") if str(path).endswith(".gz") else open(path, mode)


def dumps(data) -> str:
    """ Compact JSON, as saved """
    return json.dumps(data, separators=(",", ":"))


def is_current_format(path: str) -> bool:
    """ Whether the file at [path] is a version 2 animation, from its first bytes (dumps writes "format" first) """
    with _open(path, "r") as f:
        head = f.read(256)
    return f'"format":"{FORMAT}"' in head.replace(" ", "")


def load_trajectory(path: str):
    """ The raw JSON content: a version 1 list of frames, or a version 2 dict """
    with _open(path, "r") as f:
        return json.load(f)


def save_trajectory(data, path: str) -> None:
    with _open(path, "w") as f:
        f.write(dumps(data))


def load_animation(path: str) -> dict:
    """ An animation file as a version 2 dict, whatever its version """
    return to_animation(load_trajectory(path))


def to_animation(data) -> dict:
    if isinstance(data, dict) and data.get("format") == FORMAT:
        if data.get("version", VERSION) > VERSION:
            raise ValueError(f"Animation version {data['version']} is newer than this reader ({VERSION})")
        return data
    anim = frames_to_animation(data)
    anim["source_version"] = 1  # converted from an old recording (paths through points, no tendon lengths)
    return anim


def trajectory_fps(frames: list[dict], default: float = 30.0) -> float:
    """ Playback rate matching the recording rate (frames are evenly spaced in time) """
    if len(frames) < 2 or frames[1]["time"] <= frames[0]["time"]:
        return default
    return 1.0 / (frames[1]["time"] - frames[0]["time"])


def _round(values) -> list:
    return np.round(np.asarray(values, dtype=np.float64), DECIMALS).tolist()


def continuous_quats(q: np.ndarray) -> np.ndarray:
    """
    Flip the signs of quaternions [n_frames, ..., 4] (in place, any component order) so that consecutive frames
    interpolate the short way
    """
    for i in range(1, len(q)):
        flip = np.sum(q[i] * q[i - 1], axis=-1) < 0.0
        q[i][flip] *= -1.0
    return q


def _transforms(frames: list[dict], key: str) -> np.ndarray:
    """ [n_frames, n, 7] pos + rot of the objects frames[*][key] """
    n = len(frames[0].get(key, []))
    out = np.zeros((len(frames), n, 7))
    for i, frame in enumerate(frames):
        objs = frame.get(key, [])
        if len(objs) != n:
            raise ValueError(f"The number of {key} changes between frames ({n} -> {len(objs)})")
        if n:
            out[i] = [list(o["pos"]) + list(o["rot"]) for o in objs]
    continuous_quats(out[..., 3:7])
    return out


def frames_to_animation(frames: list[dict]) -> dict:
    """ Version 2 animation from a list of version 1 frames (see the module docstring) """
    if not frames:
        return {"format": FORMAT, "version": VERSION, "fps": 30.0, "scene_settings": {}, "visuals": [],
                "colliders": [], "muscles": [], "wrap_surfaces": [], "frames": []}
    first = frames[0]

    muscles = []
    for k, m in enumerate(first.get("muscles", [])):
        muscles.append({
            "name": m["name"],
            "max_isometric_force": m.get("max_isometric_force", 1000.0),
            "optimal_fiber_length": m.get("optimal_fiber_length"),
            "tendon_slack_length": m.get("tendon_slack_length"),
            "num_points": max(len(frame["muscles"][k]["points"]) for frame in frames),
        })
    visual_X = _transforms(frames, "visuals")
    collider_X = _transforms(frames, "colliders")
    wrap_X = _transforms(frames, "wrap_surfaces")

    out_frames = []
    for i, frame in enumerate(frames):
        points = []
        for k, m in enumerate(frame.get("muscles", [])):
            pts = list(m["points"])
            pts += [pts[-1]] * (muscles[k]["num_points"] - len(pts))  # constant point count (old recordings)
            points.extend(pts)
        colliders = frame.get("colliders", [])
        collider_rows = np.concatenate([collider_X[i], [[c.get("contact_force", 0.0)] for c in colliders]], axis=1) \
            if colliders else np.zeros((0, 8))
        out = {
            "time": frame["time"],
            "visuals": _round(visual_X[i]),
            "colliders": _round(collider_rows),
            "wrap_surfaces": _round(wrap_X[i]),
            "muscle_points": _round(np.reshape(points, -1)),
            "muscles": {key: _round([m.get(key, 0.0) for m in frame.get("muscles", [])]) for key in MUSCLE_FRAME_FIELDS},
            "beam_points": [_round(p["pos"]) for p in frame.get("beam_points", [])],
            "arrows": frame.get("arrows", []),
            "targets": frame.get("targets", []),
            "guides": frame.get("guides", []),
        }
        com = frame.get("cam_pos", [0.0, 1.0, 0.0])
        out["cam_pos"] = _round(com)
        out["foot_l_pos"] = _round(frame.get("foot_l_pos", [com[0], com[1] - 1.0, com[2]]))
        out["foot_r_pos"] = _round(frame.get("foot_r_pos", [com[0], com[1] - 1.0, com[2]]))
        out_frames.append(out)

    return {
        "format": FORMAT,
        "version": VERSION,
        "fps": trajectory_fps(frames),
        "scene_settings": first.get("scene_settings") or {},
        "visuals": [{"mesh_file": v["mesh_file"], "scale": v.get("scale", [1.0, 1.0, 1.0]),
                     "opacity": v.get("opacity", 1.0)} for v in first.get("visuals", [])],
        "colliders": [{"name": c["name"], "geom_type": c["geom_type"], "size": c["scale"]}
                      for c in first.get("colliders", [])],
        "muscles": muscles,
        "wrap_surfaces": [{"name": w["name"], "wrap_type": w["wrap_type"], "size": w["size"],
                           "half_length": w.get("half_length", 0.0)} for w in first.get("wrap_surfaces", [])],
        "frames": out_frames,
    }
