"""
The animation JSON read by the dashboard and Blender: a list of frames, each with the scene settings, visuals,
colliders, muscles, and camera/foot positions for following the model (format: see msk_envs/trajectory_io.py).
"""
import numpy as np

from msk_envs.trajectory_io import save_trajectory
from .frame_data import FrameData, to_json_value


def track_com_smooth_window(frame_data: list[FrameData], window_size: int = 10) -> list[list[float]]:
    """ Center of mass at each frame, averaged over a window of [window_size] frames """
    com_positions = []
    num_frames = len(frame_data)
    half_window = window_size // 2
    for i in range(num_frames):
        start_idx = max(0, i - half_window)
        end_idx = min(num_frames, i + half_window + 1)
        window_coms = [
            np.array(frame_data[j].kinetic_data.com)
            for j in range(start_idx, end_idx)
        ]
        avg_pos = sum(window_coms) / len(window_coms)
        com_positions.append(avg_pos.tolist())
    return com_positions


def create_animation_json(frame_data: list[FrameData], out_file: str):
    """ Dump all relevant animation data to json (gzipped if out_file ends with .gz) """
    cam_positions = track_com_smooth_window(frame_data)

    stacked_frames = []
    for frame, cam_pos in zip(frame_data, cam_positions):
        stacked_frames.append({
            "scene_settings": to_json_value(frame.scene_settings),
            "time": frame.time,
            "visuals": to_json_value(frame.visuals),
            "beam_points": to_json_value(frame.beam_points),
            "colliders": to_json_value(frame.colliders),
            "muscles": [muscle.to_anim_dict() for muscle in frame.muscles],
            "arrows": to_json_value(frame.arrows),
            "targets": to_json_value(frame.targets),
            "cam_pos": list(cam_pos),
            "foot_l_pos": list(frame.kinetic_data.foot_pos_l),
            "foot_r_pos": list(frame.kinetic_data.foot_pos_r),
        })

    save_trajectory(stacked_frames, out_file)
