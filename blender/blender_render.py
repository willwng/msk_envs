"""
Import a recorded trajectory into Blender as an animation (track, model meshes, muscles, shoes, camera).

From the command line (runs headless and saves the scene):

    blender -b -P blender/blender_render.py -- --traj <trajectory.json[.gz]> --geometry <bolt geometry dir> \
        --save scene.blend [--camera side|olympic|front] [--no-track]

Or open this file in Blender's Scripting tab, set TRAJ_PATH and GEOMETRY_PATH below, and run it.
"""
import argparse
import importlib
import os
import sys

import bpy

TRAJ_PATH = ""
""" Trajectory to import when run from Blender's UI """
GEOMETRY_PATH = ""
""" bolt's geometry directory (meshes), when run from Blender's UI """


def _script_dir() -> str:
    # Run with -P, __file__ is this file. Run from the Scripting tab, it is the text block's name, so fall back to the
    # directory of the open .blend file.
    path = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(path, "scene.py")):
        return path
    return os.path.dirname(bpy.data.filepath)


SCRIPT_DIR = _script_dir()
for path in [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)]:  # this folder's modules, and the repo (for msk_envs)
    if path not in sys.path:
        sys.path.append(path)

# Blender keeps modules loaded between runs of a script; reload them so edits take effect
import camera, colliders, material, muscle, options, scene, targets, track, util, visuals  # noqa: E401, E402

for module in [util, material, options, scene, camera, track, visuals, colliders, targets, muscle]:
    importlib.reload(module)

from msk_envs.trajectory_io import load_trajectory, trajectory_fps  # noqa: E402

SHOE_COLLIDER_PREFIXES = ("left_", "right_")
""" Only colliders with these name prefixes (the shoes) are drawn """


def parse_args():
    parser = argparse.ArgumentParser(description="Import a recorded trajectory as a Blender animation")
    parser.add_argument("--traj", default=TRAJ_PATH, help="Trajectory file (.json or .json.gz)")
    parser.add_argument("--geometry", default=GEOMETRY_PATH, help="bolt's geometry directory")
    parser.add_argument("--camera", default="side", choices=sorted(camera.CAMERA_OFFSETS))
    parser.add_argument("--no-track", action="store_true", help="Don't build the running track")
    parser.add_argument("--save", default="", help="Save the scene to this .blend file")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if not args.traj or not args.geometry:
        parser.error("set --traj and --geometry (or TRAJ_PATH and GEOMETRY_PATH at the top of this script)")
    return args


def main():
    args = parse_args()
    frames = load_trajectory(args.traj)

    scene.reset_scene()
    visuals.clear_visual_cache()
    colliders.clear_collider_cache()
    targets.clear_target_cache()
    options.setup_renderer(fps=int(round(trajectory_fps(frames))))
    cam = camera.setup_camera()
    scene.hide_objects()
    if not args.no_track:
        track.setup_track()

    for frame_index, frame_data in enumerate(frames):
        frame_num = frame_index + 1  # Blender frames start at 1
        bpy.context.scene.frame_set(frame_num)

        camera.update_camera(cam, frame_data["cam_pos"], frame_num, args.camera)

        for visual in frame_data["visuals"]:
            visuals.update_visual(visual, frame_num, args.geometry)

        for collider in frame_data["colliders"]:
            if collider["name"].startswith(SHOE_COLLIDER_PREFIXES):
                colliders.update_collider(collider, frame_num)

        for muscle_data in frame_data["muscles"]:
            muscle.update_muscle(muscle_data, frame_num)

        for i, target in enumerate(frame_data.get("targets", [])):
            targets.update_target(f"target_{i}", target, frame_num)

        print(f"Built frame: {frame_num} of {len(frames)}")

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(frames)
    print(f"Animation created with {len(frames)} frames")

    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))
        print(f"Saved scene to {args.save}")


if __name__ == "__main__":
    main()
