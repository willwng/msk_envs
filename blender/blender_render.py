"""
Import a recorded trajectory into Blender as an animation: bones, muscles (tubes along their wrapped paths, colored by
activation), contact geometry, muscle wrapping surfaces, targets, command arrows, the track, and a following camera.

From the command line (runs headless and saves the scene; the muscles' point cache is written next to it):

    blender -b -P blender/blender_render.py -- --traj <trajectory.json[.gz]> --geometry <bolt geometry dir> \
        --save scene.blend [--camera side|olympic|front|close|tracking] [--no-track] [--wrap-surfaces] \
        [--colliders shoes|all|none] [--muscle-color activation|excitation|force] [--muscle-scale 1.0] \
        [--muscle-style capsules|tubes|lines] [--no-tendons] [--muscles "vas* glut*"] \
        [--render out/frame_ --frames 1-100] \
        [--engine CYCLES|EEVEE] [--samples 128]

Or open this file in Blender's Scripting tab, set TRAJ_PATH and GEOMETRY_PATH below, and run it.
The muscles' thickness can be changed after import in the "Tubes" modifier of the Muscles object.
"""
import argparse
import fnmatch
import importlib
import os
import sys
import tempfile
import time

import bpy
import numpy as np

TRAJ_PATH = ""
""" Trajectory to import when run from Blender's UI """
GEOMETRY_PATH = ""
""" bolt's geometry directory (meshes), when run from Blender's UI """


def _script_dir() -> str:
    # Run with -P, __file__ is this file. Run from the Scripting tab, it is the text block's name, so fall back to the
    # directory of the open .blend file.
    path = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(path, "muscle.py")):
        return path
    return os.path.dirname(bpy.data.filepath)


SCRIPT_DIR = _script_dir()
for path in [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)]:  # this folder's modules, and the repo (for msk_envs)
    if path not in sys.path:
        sys.path.append(path)

# Blender keeps modules loaded between runs of a script; reload them so edits take effect
import camera, colliders, material, muscle, options, scene, targets, track, util, visuals  # noqa: E401, E402
import msk_envs.muscle_style, msk_envs.trajectory_io  # noqa: E401, E402

for module in [msk_envs.muscle_style, msk_envs.trajectory_io, util, material, options, scene, camera, track,
               visuals, colliders, targets, muscle]:
    importlib.reload(module)

from msk_envs.trajectory_io import load_animation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Import a recorded trajectory as a Blender animation")
    parser.add_argument("--traj", default=TRAJ_PATH, help="Trajectory file (.json or .json.gz)")
    parser.add_argument("--geometry", default=GEOMETRY_PATH, help="bolt's geometry directory")
    parser.add_argument("--camera", default="side", choices=sorted(camera.CAMERA_OFFSETS))
    parser.add_argument("--no-track", action="store_true", help="Don't build the running track (draw a plain ground)")
    parser.add_argument("--wrap-surfaces", action="store_true", help="Draw the muscle wrapping surfaces")
    parser.add_argument("--colliders", default="shoes", choices=["shoes", "all", "none"],
                        help="Contact geometry to draw (shoes: colliders named left_*/right_*)")
    parser.add_argument("--no-bones", action="store_true")
    parser.add_argument("--no-muscles", action="store_true")
    parser.add_argument("--muscles", default="", help="Only these muscles (names with * wildcards, space separated)")
    parser.add_argument("--muscle-color", default="activation", choices=["activation", "excitation", "force"])
    parser.add_argument("--muscle-scale", type=float, default=1.0, help="Muscle thickness")
    parser.add_argument("--muscle-style", default="capsules", choices=list(muscle.STYLES),
                        help="tubes: shaped bellies and tendons; capsules: a capsule per path segment; "
                             "lines: thin tubes of constant radius")
    parser.add_argument("--no-tendons", action="store_true", help="Draw the whole muscle-tendon path as fibers, in the muscle color")
    parser.add_argument("--frames", default="", help="Import only these frames (first-last, 1-based)")
    parser.add_argument("--save", default="", help="Save the scene to this .blend file")
    parser.add_argument("--cache-dir", default="",
                        help="Where to write the muscles' point cache (default: next to --save, or a temp dir)")
    parser.add_argument("--render", default="", help="Render the animation to this path prefix (e.g. out/frame_)")
    parser.add_argument("--engine", default="CYCLES", choices=["CYCLES", "EEVEE"])
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--motion-blur", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if not args.traj or not args.geometry:
        parser.error("set --traj and --geometry (or TRAJ_PATH and GEOMETRY_PATH at the top of this script)")
    return args


def select_frames(anim: dict, spec: str) -> dict:
    if not spec:
        return anim
    first, _, last = spec.partition("-")
    first = max(1, int(first))
    last = int(last) if last else len(anim["frames"])
    return {**anim, "frames": anim["frames"][first - 1:last]}


def main():
    args = parse_args()
    t0 = time.time()
    anim = select_frames(load_animation(args.traj), args.frames)
    n_frames = len(anim["frames"])
    frames = np.arange(1, n_frames + 1, dtype=np.float64)  # Blender frames start at 1
    print(f"Loaded {args.traj}: {n_frames} frames at {anim['fps']:.1f} Hz, {len(anim['muscles'])} muscles, "
          f"{len(anim.get('wrap_surfaces', []))} wrap surfaces"
          + (" (old format: muscle paths through their points)" if anim.get("source_version") == 1 else ""))

    scene.reset_scene()
    visuals.clear_visual_cache()
    width, height = (int(x) for x in args.resolution.lower().split("x"))
    options.setup_renderer(fps=anim["fps"], engine=args.engine, samples=args.samples, resolution=(width, height),
                           motion_blur=args.motion_blur)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, n_frames

    root = util.create_root()
    model_col = util.collection("Model")
    env_col = util.collection("Environment")
    camera.setup_camera(anim, frames, args.camera)
    if args.no_track:
        colliders.build_ground(env_col, material.create_ground_material())
    else:
        track.setup_track()

    if not args.no_bones:
        objs = visuals.build_visuals(anim, root, util.collection("Bones", model_col), material.create_bone_material(),
                                     args.geometry, frames)
        print(f"Bones: {len(objs)} objects ({time.time() - t0:.1f} s)")

    if not args.no_muscles:
        only = None
        if args.muscles:
            patterns = args.muscles.split()
            only = {i for i, m in enumerate(anim["muscles"]) if any(fnmatch.fnmatch(m["name"], p) for p in patterns)}
        cache_dir = args.cache_dir or (os.path.join(os.path.dirname(os.path.abspath(args.save)),
                                                    os.path.splitext(os.path.basename(args.save))[0] + "_cache")
                                       if args.save else tempfile.mkdtemp(prefix="msk_blender_"))
        obj = muscle.build_muscles(anim, root, model_col, material.create_muscle_material(), cache_dir,
                                   color_by=args.muscle_color, thickness=args.muscle_scale, only=only,
                                   style=args.muscle_style, tendons=not args.no_tendons)
        if obj is not None:
            print(f"Muscles: point cache {obj['cache_file']} ({time.time() - t0:.1f} s)")

    objs = colliders.build_colliders(anim, root, util.collection("Colliders", model_col),
                                     material.create_collider_material(), frames, args.colliders)
    print(f"Colliders: {len(objs)}")
    if args.wrap_surfaces:
        objs = colliders.build_wrap_surfaces(anim, root, util.collection("WrapSurfaces", model_col),
                                             material.create_wrap_material(), frames)
        print(f"Wrap surfaces: {len(objs)}")
    overlay_col = util.collection("Overlays", model_col)
    targets.build_targets(anim, root, overlay_col, frames)
    targets.build_guides(anim, root, overlay_col, frames)

    sc.frame_set(1)
    print(f"Animation created with {n_frames} frames in {time.time() - t0:.1f} s")

    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))
        print(f"Saved scene to {args.save}")
    if args.render:
        sc.render.filepath = os.path.abspath(args.render)
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
