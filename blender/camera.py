""" A camera that follows the model's (smoothed) center of mass, aimed by a Track To constraint """
import bpy
import numpy as np

from util import animate, vec_yup_zup

CAMERA_OFFSETS = {
    "side": lambda p: [p[0] + 1.0, p[1] - 7.5, p[2] + 0.75],
    "olympic": lambda p: [0.8 * (p[0] + 10.0), p[1] - 10.0, p[2] + 3.0],
    "front": lambda p: [p[0] + 7.5, p[1], p[2] + 1.0],
    "close": lambda p: [p[0] + 0.6, p[1] - 3.2, p[2] + 0.35],
    "tracking": lambda p: [p[0] - 4.0, p[1] - 4.0, p[2] + 1.0],
}
""" Camera location for each view, given the (z-up) position it follows """
LENS = {"side": 50, "olympic": 50, "front": 50, "close": 50, "tracking": 35}


def setup_camera(anim: dict, frames: np.ndarray, view: str = "side"):
    target = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(target)
    cam_data = bpy.data.cameras.new("AnimationCamera")
    cam_data.lens = LENS.get(view, 50)
    cam_data.clip_start = 0.05
    cam_data.clip_end = 1000
    camera = bpy.data.objects.new("AnimationCamera", cam_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    track = camera.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    com = np.array([vec_yup_zup(f["cam_pos"]) for f in anim["frames"]])
    animate(target, "location", frames, com)
    animate(camera, "location", frames, np.array([CAMERA_OFFSETS[view](p) for p in com]))
    return camera
