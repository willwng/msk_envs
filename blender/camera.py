import bpy
from mathutils import Vector
from util import vec_yup_zup


def setup_camera():
    bpy.ops.object.camera_add(location=(0, -10, 5))
    camera = bpy.context.active_object
    camera.name = "AnimationCamera"

    bpy.context.scene.camera = camera
    camera.data.lens = 50  # 50mm
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000
    return camera


def point_camera_at_target(camera, target_location):
    direction = Vector(target_location) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


CAMERA_OFFSETS = {
    "side": lambda p: [p[0] + 1.0, p[1] - 7.5, p[2] + 0.75],
    "olympic": lambda p: [0.8 * (p[0] + 10.0), p[1] - 10.0, p[2] + 3.0],
    "front": lambda p: [p[0] + 7.5, p[1], p[2] + 1.0],
}
""" Camera location for each view, given the (z-up) position it follows """


def update_camera(camera, camera_pos, frame_idx, view: str = "side"):
    """ Place the camera for [view] relative to [camera_pos] (y-up), looking at it, and keyframe it """
    camera_pos = vec_yup_zup(camera_pos)
    camera.location = CAMERA_OFFSETS[view](camera_pos)
    point_camera_at_target(camera, camera_pos)

    camera.keyframe_insert(data_path="location", frame=frame_idx)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
