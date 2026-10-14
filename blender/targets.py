import math

import bpy
import mathutils

from material import create_target_material
from util import vec_yup_zup, quat_xyzw_to_wxyz

target_materials = {}
""" By active flag, created on first use """


def get_target_material(active: bool):
    if active not in target_materials:
        target_materials[active] = create_target_material(active=active)
    return target_materials[active]


def create_target(name, active):
    print(f"Creating target {name}")
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        location=(0, 0, 0)
    )
    target = bpy.context.active_object
    target.name = name
    target.hide_render = False
    target.hide_viewport = False
    target.data.materials.append(get_target_material(active))
    bpy.ops.object.shade_smooth()
    return target


def update_target(name, target, frame_num):
    # Fetch object data
    pos = vec_yup_zup(target["pos"])
    radius = target["radius"]
    # Rotation y-up to z-up
    quat_data = target["rot"]
    rotation = mathutils.Quaternion(quat_xyzw_to_wxyz(quat_data))
    q_conv = mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(90.0))
    rotation = q_conv @ rotation

    # Update or create object
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = create_target(name, active=target["active"])

    obj.location = pos
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = rotation
    obj.scale = (radius, radius, radius)

    obj.keyframe_insert(data_path="location", frame=frame_num)
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
    obj.keyframe_insert(data_path="scale", frame=frame_num)


def clear_target_cache():
    target_materials.clear()
