import math

import bpy
import mathutils

from material import create_collider_material
from util import vec_yup_zup, quat_xyzw_to_wxyz

collider_material = None
""" Shared by all colliders, created on first use """


def get_collider_material():
    global collider_material
    if collider_material is None:
        collider_material = create_collider_material()
    return collider_material


def create_sphere(name):
    print(f"Creating sphere {name}")
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        location=(0, 0, 0)
    )
    sphere = bpy.context.active_object
    sphere.name = name
    sphere.hide_render = False
    sphere.hide_viewport = False
    sphere.data.materials.append(get_collider_material())
    return sphere


def create_capsule(name, radius=1.0, half_depth=2.0):
    print(f"Creating capsule {name}")
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=half_depth * 2.0,
        location=(0, 0, 0)
    )
    cylinder = bpy.context.active_object

    # Create top sphere
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(0, 0, half_depth)
    )
    sphere_top = bpy.context.active_object

    # Create bottom sphere
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(0, 0, -half_depth)
    )
    sphere_bottom = bpy.context.active_object

    # Select all objects
    bpy.ops.object.select_all(action='DESELECT')

    cylinder.select_set(True)
    sphere_top.select_set(True)
    sphere_bottom.select_set(True)

    bpy.context.view_layer.objects.active = cylinder

    # Join into one mesh
    bpy.ops.object.join()

    capsule = bpy.context.active_object
    capsule.name = name

    capsule.hide_render = False
    capsule.hide_viewport = False

    capsule.data.materials.append(get_collider_material())

    return capsule


def create_collider(name, geom_type, scale):
    """ Spheres and capsules (geom_type 2 and 3, see msk_envs/trajectory_io.py); other types aren't drawn """
    if geom_type == 2:
        return create_sphere(name)
    if geom_type == 3:
        return create_capsule(name, radius=scale[0], half_depth=scale[1])
    return None


def update_collider(collider, frame_num):
    # Fetch object data
    name = collider["name"]
    geom_type = collider["geom_type"]
    pos = vec_yup_zup(collider["pos"])
    scale = collider["scale"]
    # Rotation y-up to z-up
    quat_data = collider["rot"]
    rotation = mathutils.Quaternion(quat_xyzw_to_wxyz(quat_data))
    q_conv = mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(90.0))
    rotation = q_conv @ rotation

    # Update or create object
    existing_collider = bpy.data.objects.get(name)
    if existing_collider is None:
        collider = create_collider(name, geom_type, scale)
    else:
        collider = existing_collider

    if collider is not None:
        collider.location = pos
        collider.rotation_mode = 'QUATERNION'
        collider.rotation_quaternion = rotation
        if geom_type == 2:  # unit sphere scaled to the radius (scale[0])
            collider.scale = (scale[0], scale[0], scale[0])

        collider.keyframe_insert(data_path="location", frame=frame_num)
        collider.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        collider.keyframe_insert(data_path="scale", frame=frame_num)


def clear_collider_cache():
    global collider_material
    collider_material = None
