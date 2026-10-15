import math
import os

import bpy
import mathutils

from util import vec_yup_zup, quat_xyzw_to_wxyz
from material import create_bone_material

if not hasattr(bpy.ops.wm, 'obj_import') and not hasattr(bpy.ops.import_scene, 'obj'):
    bpy.ops.preferences.addon_enable(module="io_scene_obj")

loaded_objs = {}
""" Imported objects, by (obj path, mesh id) """
bone_material = None
""" Shared by all visuals, created on first use """


def get_bone_material():
    global bone_material
    if bone_material is None:
        bone_material = create_bone_material()
    return bone_material


def import_obj(obj_path, identifier):
    # .vtp meshes are imported from the .obj next to them
    root, ext = os.path.splitext(obj_path)
    if ext == ".vtp":
        obj_path = root + ".obj"

    # Cached
    key = (obj_path, identifier)
    if key in loaded_objs:
        return loaded_objs[key]

    bpy.ops.wm.obj_import(filepath=obj_path)
    imported = bpy.context.selected_objects

    # Apply smooth shading to all imported objects
    for obj in imported:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.faces_shade_smooth()
            bpy.ops.object.mode_set(mode='OBJECT')
            obj.select_set(False)

    loaded_objs[key] = imported
    return imported


def update_visual(visual, frame_num, geometry_path):
    # Fetch object data
    identifier = visual["mesh_file"]
    obj_path = visual["mesh_file"]
    location = vec_yup_zup(visual["pos"])
    scale = visual["scale"]
    # Rotation y-up to z-up
    quat_data = visual["rot"]
    rotation = mathutils.Quaternion(quat_xyzw_to_wxyz(quat_data))
    q_conv = mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(90.0))
    rotation = q_conv @ rotation

    # Retrieve OBJ
    obj_path = os.path.join(geometry_path, os.path.basename(obj_path))
    imported_objects = import_obj(obj_path, identifier)

    # Handle multiple objects from a single OBJ file
    for obj in imported_objects:
        if obj.type != 'MESH':
            continue

        # Ensure the object is visible
        obj.rotation_mode = 'QUATERNION'
        obj.hide_render = False
        obj.hide_viewport = False

        # Set transformation
        obj.location = location
        obj.rotation_quaternion = rotation
        obj.scale = scale

        # Apply bone material
        if obj.data.materials:
            obj.data.materials[0] = get_bone_material()
        else:
            obj.data.materials.append(get_bone_material())

        # Insert keyframes
        obj.keyframe_insert(data_path="location", frame=frame_num)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        obj.keyframe_insert(data_path="scale", frame=frame_num)


def clear_visual_cache():
    global loaded_objs, bone_material
    loaded_objs = {}
    bone_material = None
