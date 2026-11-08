""" Bone meshes: one mesh per file (shared by objects that use the same file), keyframed from the recording """
import os

import bpy
import numpy as np

from util import add_object, animate_transforms, set_smooth

_meshes = {}
""" Imported mesh data, by file """


def clear_visual_cache():
    _meshes.clear()


def import_mesh(path: str):
    """ The mesh data of an OBJ file, imported without axis conversion (the root empty converts y-up to z-up) """
    if path in _meshes:
        return _meshes[path]
    root, ext = os.path.splitext(path)
    if ext.lower() == ".vtp":  # Blender can't read .vtp: bolt's geometry directory has an .obj next to each
        path_obj = root + ".obj"
    else:
        path_obj = path
    if not os.path.exists(path_obj):
        print(f"Missing mesh {path_obj}")
        _meshes[path] = None
        return None
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=path_obj, forward_axis="Y", up_axis="Z")
    imported = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not imported:
        _meshes[path] = None
        return None
    if len(imported) > 1:
        with bpy.context.temp_override(active_object=imported[0], selected_editable_objects=imported):
            bpy.ops.object.join()
    obj = imported[0]
    mesh = obj.data
    mesh.name = os.path.basename(root)
    set_smooth(mesh)
    for o in [o for o in bpy.data.objects if o not in before]:
        bpy.data.objects.remove(o)
    _meshes[path] = mesh
    return mesh


def build_visuals(anim: dict, root, col, material, geometry_dir: str, frames: np.ndarray):
    objs = []
    X = np.array([f["visuals"] for f in anim["frames"]], dtype=np.float64)  # [n_frames, n_visuals, 7]
    for k, visual in enumerate(anim["visuals"]):
        mesh = import_mesh(os.path.join(geometry_dir, os.path.basename(visual["mesh_file"])))
        if mesh is None:
            continue
        if not mesh.materials:
            mesh.materials.append(material)
        obj = add_object(f"{mesh.name}_{k}", mesh, col, root)
        animate_transforms(obj, frames, X[:, k], scale=visual.get("scale", (1, 1, 1)))
        objs.append(obj)
    return objs
