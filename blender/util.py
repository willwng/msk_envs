""" Helpers: the y-up root, fast keyframing, and point caches """
import math
import struct

import bpy
import numpy as np

from msk_envs.trajectory_io import continuous_quats


def vec_yup_zup(v):
    """ A y-up (simulation) point in Blender's z-up world """
    return [v[0], -v[2], v[1]]


def collection(name: str, parent=None):
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def add_object(name: str, data, col, root, material=None):
    """ A new object of [data] (a mesh, or None for an empty) in [col], parented to [root] """
    obj = bpy.data.objects.new(name, data)
    col.objects.link(obj)
    obj.parent = root
    if material is not None:
        obj.data.materials.append(material)
    return obj


def move_to(obj, col, root):
    """ Move [obj] (e.g. made by a bpy.ops operator) into [col] only, parented to [root] """
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    obj.parent = root
    return obj


def create_root(name: str = "MSK") -> bpy.types.Object:
    """
    An empty that turns the simulation's y-up frame into Blender's z-up one. Everything recorded is parented to it,
    so objects keep the recorded (y-up) positions and rotations
    """
    root = bpy.data.objects.new(name, None)
    root.empty_display_type = "ARROWS"
    root.empty_display_size = 0.2
    root.rotation_euler = (math.pi / 2, 0.0, 0.0)
    bpy.context.scene.collection.objects.link(root)
    return root


def _fcurve(action, obj, data_path: str, index: int):
    if hasattr(action, "fcurve_ensure_for_datablock"):  # Blender 4.4+ (slotted actions)
        return action.fcurve_ensure_for_datablock(obj, data_path, index=index)
    fc = action.fcurves.find(data_path, index=index)
    return fc or action.fcurves.new(data_path, index=index)


INTERPOLATION = {"CONSTANT": 0, "LINEAR": 1, "BEZIER": 2}


def animate(obj, data_path: str, frames: np.ndarray, values: np.ndarray, interpolation: str = "LINEAR"):
    """
    Keyframe obj.<data_path> (values [n_frames, n_components], or [n_frames] for a scalar) at [frames] in one go,
    much faster than keyframe_insert
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    frames = np.asarray(frames, dtype=np.float64)
    ad = obj.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(f"{obj.name}_action")
    for i in range(values.shape[1]):
        fc = _fcurve(ad.action, obj, data_path, i)
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(frames))
        co = np.empty(2 * len(frames))
        co[0::2], co[1::2] = frames, values[:, i]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [INTERPOLATION[interpolation]] * len(frames))
        fc.update()


def animate_transforms(obj, frames: np.ndarray, X: np.ndarray, scale=None):
    """ Keyframe location and rotation from [n_frames, 7] (pos, xyzw quaternion), in the parent's frame """
    obj.rotation_mode = "QUATERNION"
    if scale is not None:
        obj.scale = scale
    quats = continuous_quats(np.array(X[:, [6, 3, 4, 5]], dtype=np.float64))  # wxyz
    animate(obj, "location", frames, X[:, :3])
    animate(obj, "rotation_quaternion", frames, quats)


def write_pc2(path: str, points: np.ndarray, start_frame: float = 1.0, sample_rate: float = 1.0):
    """ A point cache for the Mesh Cache modifier: points [n_frames, n_points, 3] """
    points = np.ascontiguousarray(points, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<12siiffi", b"POINTCACHE2\0", 1, points.shape[1], start_frame, sample_rate,
                            points.shape[0]))
        f.write(points.tobytes())


def set_smooth(mesh):
    if hasattr(mesh, "shade_smooth"):
        mesh.shade_smooth()
    else:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
