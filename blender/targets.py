""" Per-frame overlays: task targets and command (guide) arrows """
import bpy
import numpy as np
from mathutils import Matrix

from material import create_color_material
from util import animate, animate_transforms, move_to, set_smooth


Z_TO_Y = Matrix(((1, 0, 0, 0), (0, 0, 1, 0.5), (0, -1, 0, 0), (0, 0, 0, 1)))
""" Blender's z primitives (centered) onto y, from 0 to 1 """


def _hex_rgb(color: int):
    return ((color >> 16) & 255) / 255.0, ((color >> 8) & 255) / 255.0, (color & 255) / 255.0


def build_targets(anim: dict, root, col, frames: np.ndarray):
    n = max((len(f.get("targets", [])) for f in anim["frames"]), default=0)
    if not n:
        return []
    active = create_color_material("TargetActive", (0.42, 0.92, 0.62), alpha=0.85)
    idle = create_color_material("Target", (0.53, 0.81, 0.92), alpha=0.85)
    objs = []
    for k in range(n):
        X = np.zeros((len(frames), 7))
        X[:, 6] = 1.0
        radius = np.zeros(len(frames))
        is_active = np.zeros(len(frames))
        for i, f in enumerate(anim["frames"]):
            if k < len(f.get("targets", [])):
                t = f["targets"][k]
                X[i] = list(t["pos"]) + list(t["rot"])
                radius[i] = t["radius"]
                is_active[i] = float(t["active"])
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
        obj = bpy.context.active_object
        obj.name = f"target_{k}"
        set_smooth(obj.data)
        move_to(obj, col, root)
        # Active and idle versions, switched by visibility
        obj.data.materials.append(idle)
        twin = obj.copy()
        twin.data = obj.data.copy()
        twin.data.materials[0] = active
        twin.name = f"target_{k}_active"
        col.objects.link(twin)
        for o, show in [(obj, 1 - is_active), (twin, is_active)]:
            animate_transforms(o, frames, X)
            animate(o, "scale", frames, np.repeat((radius * show)[:, None], 3, axis=1), interpolation="CONSTANT")
            objs.append(o)
    return objs


def build_guides(anim: dict, root, col, frames: np.ndarray, shaft_radius=0.02, head_length=0.12, head_radius=0.05):
    """ Command arrows (MSKEnv.guide_arrows): a shaft and a head per arrow slot, keyframed every frame """
    n = max((len(f.get("guides", [])) for f in anim["frames"]), default=0)
    objs = []
    up = np.array([0.0, 1.0, 0.0])
    for k in range(n):
        color = next((f["guides"][k]["color"] for f in anim["frames"] if k < len(f.get("guides", []))), 0xffffff)
        material = create_color_material(f"Guide_{k}", _hex_rgb(color), roughness=0.4)
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=1.0, vertices=16)
        shaft = bpy.context.active_object
        shaft.data.transform(Z_TO_Y)
        bpy.ops.mesh.primitive_cone_add(radius1=1.0, radius2=0.0, depth=1.0, vertices=16)
        head = bpy.context.active_object
        head.data.transform(Z_TO_Y)
        X_shaft, X_head = np.zeros((len(frames), 7)), np.zeros((len(frames), 7))
        S_shaft, S_head = np.zeros((len(frames), 3)), np.zeros((len(frames), 3))
        for i, f in enumerate(anim["frames"]):
            if k >= len(f.get("guides", [])):
                X_shaft[i, 6] = X_head[i, 6] = 1.0
                continue
            g = f["guides"][k]
            start, d = np.array(g["start"]), np.array(g["direction"])
            length = np.linalg.norm(d)
            u = d / length if length > 1e-9 else up
            # Quaternion rotating +y onto u
            axis, dot = np.cross(up, u), float(np.dot(up, u))
            q = np.array([*axis, 1.0 + dot]) if dot > -0.999999 else np.array([1.0, 0.0, 0.0, 0.0])
            q /= np.linalg.norm(q)
            shaft_len = max(0.0, length - head_length)
            X_shaft[i] = [*start, *q]
            X_head[i] = [*(start + u * shaft_len), *q]
            S_shaft[i] = (shaft_radius, shaft_len, shaft_radius) if shaft_len > 0 else 0.0
            S_head[i] = (head_radius, min(length, head_length), head_radius) if length > 1e-9 else 0.0
        for obj, X, S, name in [(shaft, X_shaft, S_shaft, "shaft"), (head, X_head, S_head, "head")]:
            obj.name = f"guide_{k}_{name}"
            obj.data.materials.append(material)
            move_to(obj, col, root)
            animate_transforms(obj, frames, X)
            animate(obj, "scale", frames, S)
            objs.append(obj)
    return objs
