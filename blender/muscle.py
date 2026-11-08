"""
Muscles: one object for all of them, animated by a point cache instead of keyframes.

The mesh has a vertex for each tube cross-section of each muscle (muscle_style.num_rings), joined by edges, followed by
as many "data" vertices whose positions carry (radius, value, fiber weight) of the matching cross-section. A Mesh
Cache modifier plays the recorded positions (a .pc2 file next to the .blend), then a Geometry Nodes modifier copies the
data onto the path vertices, deletes the data vertices and turns the paths into tubes. The shape follows
msk_envs/muscle_style.py, like the dashboard: "tubes" (thick bellies, thin tendons) or "lines" (thin tubes of constant
radius, tendons in the tendon color) or "capsules" (a capsule per segment between path points: a tube through them with
a sphere at each joint, as the dashboard drew muscles before). With tendons=False, the whole muscle-tendon path is
drawn as fibers (one belly along all of it, for tubes) in the value color.
"""
import os
from dataclasses import dataclass
from typing import Callable

import bpy
import numpy as np

from msk_envs import muscle_style
from util import add_object, write_pc2


def _tubes(path: np.ndarray, n: int, frac: float, belly: float):
    c, length = muscle_style.resample_polyline(path, n)
    if length <= 1e-6:
        return c, np.zeros(n), np.zeros(n)
    radius = muscle_style.radius_profile(n, belly, frac)
    return c, radius, muscle_style.fiber_weight(radius, belly)


def _lines(path: np.ndarray, n: int, frac: float, belly: float):
    c, length = muscle_style.resample_polyline(path, n)
    if length <= 1e-6:
        return c, np.zeros(n), np.zeros(n)
    return c, np.full(n, muscle_style.LINE_RADIUS), muscle_style.line_fiber_weight(n, frac)


def _capsules(path: np.ndarray, n: int, frac: float, belly: float):
    c, fiber, scale = muscle_style.split_at_tendons(path, frac)
    return c, muscle_style.CAPSULE_RADIUS * scale, fiber


@dataclass(frozen=True)
class Style:
    num_points: Callable[[int], int]
    """ Cross-sections of a muscle whose path has this many points """
    sample: Callable
    """ (path [k, 3], cross-sections n, tendon fraction, belly radius) -> centers [n, 3], radius [n], fiber [n] """
    sides: int
    """ Of the tube """
    joints: bool = False
    """ A sphere at each cross-section """


STYLES = {
    "tubes": Style(muscle_style.num_rings, _tubes, sides=12),
    # Through the path points, and the tendon ends (twice each)
    "capsules": Style(lambda k: k + 4, _capsules, sides=12, joints=True),
    "lines": Style(muscle_style.num_rings, _lines, sides=6),
}


def sample_tubes(anim: dict, color_by: str = "activation", only=None, style: str = "tubes", tendons: bool = True):
    """
    Cross-section centers [n_frames, n, 3] and data (radius at thickness 1, value, fiber weight) [n_frames, n, 3]
    of every muscle (or those whose index is in [only]), and the (start, count) of each muscle's cross-sections
    """
    shape = STYLES[style]
    muscles = anim["muscles"]
    adr = np.cumsum([0] + [m["num_points"] for m in muscles])
    layout, start = [], 0
    for i, m in enumerate(muscles):
        if only is None or i in only:
            layout.append((i, start, shape.num_points(m["num_points"])))
            start += layout[-1][2]
    frames = anim["frames"]
    centers = np.zeros((len(frames), start, 3), dtype=np.float32)
    data = np.zeros((len(frames), start, 3), dtype=np.float32)
    for f, frame in enumerate(frames):
        points = np.asarray(frame["muscle_points"], dtype=np.float64).reshape(-1, 3)
        mdata = frame["muscles"]
        values = np.clip(muscle_style.muscle_values(muscles, mdata, color_by), 0.0, 1.0)
        for i, s, n in layout:
            # Without tendons, the whole path is fibers
            frac = muscle_style.tendon_fraction(mdata["path_length"][i], mdata["tendon_length"][i]) if tendons else 0.0
            belly = muscle_style.belly_radius(muscles[i]["max_isometric_force"], 1.0, mdata["fiber_length"][i],
                                              muscles[i].get("optimal_fiber_length"))
            c, radius, fiber = shape.sample(points[adr[i]:adr[i + 1]], n, frac, belly)
            centers[f, s:s + n] = c
            data[f, s:s + n, 0] = radius
            data[f, s:s + n, 1] = values[i]
            data[f, s:s + n, 2] = fiber if tendons else np.where(radius > 0, 1.0, fiber)
    return centers, data, [(s, n) for _, s, n in layout]


def _socket(tree, name, in_out, socket_type, default=None):
    if hasattr(tree, "interface"):  # Blender 4.0+
        socket = tree.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)
    else:
        sockets = tree.inputs if in_out == "INPUT" else tree.outputs
        socket = sockets.new(socket_type, name)
    if default is not None:
        socket.default_value = default
    return socket


def create_tube_nodes(num_path_points: int, material, sides: int = 12, joints: bool = False):
    """
    The Geometry Nodes group described in the module docstring, and its Thickness input; [joints] adds a sphere at
    each path point
    """
    tree = bpy.data.node_groups.new("MuscleTubes", "GeometryNodeTree")
    _socket(tree, "Geometry", "INPUT", "NodeSocketGeometry")
    thickness = _socket(tree, "Thickness", "INPUT", "NodeSocketFloat", 1.0)
    _socket(tree, "Geometry", "OUTPUT", "NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links

    def node(kind, x, y, **props):
        n = nodes.new(kind)
        n.location = (x, y)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    group_in = node("NodeGroupInput", -1400, 0)
    group_out = node("NodeGroupOutput", 1400, 0)
    index = node("GeometryNodeInputIndex", -1400, -250)
    position = node("GeometryNodeInputPosition", -1400, -150)
    count = node("FunctionNodeInputInt", -1400, -350)
    count.integer = num_path_points
    offset = node("FunctionNodeIntegerMath", -1200, -300, operation="ADD") \
        if hasattr(bpy.types, "FunctionNodeIntegerMath") else node("ShaderNodeMath", -1200, -300, operation="ADD")
    links.new(index.outputs[0], offset.inputs[0])
    links.new(count.outputs[0], offset.inputs[1])

    sample = node("GeometryNodeSampleIndex", -1000, 0, data_type="FLOAT_VECTOR", domain="POINT")
    links.new(group_in.outputs["Geometry"], sample.inputs["Geometry"])
    links.new(position.outputs[0], sample.inputs["Value"])
    links.new(offset.outputs[0], sample.inputs["Index"])
    xyz = node("ShaderNodeSeparateXYZ", -800, -100)
    links.new(sample.outputs[0], xyz.inputs[0])

    geometry = group_in.outputs["Geometry"]
    for k, name in enumerate(["radius", "value", "fiber"]):
        store = node("GeometryNodeStoreNamedAttribute", -600 + 180 * k, 0, data_type="FLOAT", domain="POINT")
        store.inputs["Name"].default_value = name
        links.new(geometry, store.inputs["Geometry"])
        links.new(xyz.outputs[k], store.inputs["Value"])
        geometry = store.outputs["Geometry"]

    # Drop the data vertices
    is_data = node("FunctionNodeCompare", -100, -250, data_type="INT", operation="GREATER_EQUAL")
    links.new(index.outputs[0], is_data.inputs[2])
    links.new(count.outputs[0], is_data.inputs[3])
    delete = node("GeometryNodeDeleteGeometry", 50, 0, domain="POINT")
    links.new(geometry, delete.inputs["Geometry"])
    links.new(is_data.outputs[0], delete.inputs["Selection"])

    to_curve = node("GeometryNodeMeshToCurve", 250, 0)
    links.new(delete.outputs[0], to_curve.inputs["Mesh"])
    radius = node("GeometryNodeInputNamedAttribute", 250, -250, data_type="FLOAT")
    radius.inputs["Name"].default_value = "radius"
    scaled = node("ShaderNodeMath", 450, -250, operation="MULTIPLY")
    links.new(radius.outputs["Attribute"], scaled.inputs[0])
    links.new(group_in.outputs["Thickness"], scaled.inputs[1])
    set_radius = node("GeometryNodeSetCurveRadius", 450, 0)
    links.new(to_curve.outputs[0], set_radius.inputs["Curve"])
    links.new(scaled.outputs[0], set_radius.inputs["Radius"])

    profile = node("GeometryNodeCurvePrimitiveCircle", 450, -450)
    profile.inputs["Resolution"].default_value = sides
    profile.inputs["Radius"].default_value = 1.0
    to_mesh = node("GeometryNodeCurveToMesh", 700, 0)
    links.new(set_radius.outputs[0], to_mesh.inputs["Curve"])
    links.new(profile.outputs["Curve"], to_mesh.inputs["Profile Curve"])
    if "Fill Caps" in to_mesh.inputs:
        to_mesh.inputs["Fill Caps"].default_value = True
    if "Scale" in to_mesh.inputs:  # Blender 4.2+: scale profiles by the curve radius explicitly
        radius_in = node("GeometryNodeInputRadius", 450, -600)
        links.new(radius_in.outputs[0], to_mesh.inputs["Scale"])

    tube = to_mesh.outputs[0]
    if joints:  # a sphere of the point's radius on each path point (they carry its attributes)
        sphere = node("GeometryNodeMeshUVSphere", 450, -800)
        sphere.inputs["Segments"].default_value = sides
        sphere.inputs["Rings"].default_value = max(4, sides // 2)
        sphere.inputs["Radius"].default_value = 1.0
        instance = node("GeometryNodeInstanceOnPoints", 700, -400)
        links.new(delete.outputs[0], instance.inputs["Points"])
        links.new(sphere.outputs["Mesh"], instance.inputs["Instance"])
        links.new(scaled.outputs[0], instance.inputs["Scale"])
        realize = node("GeometryNodeRealizeInstances", 850, -400)
        links.new(instance.outputs[0], realize.inputs[0])
        join = node("GeometryNodeJoinGeometry", 900, -150)
        links.new(realize.outputs[0], join.inputs[0])
        links.new(tube, join.inputs[0])
        tube = join.outputs[0]
    smooth = node("GeometryNodeSetShadeSmooth", 1000, 0)
    links.new(tube, smooth.inputs["Geometry"])
    set_material = node("GeometryNodeSetMaterial", 1200, 0)
    set_material.inputs["Material"].default_value = material
    links.new(smooth.outputs[0], set_material.inputs["Geometry"])
    links.new(set_material.outputs[0], group_out.inputs["Geometry"])
    return tree, thickness


def build_muscles(anim: dict, root, col, material, cache_dir: str, color_by: str = "activation",
                  thickness: float = 1.0, only=None, name: str = "Muscles", style: str = "capsules",
                  tendons: bool = True):
    """ The muscles object (see the module docstring); its point cache is written to [cache_dir] """
    if not anim["muscles"] or not anim["frames"]:
        return None
    centers, data, layout = sample_tubes(anim, color_by, only, style, tendons)
    if centers.shape[1] == 0:
        return None
    n = centers.shape[1]
    points = np.concatenate([centers, data], axis=1)  # [n_frames, 2n, 3]

    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{name}.pc2")
    write_pc2(cache, points, start_frame=1.0)

    mesh = bpy.data.meshes.new(name)
    edges = [(s + k, s + k + 1) for s, count in layout for k in range(count - 1)]
    mesh.from_pydata(points[0].tolist(), edges, [])
    obj = add_object(name, mesh, col, root, material)

    mesh_cache = obj.modifiers.new("Recording", "MESH_CACHE")
    mesh_cache.cache_format = "PC2"
    mesh_cache.filepath = bpy.path.relpath(cache) if bpy.data.filepath else cache
    mesh_cache.time_mode = "FRAME"
    mesh_cache.play_mode = "SCENE"
    mesh_cache.frame_start = 1.0  # cache sample 0 is Blender frame 1
    mesh_cache.interpolation = "LINEAR"
    mesh_cache.forward_axis = "POS_Y"
    mesh_cache.up_axis = "POS_Z"

    tubes = obj.modifiers.new("Tubes", "NODES")
    shape = STYLES[style]
    tubes.node_group, thickness_socket = create_tube_nodes(n, material, shape.sides, shape.joints)
    tubes[thickness_socket.identifier] = float(thickness)
    obj["cache_file"] = cache
    return obj
