""" Contact geometry (bolt.GeomType) and muscle wrapping surfaces (bolt.WrapType), keyframed from the recording """
import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from util import add_object, animate, animate_transforms, set_smooth


def _mesh(name: str, build) -> bpy.types.Mesh:
    bm = bmesh.new()
    build(bm)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    set_smooth(mesh)
    return mesh


def _sphere(bm, radius=1.0, matrix=Matrix()):
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=radius, matrix=matrix)


def _capsule(bm, radius, half_height):
    """ Along z, like Bolt's capsules """
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=radius)
    for v in bm.verts:  # stretch the sphere's two halves apart
        v.co.z += half_height if v.co.z > 1e-9 else (-half_height if v.co.z < -1e-9 else 0.0)


def _cylinder(bm, radius, half_height, caps=True):
    bmesh.ops.create_cone(bm, cap_ends=caps, segments=48, radius1=radius, radius2=radius, depth=2 * half_height)


def _torus(bm, major, minor, rings=48, sides=16):
    verts = []
    for i in range(rings):
        u = 2 * np.pi * i / rings
        row = []
        for j in range(sides):
            v = 2 * np.pi * j / sides
            r = major + minor * np.cos(v)
            row.append(bm.verts.new((r * np.cos(u), r * np.sin(u), minor * np.sin(v))))
        verts.append(row)
    for i in range(rings):
        for j in range(sides):
            bm.faces.new((verts[i][j], verts[(i + 1) % rings][j], verts[(i + 1) % rings][(j + 1) % sides],
                          verts[i][(j + 1) % sides]))


def collider_mesh(name: str, geom_type: int, size, caps: bool = True):
    """ caps: whether cylinders are closed """
    s = [float(x) for x in size]
    if geom_type == 2:
        return _mesh(name, lambda bm: _sphere(bm, s[0]))
    if geom_type == 3:
        return _mesh(name, lambda bm: _capsule(bm, s[0], s[1]))
    if geom_type == 4:
        return _mesh(name, lambda bm: _sphere(bm, 1.0, Matrix.Diagonal((s[0], s[1], s[2], 1.0))))
    if geom_type == 5:
        return _mesh(name, lambda bm: _cylinder(bm, s[0], s[1], caps))
    if geom_type == 6:
        mesh = _mesh(name, lambda bm: bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal(
            (2 * s[0], 2 * s[1], 2 * s[2], 1.0))))
        mesh.polygons.foreach_set("use_smooth", [False] * len(mesh.polygons))
        return mesh
    return None


WRAP_TO_GEOM_TYPE = {0: 2, 1: 5, 2: 4}
""" Wrapping surfaces (bolt.WrapType) drawn as the collider of this bolt.GeomType: sphere, cylinder, ellipsoid """


def wrap_mesh(name: str, surface: dict):
    s = surface["size"]
    if surface["wrap_type"] == 3:
        return _mesh(name, lambda bm: _torus(bm, float(s[0]), float(s[1])))
    if surface["wrap_type"] == 1:  # infinite: drawn half_length (recorded) along z, open
        return collider_mesh(name, 5, (s[0], surface["half_length"]), caps=False)
    geom_type = WRAP_TO_GEOM_TYPE.get(surface["wrap_type"])
    return collider_mesh(name, geom_type, s) if geom_type is not None else None


def build_colliders(anim: dict, root, col, material, frames: np.ndarray, which: str = "shoes"):
    """ which: "all", "shoes" (colliders named left_*/right_*, and hurdles) or "none". Planes aren't drawn """
    if which == "none" or not anim["colliders"]:
        return []
    rows = np.array([f["colliders"] for f in anim["frames"]], dtype=np.float64)  # [n_frames, n, 8]
    objs = []
    for k, c in enumerate(anim["colliders"]):
        name = c["name"]
        if which == "shoes" and not (name.startswith(("left_", "right_")) or "hurdle" in name):
            continue
        mesh = collider_mesh(name, c["geom_type"], c["size"])
        if mesh is None:
            continue
        obj = add_object(name, mesh, col, root, material)
        animate_transforms(obj, frames, rows[:, k, :7])
        obj["contact"] = 0.0
        animate(obj, '["contact"]', frames, (rows[:, k, 7] > 0).astype(float), interpolation="CONSTANT")
        objs.append(obj)
    return objs


def build_wrap_surfaces(anim: dict, root, col, material, frames: np.ndarray):
    surfaces = anim.get("wrap_surfaces", [])
    if not surfaces:
        return []
    X = np.array([f["wrap_surfaces"] for f in anim["frames"]], dtype=np.float64)
    objs = []
    for k, surface in enumerate(surfaces):
        mesh = wrap_mesh(surface["name"], surface)
        if mesh is None:
            continue
        obj = add_object(surface["name"], mesh, col, root, material)
        animate_transforms(obj, frames, X[:, k])
        objs.append(obj)
    return objs


def build_ground(col, material):
    """ A plane at the recorded ground (z-up world), for scenes without the track """
    mesh = bpy.data.meshes.new("Ground")
    mesh.from_pydata([(-150, -30, 0), (250, -30, 0), (250, 30, 0), (-150, 30, 0)], [], [(0, 1, 2, 3)])
    mesh.materials.append(material)
    ground = bpy.data.objects.new("Ground", mesh)  # object coordinates are meters, for the checker texture
    col.objects.link(ground)
    return ground
