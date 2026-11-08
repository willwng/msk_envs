""" Materials (Cycles and EEVEE) """
import bpy

from msk_envs import muscle_style


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear(rgb, alpha=1.0):
    return tuple(srgb_to_linear(c) for c in rgb) + (alpha,)


def _new(name: str):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, bsdf


def _set(bsdf, **inputs):
    """ Set Principled inputs that exist in this Blender version (names changed in 4.0) """
    aliases = {"Subsurface Weight": "Subsurface", "Specular IOR Level": "Specular", "Sheen Weight": "Sheen",
               "Coat Weight": "Clearcoat"}
    for name, value in inputs.items():
        name = name.replace("_", " ")
        key = name if name in bsdf.inputs else aliases.get(name)
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = value


def _transparent(mat, bsdf, alpha: float):
    _set(bsdf, Alpha=alpha)
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    elif hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    if hasattr(mat, "use_transparency_overlap"):
        mat.use_transparency_overlap = False


def create_bone_material():
    mat, bsdf = _new("Bone")
    _set(bsdf, Base_Color=_linear((0.90, 0.86, 0.78)), Roughness=0.6, Subsurface_Weight=0.08,
         Specular_IOR_Level=0.3)
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.004, 0.002, 0.001)
    return mat


def create_muscle_material():
    """ Colored by the "value" attribute (e.g. activation) through muscle_style's colormap, tendons by "fiber" """
    mat, bsdf = _new("Muscle")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links

    value = nodes.new("ShaderNodeAttribute")
    value.attribute_type = "GEOMETRY"
    value.attribute_name = "value"
    value.location = (-700, 100)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-500, 100)
    elements = ramp.color_ramp.elements
    stops = muscle_style.COLOR_STOPS
    while len(elements) < len(stops):
        elements.new(0.5)
    for element, (pos, rgb) in zip(elements, stops):
        element.position = pos
        element.color = _linear(rgb)
    links.new(value.outputs["Fac"], ramp.inputs["Fac"])

    fiber = nodes.new("ShaderNodeAttribute")
    fiber.attribute_type = "GEOMETRY"
    fiber.attribute_name = "fiber"
    fiber.location = (-500, -150)
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.location = (-200, 0)
    mix.inputs["A"].default_value = _linear(muscle_style.TENDON_COLOR)
    links.new(fiber.outputs["Fac"], mix.inputs["Factor"])
    links.new(ramp.outputs["Color"], mix.inputs["B"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    _set(bsdf, Roughness=0.42, Subsurface_Weight=0.15, Sheen_Weight=0.15, Specular_IOR_Level=0.45)
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.003, 0.001, 0.0008)
    return mat


def create_collider_material():
    """ Light blue, green while the "contact" object property is 1 """
    mat, bsdf = _new("Collider")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    contact = nodes.new("ShaderNodeAttribute")
    contact.attribute_type = "OBJECT"
    contact.attribute_name = "contact"
    contact.location = (-500, 0)
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.location = (-200, 0)
    mix.inputs["A"].default_value = _linear((0.53, 0.81, 0.92))
    mix.inputs["B"].default_value = _linear((0.27, 0.88, 0.54))
    links.new(contact.outputs["Fac"], mix.inputs["Factor"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    _set(bsdf, Roughness=0.5)
    _transparent(mat, bsdf, 0.5)
    return mat


def create_wrap_material():
    mat, bsdf = _new("WrapSurface")
    _set(bsdf, Base_Color=_linear((0.31, 0.69, 1.0)), Roughness=0.25)
    _transparent(mat, bsdf, 0.25)
    return mat


def create_color_material(name: str, rgb, alpha: float = 1.0, roughness: float = 0.6):
    mat, bsdf = _new(name)
    _set(bsdf, Base_Color=_linear(rgb), Roughness=roughness)
    if alpha < 1.0:
        _transparent(mat, bsdf, alpha)
    return mat


def create_ground_material():
    """ A dark slate checker grid (1 m squares), as the dashboard's ground, so the pale bones stand out """
    mat, bsdf = _new("Ground")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    coords.location = (-900, 0)
    checker = nodes.new("ShaderNodeTexChecker")
    checker.location = (-650, 100)
    checker.inputs["Scale"].default_value = 1.0
    checker.inputs["Color1"].default_value = _linear((0.34, 0.38, 0.42))
    checker.inputs["Color2"].default_value = _linear((0.32, 0.35, 0.39))
    links.new(coords.outputs["Object"], checker.inputs["Vector"])
    brick = nodes.new("ShaderNodeTexBrick")  # thin lines every 10 cm
    brick.location = (-650, -200)
    brick.offset = 0.0
    brick.squash = 1.0
    brick.inputs["Scale"].default_value = 10.0
    brick.inputs["Mortar Size"].default_value = 0.01
    brick.inputs["Brick Width"].default_value = 1.0
    brick.inputs["Row Height"].default_value = 1.0
    brick.inputs["Color1"].default_value = (1, 1, 1, 1)
    brick.inputs["Color2"].default_value = (1, 1, 1, 1)
    brick.inputs["Mortar"].default_value = (1.25, 1.25, 1.25, 1)  # lines a little lighter than the squares
    links.new(coords.outputs["Object"], brick.inputs["Vector"])
    mult = nodes.new("ShaderNodeMix")
    mult.data_type = "RGBA"
    mult.blend_type = "MULTIPLY"
    mult.location = (-350, 0)
    mult.inputs["Factor"].default_value = 1.0
    links.new(checker.outputs["Color"], mult.inputs["A"])
    links.new(brick.outputs["Color"], mult.inputs["B"])
    links.new(mult.outputs["Result"], bsdf.inputs["Base Color"])
    _set(bsdf, Roughness=0.9)
    return mat
