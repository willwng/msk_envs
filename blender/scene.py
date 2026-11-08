import bpy


def clear_scene():
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # Clear all collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    # Clear all mesh data
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    # Clear all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    # Clear all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)

    # Clear all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)

    # Clear all cameras
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)

    # Clear all lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    # Clear all curves
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)

    # Clear all armatures
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)

    # Clear all actions
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)

    return


def setup_lighting():
    """ Creates a simple sun light """
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun_light = bpy.context.active_object
    sun_light.name = "SunLight"
    sun_light.data.energy = 5.5
    sun_light.data.angle = 0.0873  # 5 deg
    sun_light.rotation_euler = (0.785, 0, 0.785)
    return


def setup_sky():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    # Clear existing nodes
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    # Create nodes
    sky_tex = nodes.new("ShaderNodeTexSky")
    sky_tex.sun_disc = False
    sky_tex.sun_elevation = 0.785398  # 45 degrees in radians
    sky_tex.sun_rotation = 0.785398  # 45 degrees in radians
    sky_tex.altitude = 0.0

    if 'NISHITA' in sky_tex.bl_rna.properties['sky_type'].enum_items:
        sky_tex.sky_type = 'NISHITA'
        sky_tex.air_density = 0.6
        sky_tex.dust_density = 0.0
        sky_tex.ozone_density = 2.0
    else:
        sky_tex.sky_type = 'SINGLE_SCATTERING'
        sky_tex.air_density = 0.6
        sky_tex.aerosol_density = 0.0
        sky_tex.ozone_density = 2.0

    background = nodes.new("ShaderNodeBackground")
    background.inputs['Strength'].default_value = 0.05
    # Position nodes
    output = nodes.new("ShaderNodeOutputWorld")
    sky_tex.location = (-400, 0)
    background.location = (-200, 0)
    output.location = (0, 0)
    # Link nodes
    links.new(sky_tex.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])


def reset_scene():
    clear_scene()
    setup_lighting()
    setup_sky()
    return
