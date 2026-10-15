import bpy


def create_bone_material():
    # Create a new material
    bone_material = bpy.data.materials.new(name="BoneMaterial")
    bone_material.use_nodes = True
    nodes = bone_material.node_tree.nodes
    links = bone_material.node_tree.links
    # Clear default nodes
    nodes.clear()
    # Add Principled BSDF node
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value = (0.85, 0.78, 0.68, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0

    # Handle different Blender versions for specular input - reduce reflectivity
    if 'Specular IOR' in bsdf.inputs:
        bsdf.inputs['Specular IOR'].default_value = 1.1  # Even lower IOR for minimal reflection
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.05  # Very low specular for matte look
    # Handle subsurface scattering (may vary by version)
    if 'Subsurface Weight' in bsdf.inputs:
        bsdf.inputs['Subsurface Weight'].default_value = 0.15
        if 'Subsurface Color' in bsdf.inputs:
            bsdf.inputs['Subsurface Color'].default_value = (0.8, 0.7, 0.55, 1.0)  # Warmer subsurface
    elif 'Subsurface' in bsdf.inputs:
        bsdf.inputs['Subsurface'].default_value = 0.15
        if 'Subsurface Color' in bsdf.inputs:
            bsdf.inputs['Subsurface Color'].default_value = (0.8, 0.7, 0.55, 1.0)  # Warmer subsurface
    # Add Material Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    # Link BSDF to Material Output
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return bone_material


def create_muscle_material():
    muscle_material = bpy.data.materials.new(name="MuscleMaterial")
    muscle_material.use_nodes = True
    nodes = muscle_material.node_tree.nodes
    links = muscle_material.node_tree.links

    nodes.clear()

    # Use a Value node (will be driven by custom property)
    value_node = nodes.new(type='ShaderNodeValue')
    value_node.location = (-600, 0)

    # Add ColorRamp node for smooth transition
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = (-400, 0)

    # Interpolate between blue and red
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.0, 0.0, 1.0, 1.0)
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (1.0, 0.0, 0.0, 1.0)

    # Add Principled BSDF node
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Metallic'].default_value = 0.3
    if 'Specular IOR' in bsdf.inputs:
        bsdf.inputs['Specular IOR'].default_value = 2.0
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 1.0

    # Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)

    # Connect nodes
    links.new(value_node.outputs['Value'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Save reference to the value_node to attach driver later
    muscle_material["activation_value_node_name"] = value_node.name

    return muscle_material


def create_collider_material():
    """Light blue, semi-transparent collider material"""
    collider_material = bpy.data.materials.new(name="ColliderMaterial")
    collider_material.use_nodes = True

    nodes = collider_material.node_tree.nodes
    links = collider_material.node_tree.links
    nodes.clear()

    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    # Light blue color
    bsdf.inputs['Base Color'].default_value = (0.4, 0.7, 1.0, 1.0)

    # Matte look
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Metallic'].default_value = 0.0

    # Transparency
    if 'Alpha' in bsdf.inputs:
        bsdf.inputs['Alpha'].default_value = 0.3

    # Reduce reflections
    if 'Specular IOR' in bsdf.inputs:
        bsdf.inputs['Specular IOR'].default_value = 1.1
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.05

    # Material Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return collider_material


def create_target_material(active: bool):
    target_material = bpy.data.materials.new(name="TargetMaterial")
    target_material.use_nodes = True
    nodes = target_material.node_tree.nodes
    links = target_material.node_tree.links
    # Clear default nodes
    nodes.clear()
    # Add Principled BSDF node
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    if active:
        bsdf.inputs["Base Color"].default_value = (0.0, 1.0, 0.0, 1.0)
    else:
        bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0

    # Handle different Blender versions for specular input - reduce reflectivity
    if 'Specular IOR' in bsdf.inputs:
        bsdf.inputs['Specular IOR'].default_value = 1.1  # Even lower IOR for minimal reflection
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.05  # Very low specular for matte look
    # Handle subsurface scattering (may vary by version)
    if 'Subsurface Weight' in bsdf.inputs:
        bsdf.inputs['Subsurface Weight'].default_value = 0.15
        if 'Subsurface Color' in bsdf.inputs:
            bsdf.inputs['Subsurface Color'].default_value = (0.8, 0.7, 0.55, 1.0)  # Warmer subsurface
    elif 'Subsurface' in bsdf.inputs:
        bsdf.inputs['Subsurface'].default_value = 0.15
        if 'Subsurface Color' in bsdf.inputs:
            bsdf.inputs['Subsurface Color'].default_value = (0.8, 0.7, 0.55, 1.0)  # Warmer subsurface
    # Add Material Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    # Link BSDF to Material Output
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return target_material
