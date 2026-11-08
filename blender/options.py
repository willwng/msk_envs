import bpy


def setup_renderer(fps: float, engine: str = "CYCLES", samples: int = 128, resolution=(1920, 1080),
                   motion_blur: bool = False):
    scene = bpy.context.scene
    # Blender wants an integer rate: use fps_base for fractional ones (e.g. 59.99 from recorded times)
    scene.render.fps = max(1, int(round(fps)))
    scene.render.fps_base = scene.render.fps / fps if fps > 0 else 1.0
    if abs(scene.render.fps_base - 1.0) < 1e-3:
        scene.render.fps_base = 1.0
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.use_motion_blur = motion_blur
    scene.render.motion_blur_shutter = 0.3
    if hasattr(scene.render, "motion_blur_position"):
        scene.render.motion_blur_position = 'CENTER'
    elif hasattr(scene.cycles, "motion_blur_position"):
        scene.cycles.motion_blur_position = 'CENTER'

    engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    if engine == "EEVEE":
        engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.preview_samples = 32
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 6
        scene.cycles.transparent_max_bounces = 16
    if hasattr(scene, "view_settings"):
        looks = {i.identifier for i in scene.view_settings.bl_rna.properties["view_transform"].enum_items}
        if "AgX" in looks:
            scene.view_settings.view_transform = "AgX"
