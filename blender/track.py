import bpy
import bmesh
import math


def setup_track():
    """ Build a standard 400m track (same dimensions as msk_envs/envs/curve.py), with lane markings and a billboard """
    # ==========================================
    # --- IAAF Track Dimensions (in meters) ---
    # ==========================================
    straight_length = 84.39
    inner_radius = 36.5
    lane_width = 1.22
    num_lanes = 8
    line_width = 0.05
    line_z = 0.005
    segments_per_curve = 128
    chute_length = 30.0

    # --- Stadium Background Dimensions ---
    grass_buffer_width = 15.0
    grass_z = 0.0

    r_track_out = inner_radius + (lane_width * num_lanes)
    r_grass_out = r_track_out + grass_buffer_width

    # ==========================================
    # --- ORIGIN SHIFT CALCULATION           ---
    # ==========================================
    finish_x = straight_length / 2
    start_x_original = finish_x - 100.5
    lane_4_y_original = -(inner_radius + (3.5 * lane_width))

    O_X = -start_x_original
    O_Y = -lane_4_y_original

    def shift_verts(verts):
        return [(v[0] + O_X, v[1] + O_Y, v[2]) for v in verts]

    # ==============================================
    # --- Helper Function: Triangulate N-Gons    ---
    # ==============================================
    # Safely triangulates custom shapes to prevent Cycles shading errors
    def clean_ngon(mesh):
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(mesh)
        bm.free()

    # ==============================================
    # --- Helper Function: Generate a track ring ---
    # ==============================================
    def get_track_ring_data(r_in, r_out, z_elev, extend_outer=False):
        verts = []
        faces = []

        def get_semi_circle(center_x, radius, is_right_side, is_outer):
            pts = []
            start_angle = -math.pi / 2 if is_right_side else math.pi / 2
            end_angle = math.pi / 2 if is_right_side else 3 * math.pi / 2
            for i in range(segments_per_curve + 1):
                angle = start_angle + (end_angle - start_angle) * (i / segments_per_curve)
                r_actual = radius

                # Morph the outer radius to organically form the 100m sprint chute
                if extend_outer and is_outer and not is_right_side:
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                    max_t = radius
                    chute_left = -chute_length
                    chute_top = -inner_radius
                    chute_bottom = -r_track_out

                    if dx < -0.00001:
                        t = chute_left / dx
                        if chute_bottom <= t * dy <= chute_top:
                            max_t = max(max_t, t)
                    if dy < -0.00001:
                        t1 = chute_bottom / dy
                        if chute_left <= t1 * dx <= 0.0001:
                            max_t = max(max_t, t1)
                        t2 = chute_top / dy
                        if chute_left <= t2 * dx <= 0.0001:
                            max_t = max(max_t, t2)
                    r_actual = max_t

                pts.append((center_x + r_actual * math.cos(angle), r_actual * math.sin(angle), z_elev))
            return pts[:-1]

        in_r = get_semi_circle(straight_length / 2, r_in, True, False)
        in_l = get_semi_circle(-straight_length / 2, r_in, False, False)
        inner_verts = in_r + in_l

        out_r = get_semi_circle(straight_length / 2, r_out, True, True)
        out_l = get_semi_circle(-straight_length / 2, r_out, False, True)
        outer_verts = out_r + out_l

        verts = inner_verts + outer_verts
        num_inner = len(inner_verts)
        for i in range(num_inner):
            next_i = (i + 1) % num_inner
            faces.append((i, i + num_inner, next_i + num_inner, next_i))
        return verts, faces

    # ============================
    # --- Materials Setup      ---
    # ============================
    def create_materials():
        mat_track = bpy.data.materials.new(name="Stadium_Tartan_Rubber")
        mat_track.use_nodes = True
        nodes = mat_track.node_tree.nodes
        links = mat_track.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = (0.5, 0.08, 0.05, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.85
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs['Scale'].default_value = 350.0
        bump = nodes.new(type="ShaderNodeBump")
        bump.inputs['Strength'].default_value = 0.6
        bump.inputs['Distance'].default_value = 0.005
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

        mat_white = bpy.data.materials.new(name="Stadium_White_Paint")
        mat_white.use_nodes = True
        w_bsdf = mat_white.node_tree.nodes.get("Principled BSDF")
        w_bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
        w_bsdf.inputs['Roughness'].default_value = 0.6

        mat_turf = bpy.data.materials.new(name="Stadium_Infield_Turf")
        mat_turf.use_nodes = True
        t_nodes = mat_turf.node_tree.nodes
        t_links = mat_turf.node_tree.links
        t_bsdf = t_nodes.get("Principled BSDF")
        t_bsdf.inputs['Roughness'].default_value = 0.95

        tex_coord = t_nodes.new(type="ShaderNodeTexCoord")
        mapping = t_nodes.new(type="ShaderNodeMapping")
        wave = t_nodes.new(type="ShaderNodeTexWave")
        wave.wave_type = 'BANDS'
        wave.bands_direction = 'Y'
        wave.inputs['Scale'].default_value = 0.04

        color_ramp = t_nodes.new(type="ShaderNodeValToRGB")
        color_ramp.color_ramp.interpolation = 'EASE'
        color_ramp.color_ramp.elements[0].position = 0.4
        color_ramp.color_ramp.elements[0].color = (0.02, 0.14, 0.015, 1.0)
        color_ramp.color_ramp.elements[1].position = 0.6
        color_ramp.color_ramp.elements[1].color = (0.035, 0.22, 0.025, 1.0)

        t_noise = t_nodes.new(type="ShaderNodeTexNoise")
        t_noise.inputs['Scale'].default_value = 600.0
        t_noise.inputs['Detail'].default_value = 15.0

        t_bump = t_nodes.new(type="ShaderNodeBump")
        t_bump.inputs['Strength'].default_value = 0.6
        t_bump.inputs['Distance'].default_value = 0.01

        t_links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
        t_links.new(mapping.outputs['Vector'], wave.inputs['Vector'])
        t_links.new(wave.outputs['Fac'], color_ramp.inputs['Fac'])
        t_links.new(color_ramp.outputs['Color'], t_bsdf.inputs['Base Color'])
        t_links.new(t_noise.outputs['Fac'], t_bump.inputs['Height'])
        t_links.new(t_bump.outputs['Normal'], t_bsdf.inputs['Normal'])

        return mat_track, mat_white, mat_turf

    mat_track, mat_white, mat_turf = create_materials()

    # =========================================
    # --- Geometry Generation & Placement ---
    # =========================================
    stadium_col = bpy.data.collections.new("0_Stadium_Environment")
    bpy.context.scene.collection.children.link(stadium_col)
    track_col = bpy.data.collections.new("1_Track")
    markers_col = bpy.data.collections.new("2_Markers_100m")
    stadium_col.children.link(track_col)
    stadium_col.children.link(markers_col)

    # 2. Generate Base Track & Chute (Now organically morphed!)
    t_verts, t_faces = get_track_ring_data(inner_radius, r_track_out, 0.0, extend_outer=True)

    track_mesh = bpy.data.meshes.new("Track_Base_Mesh")
    track_mesh.from_pydata(shift_verts(t_verts), [], t_faces)
    clean_ngon(track_mesh)  # Triangulates the morphed quads for perfect Cycles shading
    track_obj = bpy.data.objects.new("1_Track_Base", track_mesh)
    track_col.objects.link(track_obj)
    track_obj.data.materials.append(mat_track)
    for poly in track_obj.data.polygons: poly.use_smooth = True

    # 3. Generate Lane Lines & Chute Lines
    l_verts, l_faces = [], []
    offset = 0
    for i in range(num_lanes + 1):
        center_radius = inner_radius + (i * lane_width)
        v, f = get_track_ring_data(center_radius - (line_width / 2), center_radius + (line_width / 2), line_z)
        cy = -center_radius
        c_v_off = len(v)

        # Elevate straight lines by exactly 1 millimeter so they don't Z-fight the curved lines!
        line_z_chute = line_z + 0.001

        v.extend([
            (-straight_length / 2 - chute_length, cy - line_width / 2, line_z_chute),
            (-straight_length / 2, cy - line_width / 2, line_z_chute),
            (-straight_length / 2, cy + line_width / 2, line_z_chute),
            (-straight_length / 2 - chute_length, cy + line_width / 2, line_z_chute)
        ])
        f.append((c_v_off, c_v_off + 1, c_v_off + 2, c_v_off + 3))
        for face in f: l_faces.append(tuple(idx + offset for idx in face))
        l_verts.extend(v)
        offset += len(v)

    lines_mesh = bpy.data.meshes.new("Track_Lines_Mesh")
    lines_mesh.from_pydata(shift_verts(l_verts), [], l_faces)
    lines_obj = bpy.data.objects.new("2_Track_Lines", lines_mesh)
    track_col.objects.link(lines_obj)
    lines_obj.data.materials.append(mat_white)

    # 4. Generate Finish & 100m Start Lines
    start_x = finish_x - 100

    def create_horizontal_line(name, x_pos):
        verts = [(x_pos - line_width, -inner_radius, line_z * 2), (x_pos + line_width, -inner_radius, line_z * 2),
                 (x_pos + line_width, -r_track_out, line_z * 2), (x_pos - line_width, -r_track_out, line_z * 2)]
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        mesh.from_pydata(shift_verts(verts), [], [(0, 3, 2, 1)])
        obj = bpy.data.objects.new(f"3_{name}", mesh)
        track_col.objects.link(obj)
        obj.data.materials.append(mat_white)

    create_horizontal_line("Finish_Line", finish_x)
    create_horizontal_line("100m_Start_Line", start_x)

    # 4.5 Generate 5m Notches on BOTH sides of ALL lane lines
    def create_lane_notch(x_pos, y_center, length=0.35, width=0.025, mode="both"):
        verts = []
        faces = []
        idx = 0

        # TOP side
        if mode in ("both", "top"):
            verts.extend([
                (x_pos - width / 2, y_center + width / 2, line_z * 2),
                (x_pos + width / 2, y_center + width / 2, line_z * 2),
                (x_pos + width / 2, y_center + width / 2 + length, line_z * 2),
                (x_pos - width / 2, y_center + width / 2 + length, line_z * 2),
            ])
            faces.append((idx, idx + 3, idx + 2, idx + 1))
            idx += 4

        # BOTTOM side
        if mode in ("both", "bottom"):
            verts.extend([
                (x_pos - width / 2, y_center - width / 2, line_z * 2),
                (x_pos + width / 2, y_center - width / 2, line_z * 2),
                (x_pos + width / 2, y_center - width / 2 - length, line_z * 2),
                (x_pos - width / 2, y_center - width / 2 - length, line_z * 2),
            ])
            faces.append((idx, idx + 1, idx + 2, idx + 3))

        mesh = bpy.data.meshes.new("Notch_Mesh")
        mesh.from_pydata(shift_verts(verts), [], faces)

        obj = bpy.data.objects.new("5m_Notch", mesh)
        track_col.objects.link(obj)
        obj.data.materials.append(mat_white)

    # Create notches every 5m for all lane boundaries
    for i in range(5, 100, 5):
        x = start_x + i

        for lane_idx in range(num_lanes + 1):
            y = -(inner_radius + lane_idx * lane_width)
            if lane_idx == 0:
                create_lane_notch(x, y, mode="bottom")  # only into track
            elif lane_idx == num_lanes:
                create_lane_notch(x, y, mode="top")  # only into track
            else:
                create_lane_notch(x, y, mode="both")

    # ==========================================
    # --- 5. Generate Dynamic Infield        ---
    # ==========================================
    pitch_extension = 16.0
    theta_cut = math.acos(pitch_extension / inner_radius)

    def get_arc(cx, cy, r, a_start, a_end, z_val):
        while a_end < a_start: a_end += 2 * math.pi
        angle_diff = a_end - a_start
        steps = max(2, int(segments_per_curve * (angle_diff / math.pi)))
        pts = []
        for i in range(steps + 1):
            a = a_start + angle_diff * (i / steps)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z_val))
        return pts

    # 5a. Central Infield Turf
    arc1 = get_arc(finish_x, 0, inner_radius, theta_cut, math.pi / 2, grass_z)
    arc2 = get_arc(-finish_x, 0, inner_radius, math.pi / 2, math.pi - theta_cut, grass_z)
    arc3 = get_arc(-finish_x, 0, inner_radius, math.pi + theta_cut, 3 * math.pi / 2, grass_z)
    arc4 = get_arc(finish_x, 0, inner_radius, 3 * math.pi / 2, 2 * math.pi - theta_cut, grass_z)

    turf_verts = arc1[:-1] + arc2 + arc3[:-1] + arc4
    turf_mesh = bpy.data.meshes.new("Infield_Turf_Mesh")
    turf_mesh.from_pydata(shift_verts(turf_verts), [], [list(range(len(turf_verts)))])
    clean_ngon(turf_mesh)

    turf_obj = bpy.data.objects.new("4a_Infield_Turf", turf_mesh)
    track_col.objects.link(turf_obj)
    turf_obj.data.materials.append(mat_turf)

    # 5b. Right D-Zone
    rdz_verts = get_arc(finish_x, 0, inner_radius, -theta_cut, theta_cut, 0.0)
    rdz_mesh = bpy.data.meshes.new("Right_DZone_Mesh")
    rdz_mesh.from_pydata(shift_verts(rdz_verts), [], [list(range(len(rdz_verts)))])
    clean_ngon(rdz_mesh)

    rdz_obj = bpy.data.objects.new("4b_Right_DZone", rdz_mesh)
    track_col.objects.link(rdz_obj)
    rdz_obj.data.materials.append(mat_track)

    # 5c. Left D-Zone
    ldz_verts = get_arc(-finish_x, 0, inner_radius, math.pi - theta_cut, math.pi + theta_cut, 0.0)
    ldz_mesh = bpy.data.meshes.new("Left_DZone_Mesh")
    ldz_mesh.from_pydata(shift_verts(ldz_verts), [], [list(range(len(ldz_verts)))])
    clean_ngon(ldz_mesh)

    ldz_obj = bpy.data.objects.new("4c_Left_DZone", ldz_mesh)
    track_col.objects.link(ldz_obj)
    ldz_obj.data.materials.append(mat_track)

    # 6. Generate 3D Vertical 100m Markers
    for i in range(10, 91, 10):
        text_curve = bpy.data.curves.new(type="FONT", name=f"Marker_{i}m")
        text_curve.body = f"{i}m"
        text_curve.extrude = 0.05
        text_curve.size = 1.5
        text_curve.align_x = 'CENTER'
        text_curve.align_y = 'BOTTOM'

        text_obj = bpy.data.objects.new(f"Marker_{i}m", text_curve)
        markers_col.objects.link(text_obj)
        text_obj.location = (start_x + i + O_X, -inner_radius + 2.5 + O_Y, 0.0)
        text_obj.rotation_euler = (math.pi / 2, 0, 0)
        text_obj.data.materials.append(mat_white)

    # ==========================================
    # --- 7. Wrap-Around Grass Buffer        ---
    # ==========================================
    def get_stadium_path(radius, is_inner_grass=False):
        pts = []
        for i in range(segments_per_curve + 1):
            angle = -math.pi / 2 + (math.pi * i / segments_per_curve)
            pts.append((straight_length / 2 + radius * math.cos(angle), radius * math.sin(angle)))
        for i in range(segments_per_curve + 1):
            angle = math.pi / 2 + (math.pi * i / segments_per_curve)

            r_actual = radius
            # Match the grass boundary exactly to the new morphed track boundary
            if is_inner_grass:
                dx = math.cos(angle)
                dy = math.sin(angle)
                max_t = radius
                chute_left = -chute_length
                chute_top = -inner_radius
                chute_bottom = -r_track_out

                if dx < -0.00001:
                    t = chute_left / dx
                    if chute_bottom <= t * dy <= chute_top:
                        max_t = max(max_t, t)
                if dy < -0.00001:
                    t1 = chute_bottom / dy
                    if chute_left <= t1 * dx <= 0.0001:
                        max_t = max(max_t, t1)
                    t2 = chute_top / dy
                    if chute_left <= t2 * dx <= 0.0001:
                        max_t = max(max_t, t2)
                r_actual = max_t

            pts.append((-straight_length / 2 + r_actual * math.cos(angle), r_actual * math.sin(angle)))
        return pts

    # Outer Grass Buffer
    g_verts, g_faces = [], []
    in_path = get_stadium_path(r_track_out, is_inner_grass=True)
    out_path = get_stadium_path(r_grass_out, is_inner_grass=False)

    for p in in_path: g_verts.append((p[0], p[1], grass_z))
    for p in out_path: g_verts.append((p[0], p[1], grass_z))

    n = len(in_path)
    for i in range(n - 1):
        g_faces.append((i, i + n, i + 1 + n, i + 1))
    g_faces.append((n - 1, 2 * n - 1, n, 0))

    grass_mesh = bpy.data.meshes.new("Outer_Grass_Mesh")
    grass_mesh.from_pydata(shift_verts(g_verts), [], g_faces)
    clean_ngon(grass_mesh)

    grass_obj = bpy.data.objects.new("5_Outer_Grass", grass_mesh)
    stadium_col.objects.link(grass_obj)
    grass_obj.data.materials.append(mat_turf)

    # ==============================================
    # --- Billboard Ad with UVs ---
    # ==============================================
    def create_poster_material():
        """Generates the material with an open slot for an image texture."""
        mat_ad = bpy.data.materials.new(name="Stadium_Infield_Ad")
        mat_ad.use_nodes = True
        nodes = mat_ad.node_tree.nodes
        links = mat_ad.node_tree.links
        bsdf = nodes.get("Principled BSDF")

        # We'll set a subtle bezel color for the board's edge (mat_ad)
        # The poster content will be a separate surface (mat_poster)
        bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)

        # --- Separate material for the actual poster surface with texture ---
        mat_poster = bpy.data.materials.new(name="Stadium_Infield_Ad_Surface")
        mat_poster.use_nodes = True
        p_nodes = mat_poster.node_tree.nodes
        p_links = mat_poster.node_tree.links
        p_bsdf = p_nodes.get("Principled BSDF")

        p_bsdf.inputs['Roughness'].default_value = 1.0

        # --- Create and Link Image Texture Node (Where users add their image!) ---
        tex_image = p_nodes.new(type="ShaderNodeTexImage")

        # USER_ACTION: This is the slot where you can load your custom image texture!
        tex_image.image = bpy.data.images.load("bolt.png")

        # Link image texture to base color
        p_links.new(tex_image.outputs['Color'], p_bsdf.inputs['Base Color'])

        # --- Create simple UV Mapping node so the unwrap displays properly ---
        tex_coord = p_nodes.new(type="ShaderNodeTexCoord")
        mapping = p_nodes.new(type="ShaderNodeMapping")
        p_links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
        p_links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])

        return mat_ad, mat_poster

    def create_poster_ad_geometry(collection, mat_ad, mat_poster, finish_x, straight_length, inner_radius, lane_width):
        """Generates a large billboard on the straight and unwraps for a poster."""

        def shift_single_vert(v, straight_len, lane_w):
            lane_4_yc_orig = -(36.5 + (3.5 * lane_w))
            shift_y = -lane_4_yc_orig + 2.5
            shift_x = (straight_len / 2) - half_w + 62.5

            return (v[0] + shift_x, v[1] + shift_y, v[2])

        # --- Set Dimensions using your exact ratio ---
        img_width = 3472
        img_height = 1389

        # Set how tall you want the billboard to be in Blender meters.
        # The width will automatically calculate to match your image's ratio!
        target_height_m = 3.0
        target_width_m = target_height_m * (img_width / img_height)

        # Calculate half-width to center it on the X axis
        half_w = target_width_m / 2.0

        # Unshifted Y coordinate
        target_y = -(36.5 - 0.2)

        # Apply the mathematically correct width and height to the vertices
        v_l_t = (-half_w, target_y, target_height_m)
        v_l_b = (-half_w, target_y, 0.0)
        v_r_b = (half_w, target_y, 0.0)
        v_r_t = (half_w, target_y, target_height_m)

        s_verts = [shift_single_vert(v, straight_length, lane_width) for v in [v_l_t, v_l_b, v_r_b, v_r_t]]

        # --- 1. Create Mesh ---
        bm = bmesh.new()
        for v in s_verts:
            bm.verts.new(v)

        bm.verts.ensure_lookup_table()
        bm.faces.new([bm.verts[0], bm.verts[3], bm.verts[2], bm.verts[1]])

        mesh_ad = bpy.data.meshes.new("Infield_Ad_Mesh")
        bm.to_mesh(mesh_ad)
        bm.free()

        # --- 2. UV Unwrap ---
        bm = bmesh.new()
        bm.from_mesh(mesh_ad)
        uv_layer = bm.loops.layers.uv.new()

        bm.faces.ensure_lookup_table()
        face = bm.faces[0]

        loops = list(face.loops)
        loops[0][uv_layer].uv = (0, 1)
        loops[1][uv_layer].uv = (1, 1)
        loops[2][uv_layer].uv = (1, 0)
        loops[3][uv_layer].uv = (0, 0)

        bm.to_mesh(mesh_ad)
        bm.free()

        obj_ad = bpy.data.objects.new("Ad_Infield_Billboard", mesh_ad)
        collection.objects.link(obj_ad)

        obj_ad.data.materials.append(mat_poster)

        return obj_ad

    mat_ad_frame, mat_poster_surface = create_poster_material()
    create_poster_ad_geometry(track_col, mat_ad_frame, mat_poster_surface,
                              finish_x, straight_length, inner_radius, lane_width)
