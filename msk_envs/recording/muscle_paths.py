"""
Muscle paths and wrapping surfaces for the viewers, as Bolt's own renderer draws them (bolt/render/renderer.py):
- wrapping (Scholz2015GeometryPath) muscles follow polylines sampled over their obstacles
- inactive conditional path points are skipped

Every muscle keeps the same number of points in every frame (so the viewers can keep fixed buffers): inactive
conditional points collapse onto the previous active point, as obstacles that are not in contact collapse in
Bolt's wrap sampler.

Function-based paths have no geometry, so for models that use them the paths are taken from a copy of the model
loaded with geometry paths, posed like the simulated one. That copy is only used here, so the simulation is unchanged.
"""
import dataclasses

import bolt
import numpy as np
import warp as wp

from msk_envs.geometry import quat_mul_np, quat_rotate_np

try:
    from bolt._src.muscle import wrap_path
except ImportError:  # Bolt versions without muscle wrapping
    wrap_path = None


def _wrap_supported(m) -> bool:
    return wrap_path is not None and hasattr(m, "muscle_wrap_group_tuple")


def _function_based_muscles(m) -> set[int]:
    point_paths = set(getattr(m, "muscle_pt_group_tuple", range(m.nmuscle)))
    wrap_paths = set(getattr(m, "muscle_wrap_group_tuple", ()))
    return set(range(m.nmuscle)) - point_paths - wrap_paths


@dataclasses.dataclass
class WrapSurface:
    """ A wrapping obstacle, drawn once even when several muscles wrap over it """
    name: str
    wrap_type: int
    """ bolt.WrapType value: 0 sphere, 1 cylinder, 2 ellipsoid, 3 torus """
    size: list[float]
    """ sphere/cylinder: (radius, -, -), ellipsoid: radii, torus: (torus radius, tube radius, -) """
    half_length: float
    """ Cylinders are infinite; they are drawn this long along their z-axis """
    body_id: int
    X_BS: np.ndarray
    """ Surface frame in its body, [px, py, pz, qx, qy, qz, qw] """


class MusclePathSampler:
    """ Samples the muscle path polylines (and wrapping surface transforms) of every world of an MSKEnv """

    def __init__(self, envs):
        self.envs = envs
        main_m = envs.m
        self.copy = geometry_path_model(envs) if _function_based_muscles(main_m) else None
        if self.copy is not None:
            self.m, self.d = self.copy.model, self.copy.data
            # Muscle i of the env is muscle muscle_map[i] of the copy (same model, so normally the identity)
            copy_lookup = self.copy.muscle_id_lookup
            names = {v: k for k, v in envs.muscle_id_lookup.items()}
            self.muscle_map = np.array([copy_lookup[names[i]] for i in range(main_m.nmuscle)])
        else:
            self.m, self.d = main_m, envs.d
            self.muscle_map = np.arange(main_m.nmuscle)

        m = self.m
        self.pts_adr = m.muscle_pts_adr.numpy()
        self.pts_num = m.muscle_pts_num.numpy()

        # Conditional path points: the muscles that have any, and the qpos gating each site
        self.cond_qposadr = m.site_cond_qposadr.numpy() if hasattr(m, "site_cond_qposadr") else None
        self.cond_range = m.site_cond_range.numpy() if hasattr(m, "site_cond_range") else None
        self.cond_tol = getattr(bolt, "COND_PATH_POINT_RANGE_TOL", 1e-5)
        self.cond_muscles = set()
        if self.cond_qposadr is not None:
            for i in range(m.nmuscle):
                sl = slice(self.pts_adr[i], self.pts_adr[i] + self.pts_num[i])
                if (self.cond_qposadr[sl] >= 0).any():
                    self.cond_muscles.add(i)

        # Wrapping muscles: polylines from bolt's wrap sampler
        self.wrap_polyline = {}
        if _wrap_supported(m) and m.muscle_wrap_group_tuple:
            adr, num = wrap_path.wrap_render_layout(m)
            self.wrap_polyline = {i: (a, a + n) for i, a, n in zip(m.muscle_wrap_group_tuple, adr, num)}
            device = m.muscle_pts_adr.device
            self.wrap_render_adr = wp.array(adr, dtype=int, device=device)
            self.wrap_worlds = wp.array(np.arange(self.d.nworld), dtype=int, device=device)
            self.wrap_points = wp.empty((self.d.nworld, max(a + n for a, n in zip(adr, num))), dtype=wp.vec3,
                                        device=device)

        self.wrap_surfaces = self._collect_wrap_surfaces()

    def _collect_wrap_surfaces(self) -> list[WrapSurface]:
        m = self.m
        if not _wrap_supported(m) or not getattr(m, "nwrap", 0):
            return []
        wrap_types = m.wrap_type.numpy()
        sizes = m.wrap_size.numpy()
        body_ids = m.wrap_bodyid.numpy()
        X_BS = m.wrap_X_BS.numpy()
        half_length = getattr(wrap_path, "WRAP_RENDER_CYLINDER_HALF_LENGTH", 3.0)
        body_names = {v: k for k, v in self.envs.body_id_lookup.items()}
        surfaces, seen = [], set()
        for o in range(m.nwrap):
            # The same obstacle appears once per muscle that wraps over it
            key = (int(body_ids[o]), int(wrap_types[o]), *np.round(X_BS[o], 6), *np.round(sizes[o], 6))
            if key in seen:
                continue
            seen.add(key)
            size = [float(s) for s in sizes[o]]
            surfaces.append(WrapSurface(
                name=f"{body_names.get(int(body_ids[o]), 'body')}_wrap_{len(surfaces)}",
                wrap_type=int(wrap_types[o]),
                size=size,
                half_length=float(size[0] * half_length) if wrap_types[o] == 1 else 0.0,
                body_id=int(body_ids[o]),
                X_BS=X_BS[o].astype(np.float64),
            ))
        return surfaces

    def sample(self) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Returns ([num_worlds, num_points[i], 3] polyline of each muscle i, [num_worlds, num_wrap_surfaces, 7]
        wrapping surface transforms)
        """
        m, d = self.m, self.d
        if self.copy is not None:  # pose the geometry-path copy like the simulated model
            wp.copy(d.qpos, self.envs.d.qpos)
            bolt.fk(m, d)
        if self.wrap_polyline:
            if self.copy is not None:
                # Solve the copy's paths at this pose (the solve is skipped in worlds whose integration is done)
                d.integration_done.fill_(False)
                wrap_path.sample_wrap_paths(m, d, self.wrap_render_adr, self.wrap_worlds, self.wrap_points)
            else:
                # Sample the simulation's own wrapping solution without re-solving it, which would change its warm
                # start (and so the rest of the simulation)
                wp.launch(
                    wrap_path._sample_wrap_path_kernel,
                    dim=(d.nworld, m.muscle_wrap_group.size),
                    inputs=[
                        m.muscle_pts_adr, m.muscle_pts_num, m.muscle_wrap_adr, m.muscle_wrap_num, m.wrap_type,
                        m.wrap_bodyid, m.wrap_X_BS, m.wrap_size, m.wrap_segment, m.opt.wrap_geodesic_accuracy,
                        d.mob_X_GB, d.site_pos_G, d.wrap_geodesic,
                        m.muscle_wrap_group, self.wrap_render_adr, self.wrap_worlds,
                    ],
                    outputs=[self.wrap_points],
                    device=self.wrap_points.device,
                )
            wrap_points = self.wrap_points.numpy()
        site_pos = d.site_pos_G.numpy()
        qpos = d.qpos.numpy()
        site_active = self._site_active(qpos) if self.cond_muscles else None

        paths = []
        for i in self.muscle_map:
            if i in self.wrap_polyline:
                start, end = self.wrap_polyline[i]
                paths.append(wrap_points[:, start:end])
                continue
            start = self.pts_adr[i]
            pts = site_pos[:, start:start + self.pts_num[i]]
            if i in self.cond_muscles:
                pts = self._collapse_inactive(pts, site_active[:, start:start + self.pts_num[i]])
            paths.append(pts)

        return paths, self._wrap_surface_transforms(d)

    def _site_active(self, qpos: np.ndarray) -> np.ndarray:
        """ [num_worlds, num_sites] whether each path point is active (mirrors point_path.site_is_active) """
        conditional = self.cond_qposadr >= 0
        q = qpos[:, np.where(conditional, self.cond_qposadr, 0)]
        lo, hi = self.cond_range[:, 0], self.cond_range[:, 1]
        return ~conditional | ((q >= lo - self.cond_tol) & (q <= hi + self.cond_tol))

    @staticmethod
    def _collapse_inactive(pts: np.ndarray, active: np.ndarray) -> np.ndarray:
        """ Move inactive points onto the previous active point (the next one, before the first active point) """
        pts = pts.copy()
        n = pts.shape[1]
        idx = np.where(active, np.arange(n), -1)
        prev = np.maximum.accumulate(idx, axis=1)  # last active index at or before each point
        first = np.argmax(active, axis=1)  # first active point (all points are never inactive)
        src = np.where(prev >= 0, prev, first[:, None])
        return np.take_along_axis(pts, src[..., None], axis=1)

    def _wrap_surface_transforms(self, d) -> np.ndarray:
        if not self.wrap_surfaces:
            return np.zeros((d.nworld, 0, 7))
        X_GB = d.mob_X_GB.numpy()  # [num_worlds, num_bodies, 7]
        out = np.zeros((d.nworld, len(self.wrap_surfaces), 7))
        for k, s in enumerate(self.wrap_surfaces):
            X = X_GB[:, s.body_id]
            out[:, k, :3] = X[:, :3] + quat_rotate_np(X[:, 3:], s.X_BS[:3])
            out[:, k, 3:] = quat_mul_np(X[:, 3:], s.X_BS[3:])
        return out


def geometry_path_model(envs):
    """
    The model of [envs] loaded with geometry (instead of function-based) muscle paths, for drawing the paths.
    Loaded once per env. None when it can't be loaded, or when it has no more geometry than the simulated model.
    """
    if hasattr(envs, "_geometry_path_model"):
        return envs._geometry_path_model
    spec = getattr(envs, "geometry_model_spec", None)
    copy = None
    if spec is not None:
        try:
            if spec.get("scene") is not None:
                scene = dataclasses.replace(spec["scene"], models=[
                    dataclasses.replace(model, muscle_fn_path=None) for model in spec["scene"].models])
                copy = bolt.load_scene(scene=scene, n_worlds=envs.num_worlds, integrator=spec["integrator"],
                                       requires_visuals=False)
            else:
                copy = bolt.load_model(model_path=spec["model_path"], n_worlds=envs.num_worlds,
                                       integrator=spec["integrator"], requires_visuals=False, muscle_fn_path=None)
            if copy.model.nq != envs.m.nq or copy.model.nmuscle != envs.m.nmuscle:
                print("Geometry-path model doesn't match the simulated model; drawing muscles through their points")
                copy = None
        except Exception as e:  # drawing straight paths is better than failing the recording
            print(f"Could not load the geometry-path model for drawing muscles ({e}); drawing their points")
            copy = None
    envs._geometry_path_model = copy
    return copy
