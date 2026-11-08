"""
How the viewers draw muscles: a tube along the muscle's path, thin along the tendons at both ends and thicker along
the fibers, with a color for its activation. Only needs numpy (Blender imports it).
The dashboard has the same functions in dashboard/js/muscle_style.js: keep them in sync.
"""
import numpy as np

BELLY_RADIUS_PER_SQRT_N = 0.007 / np.sqrt(1000.0)
""" Fiber (belly) radius at optimal fiber length: 0.7 cm for a 1000 N muscle, then scaled by the user's muscle scale """
TENDON_RADIUS_RATIO = 0.3
""" Tendon radius, relative to the belly """
MIN_TENDON_RADIUS = 0.0012
DEFAULT_TENDON_FRACTION = 0.3
""" Of the path length, when the recording has no fiber/tendon lengths """
BELLY_SHAPE_EXPONENT = 0.6
""" Belly radius ~ sin(pi u)^exponent along the fibers (u from 0 to 1): smaller is fuller """
VOLUME_FACTOR_RANGE = (0.75, 1.35)
""" Constant fiber volume: the belly thickens as the fibers shorten (radius ~ 1 / sqrt(length)), within this range """

TENDON_COLOR = (0.86, 0.83, 0.76)
""" sRGB, blended in where the tube thins to the tendon (see fiber_weight) """

CAPSULE_RADIUS = 0.005
""" Radius (m) of the "capsules" muscle style (both viewers): a capsule per segment between path points """

LINE_RADIUS = 0.002
""" Radius (m) of the thin, uniform tubes of Blender's "lines" muscle style (the dashboard's lines are 2.5 px wide) """

COLOR_STOPS = (
    (0.0, (0.16, 0.30, 0.80)),
    (0.5, (0.58, 0.22, 0.58)),
    (1.0, (0.90, 0.10, 0.10)),
)
""" Activation 0 -> blue, 1 -> red (as Bolt's renderer), in sRGB """


def num_rings(num_points: int) -> int:
    """ Tube cross-sections of a muscle whose path has [num_points] points """
    return int(min(48, max(20, 2 * num_points)))


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    """ Arc length at each point of the polyline [points] ([k, 3]) """
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def resample_polyline(points: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """ [n, 3] points evenly spaced along the polyline [points] ([k, 3], may repeat points), and its length """
    points = np.asarray(points, dtype=np.float64)
    cum = _arc_lengths(points)
    length = float(cum[-1])
    if length < 1e-9:
        return np.repeat(points[:1], n, axis=0), 0.0
    s = np.linspace(0.0, length, n)
    return np.stack([np.interp(s, cum, points[:, k]) for k in range(3)], axis=1), length


def tendon_fraction(path_length, tendon_length) -> float:
    if not path_length or tendon_length is None or path_length <= 1e-6 or not np.isfinite(tendon_length):
        return DEFAULT_TENDON_FRACTION
    return float(np.clip(tendon_length / path_length, 0.02, 0.9))


def belly_radius(max_isometric_force: float, scale: float = 1.0, fiber_length=None, optimal_fiber_length=None):
    radius = BELLY_RADIUS_PER_SQRT_N * np.sqrt(max(max_isometric_force, 1.0)) * scale
    if fiber_length and optimal_fiber_length and fiber_length > 1e-6:
        radius *= float(np.clip(np.sqrt(optimal_fiber_length / fiber_length), *VOLUME_FACTOR_RANGE))
    return radius


def tendon_radius(belly: float) -> float:
    return max(MIN_TENDON_RADIUS, TENDON_RADIUS_RATIO * belly)


def radius_profile(n: int, belly: float, tendon_frac: float) -> np.ndarray:
    """ [n] tube radius along the path: tendons at both ends (half the tendon length each), fibers between """
    u = np.linspace(0.0, 1.0, n)
    a, b = tendon_frac / 2.0, 1.0 - tendon_frac / 2.0
    tendon = tendon_radius(belly)
    w = np.clip((u - a) / max(b - a, 1e-6), 0.0, 1.0)
    return tendon + (max(belly, tendon) - tendon) * np.sin(np.pi * w) ** BELLY_SHAPE_EXPONENT


def value_color(value: float) -> tuple[float, float, float]:
    """ sRGB color for a value in [0, 1] (e.g. activation) """
    value = float(np.clip(value, 0.0, 1.0))
    for (u0, c0), (u1, c1) in zip(COLOR_STOPS[:-1], COLOR_STOPS[1:]):
        if value <= u1:
            t = (value - u0) / (u1 - u0)
            return tuple(c0[k] + t * (c1[k] - c0[k]) for k in range(3))
    return COLOR_STOPS[-1][1]


def fiber_weight(radius: np.ndarray, belly: float) -> np.ndarray:
    """ 0 along the tendons, rising to 1 within the fibers: how much of the value color (vs TENDON_COLOR) to show """
    tendon = tendon_radius(belly)
    if belly <= tendon:
        return np.ones_like(radius)
    return np.clip(3.0 * (radius - tendon) / (belly - tendon), 0.0, 1.0)


def line_fiber_weight(n: int, tendon_frac: float) -> np.ndarray:
    """ For the "lines" style: 1 along the fibers, 0 along the tendons (half the tendon length at each end) """
    u = np.linspace(0.0, 1.0, n)
    return ((u >= tendon_frac / 2) & (u <= 1 - tendon_frac / 2)).astype(np.float64)


def split_at_tendons(points: np.ndarray, tendon_frac: float):
    """
    For the "capsules" style: the path [points] ([k, 3]) with each point where it enters or leaves the fibers
    inserted twice (the tendon-side copy, then the fiber-side one, in path order), so every segment is all tendon or
    all fiber. Returns points [k + 4, 3], fiber weights [k + 4] (1 along the fibers, 0 along the tendons) and radius
    scales [k + 4] (just under 1 for tendon-side copies, so the fiber-side joint covers it)
    """
    points = np.asarray(points, dtype=np.float64)
    cum = _arc_lengths(points)
    length = cum[-1]
    a, b = length * tendon_frac / 2, length * (1 - tendon_frac / 2)
    out, fiber, scale = [], [], []
    cuts = [(a, (0.0, 1.0)), (b, (1.0, 0.0))]  # (arc length, fiber weight of the two copies)
    for k in range(len(points)):
        while cuts and cuts[0][0] <= cum[k] and length > 1e-9:
            s, weights = cuts.pop(0)
            p = np.array([np.interp(s, cum, points[:, j]) for j in range(3)])
            for w in weights:
                out.append(p)
                fiber.append(w)
                scale.append(1.0 if w == 1.0 else 0.995)
        out.append(points[k])
        fiber.append(1.0 if a < cum[k] < b else 0.0)
        scale.append(1.0)
    while len(out) < len(points) + 4:  # a path of length 0
        out.append(points[-1])
        fiber.append(0.0)
        scale.append(1.0)
    return np.asarray(out), np.asarray(fiber), np.asarray(scale)


def muscle_values(muscles: list[dict], frame_muscles: dict, color_by: str) -> np.ndarray:
    """
    Per-muscle colormap value in a frame (frame["muscles"] of a version 2 animation): "activation", "excitation", or
    "force" (actuation / max isometric force)
    """
    if color_by == "force":
        fmax = np.array([m["max_isometric_force"] for m in muscles])
        return np.asarray(frame_muscles.get("actuation", np.zeros(len(muscles)))) / fmax
    values = frame_muscles.get(color_by)
    return np.asarray(values) if values is not None and len(values) == len(muscles) else np.zeros(len(muscles))
