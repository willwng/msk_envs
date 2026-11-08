// How muscles are drawn: the same as msk_envs/muscle_style.py (used by Blender). Keep them in sync.

export const BELLY_RADIUS_PER_SQRT_N = 0.007 / Math.sqrt(1000.0);
export const TENDON_RADIUS_RATIO = 0.3;
export const MIN_TENDON_RADIUS = 0.0012;
export const DEFAULT_TENDON_FRACTION = 0.3;
export const BELLY_SHAPE_EXPONENT = 0.6;
export const VOLUME_FACTOR_RANGE = [0.75, 1.35];

// sRGB, blended in where the tube thins to the tendon (see fiberWeight)
export const TENDON_COLOR = [0.86, 0.83, 0.76];

// Radius (m) of the "capsules" style: a capsule per segment between path points
export const CAPSULE_RADIUS = 0.005;

// Value 0 -> blue, 1 -> red (as Bolt's renderer), in sRGB
export const COLOR_STOPS = [
    [0.0, [0.16, 0.30, 0.80]],
    [0.5, [0.58, 0.22, 0.58]],
    [1.0, [0.90, 0.10, 0.10]],
];

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function numRings(numPoints) {
    return Math.min(48, Math.max(20, 2 * numPoints));
}

export function tendonFraction(pathLength, tendonLength) {
    if (!pathLength || tendonLength === undefined || tendonLength === null || pathLength <= 1e-6
        || !isFinite(tendonLength)) return DEFAULT_TENDON_FRACTION;
    return clamp(tendonLength / pathLength, 0.02, 0.9);
}

export function bellyRadius(maxIsometricForce, scale = 1.0, fiberLength = null, optimalFiberLength = null) {
    let radius = BELLY_RADIUS_PER_SQRT_N * Math.sqrt(Math.max(maxIsometricForce, 1.0)) * scale;
    if (fiberLength && optimalFiberLength && fiberLength > 1e-6) {
        radius *= clamp(Math.sqrt(optimalFiberLength / fiberLength), ...VOLUME_FACTOR_RANGE);
    }
    return radius;
}

export function tendonRadius(belly) {
    return Math.max(MIN_TENDON_RADIUS, TENDON_RADIUS_RATIO * belly);
}

// Radius at u in [0, 1] along the path: tendons at both ends (half the tendon length each), fibers between
export function radiusAt(u, belly, tendonFrac) {
    const a = tendonFrac / 2, b = 1 - tendonFrac / 2;
    const tendon = tendonRadius(belly);
    const w = clamp((u - a) / Math.max(b - a, 1e-6), 0, 1);
    return tendon + (Math.max(belly, tendon) - tendon) * Math.pow(Math.sin(Math.PI * w), BELLY_SHAPE_EXPONENT);
}

// 0 along the tendons, rising to 1 within the fibers: how much of the value color (vs TENDON_COLOR) to show
export function fiberWeight(radius, belly) {
    const tendon = tendonRadius(belly);
    if (belly <= tendon) return 1;
    return clamp(3 * (radius - tendon) / (belly - tendon), 0, 1);
}

// sRGB color [r, g, b] for a value in [0, 1]
export function valueColor(value, out = [0, 0, 0]) {
    value = clamp(value, 0, 1);
    for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
        const [u0, c0] = COLOR_STOPS[i], [u1, c1] = COLOR_STOPS[i + 1];
        if (value <= u1) {
            const t = (value - u0) / (u1 - u0);
            for (let k = 0; k < 3; k++) out[k] = c0[k] + t * (c1[k] - c0[k]);
            return out;
        }
    }
    const last = COLOR_STOPS[COLOR_STOPS.length - 1][1];
    for (let k = 0; k < 3; k++) out[k] = last[k];
    return out;
}

// Per-muscle colormap value in a frame (its "muscles"): "activation", "excitation", or "force" (actuation / max
// isometric force)
export function muscleValues(muscles, frameMuscles, colorBy) {
    const data = frameMuscles || {};
    if (colorBy === "force") return muscles.map((m, i) => (data.actuation?.[i] || 0) / m.max_isometric_force);
    const values = data[colorBy];
    return values && values.length === muscles.length ? values : new Array(muscles.length).fill(0);
}

export function colorGradientCss() {
    return "linear-gradient(to right, " + COLOR_STOPS.map(([u, c]) =>
        `rgb(${c.map(x => Math.round(255 * x)).join(",")}) ${100 * u}%`).join(", ") + ")";
}
