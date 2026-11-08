// The muscle styles: every muscle of a trajectory drawn as tubes, capsules or lines (the "style" menu). Each class has
// object (to add to the scene), update(frame, values, scale, visible, tendons), pick(raycaster, viewportSize) and
// dispose(). See muscle_style.js for the shape and colors
import * as THREE from "three";
import {LineSegments2} from "three/addons/lines/LineSegments2.js";
import {LineSegmentsGeometry} from "three/addons/lines/LineSegmentsGeometry.js";
import {LineMaterial} from "three/addons/lines/LineMaterial.js";
import {numRings, tendonFraction, bellyRadius, radiusAt, valueColor, fiberWeight, TENDON_COLOR, CAPSULE_RADIUS}
    from "./muscle_style.js";

const SIDES = 10;  // vertices around each tube cross-section
const COS = Array.from({length: SIDES}, (_, s) => Math.cos(2 * Math.PI * s / SIDES));
const SIN = Array.from({length: SIDES}, (_, s) => Math.sin(2 * Math.PI * s / SIDES));

const srgbToLinear = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
const TENDON_LINEAR = TENDON_COLOR.map(srgbToLinear);

// Vertex colors are linear
function linearValueColor(value, out) {
    valueColor(value, out);
    for (let k = 0; k < 3; k++) out[k] = srgbToLinear(out[k]);
    return out;
}

// First point of each muscle in a frame's flat muscle_points
function pointAddresses(muscles) {
    const adr = [];
    let points = 0;
    for (const m of muscles) { adr.push(points); points += m.num_points; }
    return adr;
}

// cum[k]: arc length at point k of the polyline of n points at pts[start...] (it may repeat points); returns its length
function arcLengths(pts, start, n, cum) {
    cum[0] = 0;
    for (let k = 1; k < n; k++) {
        const o = start + 3 * k;
        cum[k] = cum[k - 1] + Math.hypot(pts[o] - pts[o - 3], pts[o + 1] - pts[o - 2], pts[o + 2] - pts[o - 1]);
    }
    return cum[n - 1];
}

function scratchArcLengths(muscles) {
    return new Float64Array(Math.max(2, ...muscles.map(m => m.num_points)));
}

// --- Tubes: thick bellies, thin tendons ---

// One mesh, rebuilt in place every frame
export class MuscleTubes {
    constructor(muscles) {
        this.muscles = muscles;
        this.pointAdr = pointAddresses(muscles);
        this.rings = [];
        this.vertAdr = [];
        this.trianglesBefore = [];  // first triangle of each muscle, for picking
        const indices = [];  // triangles between consecutive rings of each muscle
        let verts = 0;
        for (const m of muscles) {
            const rings = numRings(m.num_points);
            this.rings.push(rings);
            this.vertAdr.push(verts);
            this.trianglesBefore.push(indices.length / 3);
            for (let r = 0; r < rings - 1; r++) {
                for (let s = 0; s < SIDES; s++) {
                    const a = verts + r * SIDES + s, b = verts + r * SIDES + (s + 1) % SIDES;
                    indices.push(a, a + SIDES, b, b, a + SIDES, b + SIDES);
                }
            }
            verts += rings * SIDES;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setIndex(verts > 65535 ? new THREE.Uint32BufferAttribute(indices, 1) : new THREE.Uint16BufferAttribute(indices, 1));
        this.positions = new Float32Array(verts * 3);
        this.normals = new Float32Array(verts * 3);
        this.colors = new Float32Array(verts * 3);
        geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
        geometry.setAttribute("normal", new THREE.BufferAttribute(this.normals, 3).setUsage(THREE.DynamicDrawUsage));
        geometry.setAttribute("color", new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));
        this.geometry = geometry;
        this.material = new THREE.MeshStandardMaterial({
            vertexColors: true, roughness: 0.55, metalness: 0.0, side: THREE.DoubleSide,
        });
        this.object = new THREE.Mesh(geometry, this.material);
        this.object.castShadow = true;
        this.object.receiveShadow = true;
        this.object.frustumCulled = false;

        // Scratch buffers
        const maxRings = Math.max(2, ...this.rings);
        this.cum = scratchArcLengths(muscles);
        this.centers = new Float64Array(maxRings * 3);
        this.tangents = new Float64Array(maxRings * 3);
        this.color = [0, 0, 0];
    }

    // Index of the muscle the ray hits first, or -1
    pick(raycaster) {
        const hit = raycaster.intersectObject(this.object, false)[0];
        return hit ? this.muscleAtFace(hit.faceIndex) : -1;
    }

    muscleAtFace(faceIndex) {
        // Last muscle whose first triangle is at or before faceIndex
        let lo = 0, hi = this.trianglesBefore.length - 1;
        while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (this.trianglesBefore[mid] <= faceIndex) lo = mid; else hi = mid - 1;
        }
        return lo;
    }

    // frame: animation frame; values: per-muscle colormap value in [0, 1]; scale: user thickness scale; visible:
    // per-muscle flags (or null for all); tendons: false to draw the whole path as fibers (belly), in the value color
    update(frame, values, scale, visible, tendons = true) {
        const pts = frame.muscle_points;
        const data = frame.muscles || {};
        for (let i = 0; i < this.muscles.length; i++) {
            const m = this.muscles[i];
            const hidden = visible && !visible[i];
            const tendonFrac = tendons ? tendonFraction(data.path_length?.[i], data.tendon_length?.[i]) : 0;
            const belly = bellyRadius(m.max_isometric_force, scale, data.fiber_length?.[i], m.optimal_fiber_length);
            linearValueColor(values[i], this.color);
            this.buildTube(i, pts, 3 * this.pointAdr[i], m.num_points, hidden ? 0 : belly, tendonFrac, tendons);
        }
        this.geometry.attributes.position.needsUpdate = true;
        this.geometry.attributes.normal.needsUpdate = true;
        this.geometry.attributes.color.needsUpdate = true;
        this.geometry.boundingSphere = null;  // recomputed when picking
    }

    buildTube(i, pts, start, n, belly, tendonFrac, tendons) {
        const rings = this.rings[i];
        const C = this.centers, T = this.tangents, cum = this.cum;
        const length = arcLengths(pts, start, n, cum);
        // Evenly spaced centers
        let seg = 0;
        for (let r = 0; r < rings; r++) {
            const s = length * r / (rings - 1);
            while (seg < n - 2 && cum[seg + 1] < s) seg++;
            const span = cum[seg + 1] - cum[seg];
            const t = span > 1e-12 ? Math.min(1, Math.max(0, (s - cum[seg]) / span)) : 0;
            const o = start + 3 * seg;
            for (let k = 0; k < 3; k++) C[3 * r + k] = pts[o + k] + t * (pts[o + 3 + k] - pts[o + k]);
        }
        // Tangents
        for (let r = 0; r < rings; r++) {
            const a = Math.max(0, r - 1), b = Math.min(rings - 1, r + 1);
            let tx = C[3 * b] - C[3 * a], ty = C[3 * b + 1] - C[3 * a + 1], tz = C[3 * b + 2] - C[3 * a + 2];
            const len = Math.hypot(tx, ty, tz);
            if (len > 1e-12) { tx /= len; ty /= len; tz /= len; } else { tx = 0; ty = 1; tz = 0; }
            T[3 * r] = tx; T[3 * r + 1] = ty; T[3 * r + 2] = tz;
        }
        // Rotation-minimizing frames: carry the normal along, starting with any one (picked on the first ring)
        let nx = 0, ny = 0, nz = 0;
        const P = this.positions, N = this.normals, K = this.colors;
        const [mr, mg, mb] = this.color;
        const [tr, tg, tb] = TENDON_LINEAR;
        let v = this.vertAdr[i] * 3;
        for (let r = 0; r < rings; r++) {
            const tx = T[3 * r], ty = T[3 * r + 1], tz = T[3 * r + 2];
            const d = nx * tx + ny * ty + nz * tz;
            nx -= d * tx; ny -= d * ty; nz -= d * tz;
            let l = Math.hypot(nx, ny, nz);
            if (l < 1e-6) {  // cross(t, the axis least aligned with t)
                if (Math.abs(tx) < 0.9) { nx = 0; ny = -tz; nz = ty; } else { nx = tz; ny = 0; nz = -tx; }
                l = Math.hypot(nx, ny, nz);
            }
            nx /= l; ny /= l; nz /= l;
            const bx = ty * nz - tz * ny, by = tz * nx - tx * nz, bz = tx * ny - ty * nx;
            const radius = belly > 0 && length > 1e-6 ? radiusAt(r / (rings - 1), belly, tendonFrac) : 0;
            const cx = C[3 * r], cy = C[3 * r + 1], cz = C[3 * r + 2];
            const w = radius > 0 && tendons ? fiberWeight(radius, belly) : 1;
            const cr = tr + w * (mr - tr), cg = tg + w * (mg - tg), cb = tb + w * (mb - tb);
            for (let s = 0; s < SIDES; s++) {
                const ox = COS[s] * nx + SIN[s] * bx, oy = COS[s] * ny + SIN[s] * by, oz = COS[s] * nz + SIN[s] * bz;
                P[v] = cx + radius * ox; P[v + 1] = cy + radius * oy; P[v + 2] = cz + radius * oz;
                N[v] = ox; N[v + 1] = oy; N[v + 2] = oz;
                K[v] = cr; K[v + 1] = cg; K[v + 2] = cb;
                v += 3;
            }
        }
    }

    dispose() {
        this.geometry.dispose();
        this.material.dispose();
    }
}

// --- Capsules and lines: straight segments between path points ---

// Muscles drawn by straight segments between their path points (skipping collapsed ones), split where the path
// enters and leaves the fibers, so each segment is all tendon or all fiber
class SegmentMuscles {
    constructor(muscles) {
        this.muscles = muscles;
        this.pointAdr = pointAddresses(muscles);
        // Most segments: two more per muscle, for the splits at the tendon ends
        this.maxSegments = Math.max(1, muscles.reduce((sum, m) => sum + Math.max(0, m.num_points - 1) + 2, 0));
        this.cum = scratchArcLengths(muscles);
        this.color = [0, 0, 0];  // linear value color of the current muscle
    }

    // Calls add(i, x0, y0, z0, x1, y1, z1, rgb, first) for each segment of each visible muscle i: rgb is the linear
    // value color, or the tendon color along the tendons (with tendons), first whether it is the muscle's first
    forEachSegment(frame, values, visible, tendons, add) {
        const pts = frame.muscle_points;
        const data = frame.muscles || {};
        const cum = this.cum;
        for (let i = 0; i < this.muscles.length; i++) {
            if (visible && !visible[i]) continue;
            const n = this.muscles[i].num_points, start = 3 * this.pointAdr[i];
            const length = arcLengths(pts, start, n, cum);
            if (!(length > 1e-6)) continue;
            linearValueColor(values[i], this.color);
            const frac = tendonFraction(data.path_length?.[i], data.tendon_length?.[i]);
            const cut0 = length * frac / 2, cut1 = length * (1 - frac / 2);  // tendon | fibers | tendon
            let first = true;
            for (let k = 0; k < n - 1; k++) {
                const s0 = cum[k], s1 = cum[k + 1], span = s1 - s0;
                if (span < 1e-9) continue;  // collapsed (inactive) points
                const o = start + 3 * k;
                const dx = pts[o + 3] - pts[o], dy = pts[o + 4] - pts[o + 1], dz = pts[o + 5] - pts[o + 2];
                // Pieces ending at the cuts inside this segment, then at its end
                let from = s0;
                for (let c = 0; c < 3; c++) {
                    const to = c === 0 ? cut0 : c === 1 ? cut1 : s1;
                    if (c < 2 && !(to > s0 && to < s1)) continue;
                    const t0 = (from - s0) / span, t1 = (to - s0) / span;
                    const mid = (from + to) / 2;
                    const rgb = tendons && (mid < cut0 || mid > cut1) ? TENDON_LINEAR : this.color;
                    add(i, pts[o] + t0 * dx, pts[o + 1] + t0 * dy, pts[o + 2] + t0 * dz,
                        pts[o] + t1 * dx, pts[o + 1] + t1 * dy, pts[o + 2] + t1 * dz, rgb, first);
                    first = false;
                    from = to;
                }
            }
        }
    }
}

// Plain line segments of constant width on screen (LINE_WIDTH_PX times the thickness scale)
const LINE_WIDTH_PX = 2.5;

export class MuscleLines extends SegmentMuscles {
    constructor(muscles) {
        super(muscles);
        this.positions = new Float32Array(6 * this.maxSegments);
        this.colors = new Float32Array(6 * this.maxSegments);
        this.segmentMuscle = new Int32Array(this.maxSegments);  // muscle of each drawn segment, for picking
        this.geometry = new LineSegmentsGeometry();
        this.geometry.setPositions(this.positions);
        this.geometry.setColors(this.colors);
        this.material = new LineMaterial({vertexColors: true, linewidth: LINE_WIDTH_PX, worldUnits: false});
        this.object = new LineSegments2(this.geometry, this.material);
        this.object.frustumCulled = false;
    }

    pick(raycaster, viewportSize) {
        // The line width is in pixels of the view's viewport, which raycasting needs
        if (viewportSize) this.material.resolution.copy(viewportSize);
        raycaster.params.Line2 = {threshold: 4};
        const hit = raycaster.intersectObject(this.object, false)[0];
        return hit ? this.segmentMuscle[hit.faceIndex] : -1;
    }

    update(frame, values, scale, visible, tendons = true) {
        const P = this.positions, K = this.colors;
        let seg = 0;
        this.forEachSegment(frame, values, visible, tendons, (i, x0, y0, z0, x1, y1, z1, [r, g, b]) => {
            const o = 6 * seg;
            P[o] = x0; P[o + 1] = y0; P[o + 2] = z0; P[o + 3] = x1; P[o + 4] = y1; P[o + 5] = z1;
            K[o] = K[o + 3] = r; K[o + 1] = K[o + 4] = g; K[o + 2] = K[o + 5] = b;
            this.segmentMuscle[seg++] = i;
        });
        this.geometry.instanceCount = seg;
        this.geometry.attributes.instanceStart.data.needsUpdate = true;
        this.geometry.attributes.instanceColorStart.data.needsUpdate = true;
        this.geometry.boundingBox = this.geometry.boundingSphere = null;  // recomputed when picking
        this.material.linewidth = LINE_WIDTH_PX * scale;
    }

    dispose() {
        this.geometry.dispose();
        this.material.dispose();
    }
}

// A chain of capsules of radius CAPSULE_RADIUS (times the thickness scale), one per segment, as the dashboard drew
// muscles before: instanced cylinders, with a sphere at each joint and end
export class MuscleCapsules extends SegmentMuscles {
    constructor(muscles) {
        super(muscles);
        const max = this.maxSegments;
        this.material = new THREE.MeshStandardMaterial({roughness: 0.55, metalness: 0.0});
        const instanced = (geometry, count) => {
            const mesh = new THREE.InstancedMesh(geometry, this.material, count);
            mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(3 * count), 3)
                .setUsage(THREE.DynamicDrawUsage);
            mesh.castShadow = mesh.receiveShadow = true;
            mesh.frustumCulled = false;
            mesh.count = 0;
            return mesh;
        };
        this.cylinders = instanced(new THREE.CylinderGeometry(1, 1, 1, 12, 1, true), max);  // unit, along y
        this.spheres = instanced(new THREE.SphereGeometry(1, 12, 8), max + muscles.length);
        this.object = new THREE.Group();
        this.object.add(this.cylinders, this.spheres);
        this.cylinderMuscle = new Int32Array(max);  // for picking
        this.sphereMuscle = new Int32Array(max + muscles.length);
        this.m = new THREE.Matrix4();
        this.p = new THREE.Vector3();
        this.q = new THREE.Quaternion();
        this.s = new THREE.Vector3();
        this.d = new THREE.Vector3();
        this.up = new THREE.Vector3(0, 1, 0);
    }

    pick(raycaster) {
        const hit = raycaster.intersectObjects([this.cylinders, this.spheres], false)[0];
        if (!hit) return -1;
        return (hit.object === this.cylinders ? this.cylinderMuscle : this.sphereMuscle)[hit.instanceId];
    }

    update(frame, values, scale, visible, tendons = true) {
        const radius = CAPSULE_RADIUS * scale;
        const {m, p, q, s, d, up} = this;
        const cylColors = this.cylinders.instanceColor.array, sphColors = this.spheres.instanceColor.array;
        let nc = 0, ns = 0;
        const sphere = (i, x, y, z, rgb) => {
            this.spheres.setMatrixAt(ns, m.makeScale(radius, radius, radius).setPosition(x, y, z));
            sphColors.set(rgb, 3 * ns);
            this.sphereMuscle[ns++] = i;
        };
        this.forEachSegment(frame, values, visible, tendons, (i, x0, y0, z0, x1, y1, z1, rgb, first) => {
            d.set(x1 - x0, y1 - y0, z1 - z0);
            const length = d.length();
            q.setFromUnitVectors(up, d.divideScalar(length));
            p.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
            this.cylinders.setMatrixAt(nc, m.compose(p, q, s.set(radius, length, radius)));
            cylColors.set(rgb, 3 * nc);
            this.cylinderMuscle[nc++] = i;
            // A sphere at the end of each segment, and at the start of the first (none at a segment's start, where
            // it would overlap the previous one's end sphere in another color)
            if (first) sphere(i, x0, y0, z0, rgb);
            sphere(i, x1, y1, z1, rgb);
        });
        for (const [mesh, count] of [[this.cylinders, nc], [this.spheres, ns]]) {
            mesh.count = count;
            mesh.instanceMatrix.needsUpdate = true;
            mesh.instanceColor.needsUpdate = true;
            mesh.boundingSphere = null;  // recomputed when picking
        }
    }

    dispose() {
        this.cylinders.geometry.dispose();
        this.spheres.geometry.dispose();
        this.material.dispose();
    }
}
