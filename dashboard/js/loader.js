import * as THREE from "three";
import {OBJLoader} from "three/addons/loaders/OBJLoader.js";
import {VTKLoader} from "three/addons/loaders/VTKLoader.js";
import {mergeGeometries, mergeVertices} from "three/addons/utils/BufferGeometryUtils.js";

const objLoader = new OBJLoader();
const vtkLoader = new VTKLoader();
const geometryCache = new Map();  // url -> Promise<BufferGeometry>

// Positions only, non-indexed, so geometries can be merged and their normals recomputed
function stripped(geometry) {
    const g = geometry.index ? geometry.toNonIndexed() : geometry;
    for (const name of Object.keys(g.attributes)) if (name !== "position") g.deleteAttribute(name);
    g.clearGroups();
    return g;
}

// Bone meshes: one smooth-shaded BufferGeometry per file, shared by every visual that uses it
export function loadMeshGeometry(url) {
    if (!geometryCache.has(url)) {
        geometryCache.set(url, new Promise((resolve, reject) => {
            const done = geometry => {
                geometry = mergeVertices(stripped(geometry), 1e-6);
                geometry.computeVertexNormals();
                resolve(geometry);
            };
            if (url.toLowerCase().endsWith(".vtp")) {
                vtkLoader.load(url, done, undefined, reject);
            } else {
                objLoader.load(url, group => {
                    const geometries = [];
                    group.traverse(child => { if (child.isMesh) geometries.push(child.geometry); });
                    if (!geometries.length) { reject(new Error(`no mesh in ${url}`)); return; }
                    // A file may hold several objects (e.g. torso.obj: ribs, and spine with the skull)
                    done(geometries.length > 1 ? mergeGeometries(geometries.map(stripped)) : geometries[0]);
                }, undefined, reject);
            }
        }));
    }
    return geometryCache.get(url);
}

// Rotates three's y-axis primitives (capsules, cylinders) onto z, the axis of Bolt's capsules and wrap cylinders
const Y_TO_Z = new THREE.Matrix4().makeRotationX(Math.PI / 2);

// A collider (bolt.GeomType, see msk_envs/trajectory_io.py), in its local frame; null for ones that aren't drawn.
// segments/rings: of spheres and cylinders; open: cylinders without caps
export function colliderGeometry(geomType, size, {segments = 24, rings = 16, open = false} = {}) {
    switch (geomType) {
        case 2:
            return new THREE.SphereGeometry(size[0], segments, rings);
        case 3:
            return new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 20).applyMatrix4(Y_TO_Z);
        case 4:
            return new THREE.SphereGeometry(1, segments, rings).scale(size[0], size[1], size[2]);
        case 5:
            return new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], segments, 1, open).applyMatrix4(Y_TO_Z);
        case 6:
            return new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
        default:
            return null;
    }
}

// Wrapping surfaces (bolt.WrapType) drawn as the collider of this bolt.GeomType: sphere, cylinder, ellipsoid
const WRAP_TO_GEOM_TYPE = {0: 2, 1: 5, 2: 4};

// A muscle wrapping surface, in its surface frame
export function wrapSurfaceGeometry(surface) {
    const s = surface.size;
    if (surface.wrap_type === 3) return new THREE.TorusGeometry(s[0], s[1], 16, 48);
    const geomType = WRAP_TO_GEOM_TYPE[surface.wrap_type];
    if (geomType === undefined) return null;
    // Cylinders are infinite: drawn half_length (recorded) along z, open
    const size = surface.wrap_type === 1 ? [s[0], surface.half_length] : s;
    return colliderGeometry(geomType, size, {segments: 32, rings: 20, open: true});
}

// Ground styles (sRGB): two alternating colors, as 5 m mowing stripes across the running direction or 1 m checker
// squares, grid lines every 10 cm and 1 m, and optional speckle (grass texture). "turf" is the Blender track's grass
// (blender/track.py), darkened for the dashboard's brighter lighting
export const GROUND_STYLES = {
    turf: {label: "stadium turf", colors: ["#11330f", "#184014"], pattern: "stripes", fine: 0.035, major: 0.14, speckle: 0.12},
    grass: {label: "light grass", colors: ["#2d5121", "#355a29"], pattern: "stripes", fine: 0.04, major: 0.16, speckle: 0.10},
    charcoal: {label: "charcoal", colors: ["#1c1e21", "#202326"], pattern: "checker", fine: 0.05, major: 0.14, speckle: 0},
    slate: {label: "slate", colors: ["#353b41", "#31363c"], pattern: "checker", fine: 0.07, major: 0.18, speckle: 0},
    tartan: {label: "tartan", colors: ["#5e2820", "#642c23"], pattern: "stripes", fine: 0.04, major: 0.18, speckle: 0.08},
};
export const DEFAULT_GROUND = "grass";

// Grid lines drawn on the ground
export const GROUND_GRIDS = {fine: "10 cm + 1 m", meter: "1 m", none: "none"};
export const DEFAULT_GRID = "meter";

// The ground texture of a style with grid lines [grid] (a GROUND_GRIDS key), and the size (m) of one tile of it
export function groundTexture(renderer, styleName = DEFAULT_GROUND, grid = DEFAULT_GRID) {
    const style = GROUND_STYLES[styleName] || GROUND_STYLES[DEFAULT_GROUND];
    const stripes = style.pattern === "stripes";
    const meters = stripes ? 10 : 2;  // a tile: two 5 m stripes, or 2 x 2 squares of 1 m
    const size = stripes ? 2048 : 512;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d");
    const [a, b] = style.colors;
    ctx.fillStyle = a;
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = b;
    if (stripes) {
        ctx.fillRect(size / 2, 0, size / 2, size);  // across the running direction (+x)
    } else {
        ctx.fillRect(0, 0, size / 2, size / 2);
        ctx.fillRect(size / 2, size / 2, size / 2, size / 2);
    }
    if (style.speckle > 0) {
        const image = ctx.getImageData(0, 0, size, size);
        let seed = 12345;
        const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
        for (let i = 0; i < image.data.length; i += 4) {
            const f = 1 + style.speckle * (2 * rand() - 1);
            for (let k = 0; k < 3; k++) image.data[i + k] = Math.min(255, image.data[i + k] * f);
        }
        ctx.putImageData(image, 0, 0);
    }
    const px = size / meters;  // per meter
    ctx.strokeStyle = `rgba(255, 255, 255, ${style.fine})`;
    ctx.lineWidth = px / 128;
    for (let i = 0; grid === "fine" && i <= 10 * meters; i++) {
        const x = i * px / 10;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, size); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, x); ctx.lineTo(size, x); ctx.stroke();
    }
    ctx.strokeStyle = `rgba(255, 255, 255, ${style.major})`;
    ctx.lineWidth = px / 64;
    for (let i = 0; grid !== "none" && i <= meters; i++) {
        const x = i * px;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, size); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, x); ctx.lineTo(size, x); ctx.stroke();
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return {texture, meters};
}
