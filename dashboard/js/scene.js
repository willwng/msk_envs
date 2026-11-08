import * as THREE from "three";
import {FontLoader} from "three/addons/loaders/FontLoader.js";
import {TextGeometry} from "three/addons/geometries/TextGeometry.js";

// Sky dome with a vertical gradient, and fog that fades the ground into it
export function setupSky(scene) {
    const top = new THREE.Color(0x5aaeee), horizon = new THREE.Color(0xc4e6fb);  // bright, clear sky
    const geometry = new THREE.SphereGeometry(800, 32, 16);
    const material = new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        fog: false,
        uniforms: {top: {value: top}, horizon: {value: horizon}},
        vertexShader: `varying vec3 vDir;
            void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
        fragmentShader: `uniform vec3 top; uniform vec3 horizon; varying vec3 vDir;
            void main() { float h = clamp(vDir.y, 0.0, 1.0); gl_FragColor = vec4(mix(horizon, top, pow(h, 0.6)), 1.0);
            #include <colorspace_fragment>
            }`,
    });
    const sky = new THREE.Mesh(geometry, material);
    sky.renderOrder = -1;
    sky.frustumCulled = false;
    scene.add(sky);
    scene.fog = new THREE.Fog(horizon, 30, 140);
    return sky;
}

// Hemisphere fill and a sun whose shadow follows the model (a small shadow frustum keeps shadows crisp)
export function setupLights(scene) {
    const hemi = new THREE.HemisphereLight(0xe8f0ff, 0x7a6a58, 1.1);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xfff4e5, 2.6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    const s = 3.0;
    Object.assign(sun.shadow.camera, {left: -s, right: s, top: s, bottom: -s, near: 0.5, far: 30});
    sun.shadow.bias = -0.0002;
    sun.shadow.normalBias = 0.02;
    sun.shadow.radius = 3;
    sun.userData.offset = new THREE.Vector3(4, 9, 5);
    scene.add(sun, sun.target);
    const rim = new THREE.DirectionalLight(0xdfe8ff, 0.7);
    rim.position.set(-5, 4, -6);
    scene.add(rim);
    return sun;
}

export function followShadow(sun, target) {
    sun.target.position.copy(target);
    sun.position.copy(target).add(sun.userData.offset);
}

export function setupAxes(group) {
    const axisLength = 1, axisRadius = 0.02;
    const make = (color, rotate) => {
        const mesh = new THREE.Mesh(new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, 12),
            new THREE.MeshBasicMaterial({color}));
        mesh.geometry.translate(0, axisLength / 2, 0);
        rotate(mesh);
        return mesh;
    };
    group.add(make(0xef553b, m => m.rotation.z = -Math.PI / 2));  // x
    group.add(make(0x00cc96, () => {}));  // y
    group.add(make(0x646efa, m => m.rotation.x = Math.PI / 2));  // z
}

export function setupLanes(group, laneSpacing) {
    const space = laneSpacing ? laneSpacing : 0.7;
    for (const z of [-space, space]) group.add(...groundLine([[-5, z], [105, z]]));
}

const LINE_STRIP = new THREE.MeshBasicMaterial({color: 0xffffff, side: THREE.DoubleSide});
const LINE_PIXEL = new THREE.LineBasicMaterial({color: 0xffffff});

// A white line on the ground through [points] (x, z): a strip [width] wide, plus a 1 px line along its middle so it
// stays visible (unbroken) far away, where the strip is thinner than a pixel
function groundLine(points, width = 0.05, y = 0.002) {
    const positions = [], index = [];
    for (let i = 0; i < points.length; i++) {
        const [x, z] = points[i];
        const [xa, za] = points[Math.max(0, i - 1)], [xb, zb] = points[Math.min(points.length - 1, i + 1)];
        const len = Math.hypot(xb - xa, zb - za) || 1;
        const nx = -(zb - za) / len * width / 2, nz = (xb - xa) / len * width / 2;
        positions.push(x - nx, y, z - nz, x + nx, y, z + nz);
        if (i > 0) index.push(2 * i - 2, 2 * i - 1, 2 * i, 2 * i - 1, 2 * i + 1, 2 * i);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(index);
    geometry.computeVertexNormals();
    const strip = new THREE.Mesh(geometry, LINE_STRIP);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(([x, z]) => new THREE.Vector3(x, y, z))),
        LINE_PIXEL);
    return [strip, line];
}

export function setupCurve(group) {
    // The bend of msk_envs/envs/curve.py: the agent starts at (0, 0, 0), the center of lane 1, facing +x, and the
    // bend's center is at z = +(inner radius + half a lane). The lane's lines run from the start (angle -90 deg from
    // the center) through the apex at +x to the end of the bend (+90 deg)
    const innerRadius = 36.5, laneWidth = 1.22;
    const cz = innerRadius + laneWidth / 2;
    const segments = 256;
    for (const r of [innerRadius, innerRadius + laneWidth]) {
        const points = [];
        for (let i = 0; i <= segments; i++) {
            const a = -Math.PI / 2 + Math.PI * i / segments;
            points.push([r * Math.cos(a), cz + r * Math.sin(a)]);
        }
        group.add(...groundLine(points));
    }
}

export function setupNumbers(group) {
    new FontLoader().load("assets/fonts/helvetiker.json", font => {
        const material = new THREE.MeshStandardMaterial({color: 0xffffff, roughness: 0.6});
        for (let x = 0; x <= 100; x += 10) {
            // Standing behind the far lane, facing the default (side) camera
            const geometry = new TextGeometry(String(x), {font, size: 1, depth: 0.02, curveSegments: 8});
            geometry.computeBoundingBox();
            geometry.translate(0, -geometry.boundingBox.min.y + 0.002, 0);  // round digits dip below the baseline
            const text = new THREE.Mesh(geometry, material);
            text.position.set(x, 0.0, -1.5);
            text.castShadow = true;
            group.add(text);
        }
    });
}
