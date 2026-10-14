import * as THREE from 'three';
import {FontLoader} from 'three/addons/loaders/FontLoader.js';
import {TextGeometry} from 'three/addons/geometries/TextGeometry.js';



function setupLightsSky(scene) {
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.);
    const dirLight = new THREE.DirectionalLight(0xffffff, 2);
    dirLight.position.set(50, 50, 50); // Changed Y from -50 to 50
    dirLight.castShadow = true;
    dirLight.shadow.camera.left = -10;
    dirLight.shadow.camera.right = 100;
    dirLight.shadow.camera.top = 50;
    dirLight.shadow.camera.bottom = -50;
    dirLight.shadow.camera.near = 1;
    dirLight.shadow.camera.far = 200;

    dirLight.shadow.mapSize.width = 16384;
    dirLight.shadow.mapSize.height = 16384;
    dirLight.shadow.bias = -0.0001;

    scene.add(ambientLight);
    scene.add(dirLight);

    // Skybox (simple blue cube)
    const size = 500;
    const skybox = new THREE.Mesh(
        new THREE.BoxGeometry(size, size, size),
        new THREE.MeshBasicMaterial({color: 0x87ceeb, side: THREE.BackSide})
    );
    scene.add(skybox);
}

function setupAxes(scene, currentObjects) {
    const axesHelper = new THREE.Group();
    const axisLength = 1;
    const axisRadius = 0.025;

    function createAxis(color, rotationAxis, rotationAngle) {
        const geometry = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, 12);
        const material = new THREE.MeshBasicMaterial({color});
        const axis = new THREE.Mesh(geometry, material);
        axis.position.y = axisLength / 2;
        if (rotationAxis) {
            axis.rotateOnAxis(rotationAxis, rotationAngle);
            axis.position.set(0, 0, 0);
            axis.translateY(axisLength / 2);
        }
        return axis;
    }

    // xyz
    axesHelper.add(createAxis(0x646efa, new THREE.Vector3(0, 0, 1), -Math.PI / 2));
    axesHelper.add(createAxis(0xef553b));
    axesHelper.add(createAxis(0x00cc96, new THREE.Vector3(1, 0, 0), Math.PI / 2));
    scene.add(axesHelper);
    currentObjects.push(axesHelper);
}

function setupLanes(scene, currentObjects, laneSpacing) {
    // Lane 1
    const laneWidth = 100;     // Length of the lane
    const laneThickness = 0.05; // Thickness of the line (height of the plane)
    const laneSpace = laneSpacing ? laneSpacing : 0.7;

    const laneGeometry1 = new THREE.PlaneGeometry(laneWidth, laneThickness);
    const laneMaterial1 = new THREE.MeshBasicMaterial({color: 0xFFFFFF, side: THREE.DoubleSide});

    const lane1 = new THREE.Mesh(laneGeometry1, laneMaterial1);
    lane1.position.set(50, 0.01, -laneSpace);
    lane1.rotation.x = -Math.PI / 2; // Rotate to lay flat on the XZ plane

    scene.add(lane1);
    currentObjects.push(lane1);

    // Lane 2
    const laneGeometry2 = new THREE.PlaneGeometry(laneWidth, laneThickness);
    const laneMaterial2 = new THREE.MeshBasicMaterial({color: 0xFFFFFF, side: THREE.DoubleSide});

    const lane2 = new THREE.Mesh(laneGeometry2, laneMaterial2);
    lane2.position.set(50, 0.01, laneSpace);
    lane2.rotation.x = -Math.PI / 2;

    scene.add(lane2);
    currentObjects.push(lane2);
}

function setupCurve(scene, currentObjects, numLanes = 8) {
    const innerRadius = 36.5;
    const laneWidth = 1.22;
    const arcSegments = 128;

    // Agent starts at (0,0,0) at the CENTER of lane 1, so the circle
    // center is target_radius away in +Z (matching the Python env).
    const targetRadius = innerRadius + laneWidth / 2;  // 37.11 m
    const cz = targetRadius;

    // Inner and outer edges are laneWidth/2 from the lane center.
    const innerR = targetRadius - laneWidth / 2;  // 36.5 m  (= innerRadius)
    const outerR = targetRadius + laneWidth / 2;  // 37.72 m

    // Arc: agent at angle -PI/2 (-Z from center), CCW through apex (+X) to +PI/2 (+Z).
    const arcStart = -Math.PI / 2;
    const arcEnd = Math.PI / 2;

    const lineMat = new THREE.LineBasicMaterial({color: 0xFFFFFF});

    for (const r of [innerR, outerR]) {
        const curve = new THREE.EllipseCurve(0, cz, r, r, arcStart, arcEnd, false, 0);
        const pts3d = curve.getPoints(arcSegments).map(p => new THREE.Vector3(p.x, 0.01, p.y));

        const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts3d),
            lineMat.clone()
        );
        scene.add(line);
        currentObjects.push(line);
    }
}

function setupNumbers(scene, currentObjects) {
    const loader = new FontLoader();
    loader.load("assets/fonts/helvetiker.json", (font) => {

        const step = 10;          // Distance between numbers
        const max = 100;          // Numbers from 0 → 100
        const size = 1;           // Character size

        for (let x = 0; x <= max; x += step) {
            const textGeo = new TextGeometry(String(x), {
                font: font,
                size: size,
                depth: 0.02,      // Use 'depth' instead of 'height'
                curveSegments: 16,
            });

            textGeo.computeBoundingBox();
            const textMat = new THREE.MeshStandardMaterial({color: 0xffffff});
            const text = new THREE.Mesh(textGeo, textMat);

            text.position.set(x, 0.05, -1.5);

            text.castShadow = true;
            text.receiveShadow = true;

            scene.add(text);
            currentObjects.push(text);
        }
    })
}


export {setupLightsSky, setupAxes, setupLanes, setupCurve, setupNumbers};