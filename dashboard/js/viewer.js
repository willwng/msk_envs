import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {setupLightsSky, setupAxes, setupLanes, setupCurve, setupNumbers} from "./scene.js";
import {loadVisual, loadCollider, loadTarget} from "./loader.js";
import {drawMuscleLine, resetMuscles} from "./muscle.js";

const ViewMode = Object.freeze({
    FULLSCREEN: {key: "FULLSCREEN", label: "Full Screen"},
    PANEL: {key: "PANEL", label: "Panel View"},
    PANEL_FEET: {key: "PANEL_FEET", label: "Panel with Feet"},
});


document.addEventListener("DOMContentLoaded", () => {
    THREE.Object3D.DefaultUp = new THREE.Vector3(0, 1, 0);

    const container = document.getElementById("viewer");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    let width = container.clientWidth;
    let height = container.clientHeight;
    let cameraViewMode = ViewMode.FULLSCREEN;

    function updateCameras(com, footL, footR) {
        // backwards compatibility
        if (footL === undefined) footL = [com[0], com[1] - 1.0, com[2]];
        if (footR === undefined) footR = [com[0], com[1] - 1.0, com[2]];

        if (autoFollow1) {
            cameras[0].position.set(com[0], com[1] + 0.5, com[2] + 2.25);
            controls[0].target.set(com[0], com[1], com[2]);
        }
        if (cameraViewMode === ViewMode.PANEL) {
            if (autoFollow2) {
                cameras[1].position.set(com[0] + 2.25, com[1] + 0.5, com[2]);
                controls[1].target.set(com[0], com[1], com[2]);
            }
            if (autoFollow3) {
                cameras[2].position.set(com[0] - 1.5, com[1] + 1.0, com[2]);
                controls[2].target.set(com[0], com[1], com[2]);
            }
        } else if (cameraViewMode === ViewMode.PANEL_FEET) {
            cameras[1].position.set(footR[0], footR[1], footR[2] + 1.0);
            controls[1].target.set(footR[0], footR[1], footR[2]);
            cameras[3].position.set(footR[0] + 1.0, footR[1], footR[2]);
            controls[3].target.set(footR[0], footR[1], footR[2]);

            cameras[2].position.set(footL[0], footL[1], footL[2] - 1.0);
            controls[2].target.set(footL[0], footL[1], footL[2]);
            cameras[4].position.set(footL[0] + 1.0, footL[1], footL[2]);
            controls[4].target.set(footL[0], footL[1], footL[2]);
        }
    }


    // Frame loading
    let frames = [];
    let currentObjects = [];  // in the scene for the current frame
    let ownedObjects = [];  // created for the current frame only, disposed when it changes
    let staticObjects = [];
    let currentFrame = 0;
    let frameGeneration = 0;  // incremented on every frame change, to drop meshes that finish loading too late
    let setupDone = false;

    function addFrameObject(object, owned = true) {
        scene.add(object);
        currentObjects.push(object);
        if (owned) ownedObjects.push(object);
    }

    function clearFrameObjects() {
        frameGeneration++;
        currentObjects.forEach(obj => scene.remove(obj));
        currentObjects = [];
        // Visuals share geometry/materials with the loader cache and muscles are reused across frames,
        // so only dispose objects that were created for this frame
        const materials = new Set();
        ownedObjects.forEach(obj => obj.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) materials.add(child.material);
        }));
        materials.forEach(material => material.dispose());
        ownedObjects = [];
    }

    function loadFrame(frameIndex) {
        clearFrameObjects();
        const generation = frameGeneration;

        const frame = frames[frameIndex];
        if (!frame) return;

        // Setup scene again
        setupScene();

        // Follow com
        updateCameras(frame.cam_pos, frame.foot_l_pos, frame.foot_r_pos);

        // Write time in text. Add directl to the scene
        const time = frame.time;
        timeElement.textContent = "(t = " + time.toFixed(3) + "s)";

        // Draw objects (visual/collider)
        const drawVisuals = drawVisualsCheckbox.checked;
        const drawBeams = drawBeamsCheckbox.checked;
        const drawSphereColliders = drawSphereCollidersCheckbox.checked;
        const drawCapsuleColliders = drawCapsuleCollidersCheckbox.checked;

        if (drawVisuals) {
            for (const obj of frame.visuals) {
                const color = 0xF1ECE4;
                const mesh_url = `/geometry/${obj.mesh_file.split('/').pop()}`;
                loadVisual(mesh_url, obj.opacity, color, object => {
                    if (generation !== frameGeneration) return;  // the frame changed while the mesh loaded
                    object.scale.set(...obj.scale);
                    object.position.set(...obj.pos);
                    object.quaternion.set(obj.rot[0], obj.rot[1], obj.rot[2], obj.rot[3]);
                    addFrameObject(object, false);
                });
            }
        }
        if (drawBeams && frame.beam_points && frame.beam_points.length > 1) {  // models without beams have none
            const color = 0xF1ECE4;
            const points = [];
            for (const obj of frame.beam_points) {
                points.push(new THREE.Vector3(...obj.pos));
                const sphereGeometry = new THREE.SphereGeometry(0.015, 8, 8);
                const sphereMaterial = new THREE.MeshStandardMaterial({color: color});
                const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
                sphere.position.set(...obj.pos);
                addFrameObject(sphere);
            }
            const curve = new THREE.CatmullRomCurve3(points);
            const geometry = new THREE.TubeGeometry(curve, 100, 0.01, 8, false);
            const material = new THREE.MeshStandardMaterial({color: color});
            addFrameObject(new THREE.Mesh(geometry, material));
        }

        for (const obj of frame.colliders) {
            let color = 0x87CEEB;
            if (obj.contact_force) {
                let inContact = obj.contact_force > 0.0;
                color = inContact ? 0x6AEB9D : 0x87CEEB;
            }
            const drawAnyway = obj.name.includes("hurdle");
            loadCollider(drawSphereColliders, drawCapsuleColliders || drawAnyway,
                obj.geom_type, obj.scale, obj.rot, color, object => {
                    object.position.set(...obj.pos);
                    object.castShadow = true;
                    object.receiveShadow = true;
                    addFrameObject(object);
                });
        }

        // Targets
        if (frame.targets) {
            for (const obj of frame.targets) {
                const color = obj.active ? 0x6AEB9D : 0x87CEEB;
                loadTarget(obj.radius, obj.rot, color, object => {
                    object.position.set(...obj.pos);
                    object.castShadow = true;
                    object.receiveShadow = true;
                    addFrameObject(object);
                });
            }
        }

        // Draw muscles
        const drawMuscles = drawMusclesCheckbox.checked;
        if (drawMuscles && frame.muscles) {
            for (const muscle of frame.muscles) {
                drawMuscleLine(muscle.name, muscle, object => addFrameObject(object, false));
            }
        }

        for (const arrow of frame.arrows) {
            const origin = new THREE.Vector3(arrow.start[0], arrow.start[1], arrow.start[2]);
            const dir = new THREE.Vector3(arrow.direction[0], arrow.direction[1], arrow.direction[2]);
            const length = dir.length();
            if (length < 1e-12) continue;

            dir.normalize();

            // Create cylinder with unit length and thickness based on original length
            const radius = 0.001 * Math.sqrt(length);
            const cylinderLength = 0.2;
            const geometry = new THREE.CylinderGeometry(radius, radius, cylinderLength, 8);
            const material = new THREE.MeshStandardMaterial({color: 0x00ff00});
            const cylinder = new THREE.Mesh(geometry, material);

            // Position cylinder at the midpoint
            const midpoint = origin.clone().add(dir.clone().multiplyScalar(cylinderLength / 2));
            cylinder.position.copy(midpoint);
            cylinder.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
            addFrameObject(cylinder);

            // Create arrowhead (cone)
            const coneHeight = radius * 4;
            const coneRadius = radius * 2;
            const coneGeometry = new THREE.ConeGeometry(coneRadius, coneHeight, 8);
            const cone = new THREE.Mesh(coneGeometry, material);

            // Position cone at the end of the cylinder
            const conePosition = origin.clone().add(dir.clone().multiplyScalar(cylinderLength + coneHeight / 2));
            cone.position.copy(conePosition);
            cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
            addFrameObject(cone);
        }

        currentFrame = frameIndex;
    }

    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Panel cameras
    let cameras = [];
    let controls = [];
    for (let i = 0; i < 5; i++) {
        const cam = new THREE.PerspectiveCamera(75, (width / 3) / height, 0.1, 1000);
        const control = new OrbitControls(cam, renderer.domElement);
        cam.up.set(0, 1, 0);

        cameras.push(cam);
        controls.push(control);
    }
    cameras[0].aspect = width / height;
    cameras[0].updateProjectionMatrix();

    const drawVisualsCheckbox = document.getElementById("drawVisuals");
    const drawBeamsCheckbox = document.getElementById("drawBeams");
    const drawSphereCollidersCheckbox = document.getElementById("drawSphereColliders");
    const drawCapsuleCollidersCheckbox = document.getElementById("drawCapsuleColliders");
    const drawMusclesCheckbox = document.getElementById("drawMuscles");
    const timeElement = document.getElementById("timeValue");

    const fullScreen = document.getElementById("fullScreenButton");

    // Auto follow camera
    const resetView1 = document.getElementById("resetButton1");
    const resetView2 = document.getElementById("resetButton2");
    const resetView3 = document.getElementById("resetButton3");
    resetView1.style.display = "none";
    resetView2.style.display = "none";
    resetView3.style.display = "none";
    let autoFollow1 = true;
    let autoFollow2 = true;
    let autoFollow3 = true;

    function onMouseMove(event) {
        if (cameraViewMode === ViewMode.FULLSCREEN) {
            for (let i = 0; i < controls.length; i++) {
                controls[i].enabled = (i === 0);
            }
            return;
        }
        const x = event.clientX;
        if (x < container.getBoundingClientRect().left + width / 3) {
            for (let i = 0; i < controls.length; i++) {
                controls[i].enabled = (i === 0);
            }
        } else if (x < container.getBoundingClientRect().left + 2 * width / 3) {
            for (let i = 0; i < controls.length; i++) {
                controls[i].enabled = (i === 1);
            }
        } else {
            for (let i = 0; i < controls.length; i++) {
                controls[i].enabled = (i === 2);
            }
        }
    }

    // Event listeners
    document.addEventListener("mousemove", onMouseMove, false);
    window.addEventListener("sliderChanged", function (event) {
        const frameIndex = parseInt(event.detail);
        loadFrame(frameIndex);
    });

    window.addEventListener("loadTrajectory", function (event) {
        frames = event.detail;
        setupDone = false;
        staticObjects.forEach(obj => scene.remove(obj));
        staticObjects = [];
        loadFrame(0);
    });

    window.addEventListener("resetViewer", function (event) {
        frames = [];
        clearFrameObjects();
        staticObjects.forEach(obj => scene.remove(obj));
        staticObjects = [];
        currentFrame = 0;
        resetMuscles();
    });

    fullScreen.addEventListener('click', () => {
        let viewModes = Object.values(ViewMode);

        let currentIndex = viewModes.indexOf(cameraViewMode);
        let nextIndex = (currentIndex + 1) % viewModes.length;
        let nextNextIndex = (currentIndex + 2) % viewModes.length;

        cameraViewMode = viewModes[nextIndex];
        let nextMode = viewModes[nextNextIndex];
        fullScreen.textContent = String(nextMode.label);
        updateCameraAspects();
        loadFrame(currentFrame);
    });

    function updateCameraAspects() {
        if (cameraViewMode === ViewMode.FULLSCREEN) {
            cameras[0].aspect = width / height;
            cameras[0].updateProjectionMatrix();
        } else if (cameraViewMode === ViewMode.PANEL) {
            for (let i = 0; i < cameras.length; i++) {
                cameras[i].aspect = (width / 3) / height;
                cameras[i].updateProjectionMatrix();
            }
        } else if (cameraViewMode === ViewMode.PANEL_FEET) {
            cameras[0].aspect = (width / 3) / height;
            cameras[0].updateProjectionMatrix();
            // Split the other 2/3 among the remaining four cameras
            for (let i = 1; i < cameras.length; i++) {
                cameras[i].aspect = (width / 3) / (height / 2);
                cameras[i].updateProjectionMatrix();
            }
        }
    }

    window.addEventListener("resize", () => {
        width = container.clientWidth;
        height = container.clientHeight;
        renderer.setSize(width, height);
        updateCameraAspects();
    });

    controls[0].addEventListener("start", () => {
        resetView1.style.display = "block";
        autoFollow1 = false;
    });
    resetView1.addEventListener('click', () => {
        resetView1.style.display = "none";
        autoFollow1 = true;
        loadFrame(currentFrame);
    });
    controls[1].addEventListener("start", () => {
        resetView2.style.display = "block";
        autoFollow2 = false;
    });
    resetView2.addEventListener('click', () => {
        resetView2.style.display = "none";
        autoFollow2 = true;
        loadFrame(currentFrame);
    });
    controls[2].addEventListener("start", () => {
        resetView3.style.display = "block";
        autoFollow3 = false;
    });
    resetView3.addEventListener('click', () => {
        resetView3.style.display = "none";
        autoFollow3 = true;
        loadFrame(currentFrame);
    });


    drawVisualsCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });
    drawBeamsCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });
    drawSphereCollidersCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });
    drawCapsuleCollidersCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });
    drawMusclesCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });

    function setupScene() {
        if (setupDone) return;
        setupDone = true;

        if (frames.length === 0) return;
        if(frames[0].scene_settings) {
            const settings = frames[0].scene_settings;
            if (settings.meter_markers) setupNumbers(scene, staticObjects);
            if (settings.lanes) setupLanes(scene, staticObjects, settings.lane_width);
            if (settings.curve) setupCurve(scene, staticObjects);
            if (settings.axes) setupAxes(scene, staticObjects);
        } else {
            setupNumbers(scene, staticObjects);
            setupLanes(scene, staticObjects);
            setupAxes(scene, staticObjects);
            setupCurve(scene, staticObjects);
        }
    }

    function animate() {
        for (let i = 0; i < controls.length; i++) {
            controls[i].update();
        }

        const borderWidth = 2;
        const viewWidth = (width - borderWidth) / 3;
        const viewHeight = height;

        if (cameraViewMode === ViewMode.FULLSCREEN) {
            renderer.setViewport(0, 0, width, height);
            renderer.setScissor(0, 0, width, height);
            renderer.render(scene, cameras[0]);
            return;
        } else if (cameraViewMode === ViewMode.PANEL) {
            renderer.setScissorTest(true);
            for (let i = 0; i < 3; i++) {
                renderer.setViewport(i * (viewWidth + borderWidth), 0, viewWidth, viewHeight);
                renderer.setScissor(i * (viewWidth + borderWidth), 0, viewWidth, viewHeight);
                renderer.render(scene, cameras[i]);
            }
            renderer.setScissorTest(false);
        } else if (cameraViewMode === ViewMode.PANEL_FEET) {
            renderer.setScissorTest(true);

            // Main camera (left third)
            renderer.setViewport(0, 0, viewWidth, viewHeight);
            renderer.setScissor(0, 0, viewWidth, viewHeight);
            renderer.render(scene, cameras[0]);

            // Right cameras (right two thirds, split each third vertically among two cameras each)
            const halfViewHeight = viewHeight / 2;
            for (let i = 1; i < cameras.length; i++) {
                const col = (i - 1) % 2;
                const row = 1 - Math.floor((i - 1) / 2);
                const x = viewWidth + borderWidth + col * (viewWidth + borderWidth);
                const y = row * (halfViewHeight + borderWidth);

                renderer.setViewport(x, y, viewWidth, halfViewHeight);
                renderer.setScissor(x, y, viewWidth, halfViewHeight);
                renderer.render(scene, cameras[i]);
            }
            renderer.setScissorTest(false);

        }
    }

    setupLightsSky(scene);
    renderer.setAnimationLoop(animate);
});