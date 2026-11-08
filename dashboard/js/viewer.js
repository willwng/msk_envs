// Trajectory viewer: builds the scene once per trajectory and updates it in place every frame. Playback runs here, in
// the browser; dashboard.py only picks the file (window.mskViewer.load). Animation format: msk_envs/trajectory_io.py
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {RoomEnvironment} from "three/addons/environments/RoomEnvironment.js";
import {setupSky, setupLights, followShadow, setupAxes, setupLanes, setupCurve, setupNumbers} from "./scene.js";
import {loadMeshGeometry, colliderGeometry, wrapSurfaceGeometry, groundTexture, GROUND_STYLES, DEFAULT_GROUND,
    GROUND_GRIDS, DEFAULT_GRID} from "./loader.js";
import {MuscleTubes, MuscleLines, MuscleCapsules} from "./muscle.js";
import {colorGradientCss, muscleValues} from "./muscle_style.js";

const $ = id => document.getElementById(id);

// Camera layouts: viewports as [x, y, w, h] fractions of the canvas (y from the bottom), and what each camera follows
const LAYOUTS = {
    single: [{rect: [0, 0, 1, 1], follow: "com", offset: [0.0, 0.5, 2.6]}],
    panel: [
        {rect: [0, 0, 1 / 3, 1], follow: "com", offset: [0.0, 0.5, 2.6]},
        {rect: [1 / 3, 0, 1 / 3, 1], follow: "com", offset: [2.6, 0.5, 0.0]},
        {rect: [2 / 3, 0, 1 / 3, 1], follow: "com", offset: [-1.8, 1.0, 1.0]},
    ],
    feet: [
        {rect: [0, 0, 1 / 3, 1], follow: "com", offset: [0.0, 0.5, 2.6]},
        {rect: [1 / 3, 0.5, 1 / 3, 0.5], follow: "footR", offset: [0.0, 0.1, 0.9]},
        {rect: [2 / 3, 0.5, 1 / 3, 0.5], follow: "footR", offset: [0.9, 0.1, 0.0]},
        {rect: [1 / 3, 0, 1 / 3, 0.5], follow: "footL", offset: [0.0, 0.1, -0.9]},
        {rect: [2 / 3, 0, 1 / 3, 0.5], follow: "footL", offset: [0.9, 0.1, 0.0]},
    ],
};

class Viewer {
    constructor(container) {
        this.container = container;
        this.renderer = new THREE.WebGLRenderer({antialias: true});
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        container.appendChild(this.renderer.domElement);

        this.scene = new THREE.Scene();
        const pmrem = new THREE.PMREMGenerator(this.renderer);
        this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        this.scene.environmentIntensity = 0.35;
        setupSky(this.scene);
        this.sun = setupLights(this.scene);
        this.ground = this.makeGround();
        this.scene.add(this.ground);

        this.materials = {
            // Matte, without environment reflections: as bright as the old viewer's bones (0xf1ece4 under one ambient
            // and one directional light, without tone mapping), under this scene's brighter lighting
            bone: new THREE.MeshStandardMaterial({color: 0xc9c2b5, roughness: 1.0, metalness: 0.0, envMapIntensity: 0}),
            collider: new THREE.MeshStandardMaterial({color: 0x87ceeb, roughness: 0.5, transparent: true, opacity: 0.55,
                depthWrite: false}),
            contact: new THREE.MeshStandardMaterial({color: 0x45e08a, roughness: 0.5, transparent: true, opacity: 0.7,
                depthWrite: false}),
            wrap: new THREE.MeshStandardMaterial({color: 0x4fb0ff, roughness: 0.3, transparent: true, opacity: 0.28,
                depthWrite: false, side: THREE.DoubleSide}),
            beam: new THREE.MeshStandardMaterial({color: 0xeee6d6, roughness: 0.7}),
            target: new THREE.MeshStandardMaterial({color: 0x87ceeb, transparent: true, opacity: 0.8}),
            targetActive: new THREE.MeshStandardMaterial({color: 0x6aeb9d, transparent: true, opacity: 0.8}),
            force: new THREE.MeshStandardMaterial({color: 0x00ff00}),
        };
        this.unit = {
            cylinder: new THREE.CylinderGeometry(1, 1, 1, 12).translate(0, 0.5, 0),
            cone: new THREE.ConeGeometry(1, 1, 16).translate(0, 0.5, 0),
            sphere: new THREE.SphereGeometry(1, 20, 14),
        };
        this.colorMaterials = new Map();

        this.anim = null;
        this.frame = 0;
        this.playing = false;
        this.playTime = 0;
        this.lastTick = null;
        this.cameras = [];
        this.setupUi();
        let ground = DEFAULT_GROUND, grid = DEFAULT_GRID;
        try {
            ground = localStorage.getItem("mskGroundStyle") || ground;
            grid = localStorage.getItem("mskGroundGrid") || grid;
        } catch (e) { /* storage unavailable */ }
        this.setGround(ground, grid);
        this.setLayout("single");
        this.resize();
        window.addEventListener("resize", () => this.resize());
        new ResizeObserver(() => this.resize()).observe(container);
        this.setupPicking();
        this.renderer.setAnimationLoop(t => this.tick(t));
    }

    // --- Static scene ---
    makeGround() {
        const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400),
            new THREE.MeshStandardMaterial({roughness: 0.9, metalness: 0.0}));
        ground.geometry.rotateX(-Math.PI / 2);
        ground.geometry.translate(50, 0, 0);
        ground.receiveShadow = true;
        return ground;
    }

    // The ground's look (loader.js GROUND_STYLES) and grid lines (GROUND_GRIDS), remembered in this browser
    setGround(name = $("groundStyle").value, grid = $("groundGrid").value) {
        if (!GROUND_STYLES[name]) name = DEFAULT_GROUND;
        if (!GROUND_GRIDS[grid]) grid = DEFAULT_GRID;
        const material = this.ground.material;
        if (material.map) material.map.dispose();
        const {texture, meters} = groundTexture(this.renderer, name, grid);
        material.map = texture;
        material.map.repeat.set(400 / meters, 400 / meters);  // the ground is 400 m across
        material.needsUpdate = true;
        $("groundStyle").value = name;
        $("groundGrid").value = grid;
        try {
            localStorage.setItem("mskGroundStyle", name);
            localStorage.setItem("mskGroundGrid", grid);
        } catch (e) { /* storage unavailable */ }
    }

    clearTrajectory() {
        if (this.world) {
            this.scene.remove(this.world);
            this.world.traverse(obj => {
                if (obj.isMesh && obj.geometry && !Object.values(this.unit).includes(obj.geometry)
                    && !obj.userData.sharedGeometry) obj.geometry.dispose();
            });
        }
        if (this.muscles) this.muscles.dispose();
        this.world = null;
        this.muscles = null;
        this.anim = null;
    }

    build(anim) {
        this.clearTrajectory();
        this.anim = anim;
        const world = new THREE.Group();
        this.world = world;

        // Track markings
        const env = new THREE.Group();
        const settings = anim.scene_settings || {};
        if (settings.meter_markers) setupNumbers(env);
        if (settings.lanes) setupLanes(env, settings.lane_width);
        if (settings.curve) setupCurve(env);
        if (settings.axes) setupAxes(env);
        world.add(env);

        // Bones
        this.visualObjects = anim.visuals.map(v => {
            const obj = new THREE.Group();
            obj.scale.set(...v.scale);
            const url = `/geometry/${v.mesh_file.split("/").pop()}`;
            loadMeshGeometry(url).then(geometry => {
                const mesh = new THREE.Mesh(geometry, this.materials.bone);
                mesh.userData.sharedGeometry = true;
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                obj.add(mesh);
            }).catch(err => console.warn("Could not load", url, err));
            return obj;
        });
        this.visualGroup = new THREE.Group();
        this.visualGroup.add(...this.visualObjects);
        world.add(this.visualGroup);

        // Colliders: the ground plane moves the ground; the rest are drawn when enabled
        this.colliderObjects = anim.colliders.map(c => {
            if (c.geom_type === 0) return null;
            const geometry = colliderGeometry(c.geom_type, c.size);
            if (!geometry) return null;
            const mesh = new THREE.Mesh(geometry, this.materials.collider);
            mesh.castShadow = true;
            mesh.userData.alwaysShow = c.name.includes("hurdle");
            return mesh;
        });
        this.colliderGroup = new THREE.Group();
        this.colliderObjects.forEach(o => o && this.colliderGroup.add(o));
        world.add(this.colliderGroup);
        this.groundIndex = anim.colliders.findIndex(c => c.geom_type === 0);  // the plane moves the ground

        // Wrapping surfaces
        this.wrapObjects = (anim.wrap_surfaces || []).map(s => {
            const geometry = wrapSurfaceGeometry(s);
            return geometry ? new THREE.Mesh(geometry, this.materials.wrap) : null;
        });
        this.wrapGroup = new THREE.Group();
        this.wrapObjects.forEach(o => o && this.wrapGroup.add(o));
        world.add(this.wrapGroup);
        $("wrapCount").textContent = this.wrapObjects.length ? `(${this.wrapObjects.length})` : "(none)";

        // Muscles
        this.buildMuscles(world);
        this.muscleVisible = null;
        $("muscleFilter").value = "";

        // Per-frame overlays (arrows, targets, beams)
        this.overlay = new THREE.Group();
        world.add(this.overlay);

        this.scene.add(world);
        this.applyVisibility();
        $("timeline").max = String(Math.max(0, anim.frames.length - 1));
        this.playTime = 0;
        this.setFrame(0, true);
        this.resetCameras();
    }

    // --- Per frame ---
    setFrame(index, force = false) {
        if (!this.anim || !this.anim.frames.length) return;
        index = Math.max(0, Math.min(this.anim.frames.length - 1, index));
        if (index === this.frame && !force) return;
        this.frame = index;
        const frame = this.anim.frames[index];

        const place = (obj, X) => {
            obj.position.set(X[0], X[1], X[2]);
            obj.quaternion.set(X[3], X[4], X[5], X[6]);
        };
        frame.visuals.forEach((X, i) => place(this.visualObjects[i], X));
        frame.colliders.forEach((X, i) => {
            const obj = this.colliderObjects[i];
            if (i === this.groundIndex) place(this.ground, X);
            if (!obj) return;
            place(obj, X);
            obj.material = X[7] > 0 ? this.materials.contact : this.materials.collider;
        });
        (frame.wrap_surfaces || []).forEach((X, i) => this.wrapObjects[i] && place(this.wrapObjects[i], X));
        this.updateMuscles();
        this.pickPending = true;
        this.updateOverlay(frame);

        // Follow
        this.targets = {
            com: new THREE.Vector3(...frame.cam_pos),
            footL: new THREE.Vector3(...frame.foot_l_pos),
            footR: new THREE.Vector3(...frame.foot_r_pos),
        };
        this.followTargets();
        followShadow(this.sun, this.targets.com);

        $("timeline").value = String(index);
        $("timeValue").textContent = `t = ${frame.time.toFixed(3)} s`;
        $("frameValue").textContent = `${index} / ${this.anim.frames.length - 1}`;
    }

    muscleValues(frame) {
        return muscleValues(this.anim.muscles, frame.muscles, $("muscleColor").value);
    }

    // Tubes or lines (the "style" menu), replacing the current ones
    buildMuscles(world = this.world) {
        if (this.muscles) {
            world.remove(this.muscles.object);
            this.muscles.dispose();
            this.muscles = null;
        }
        if (!this.anim.muscles.length) return;
        const Style = {lines: MuscleLines, capsules: MuscleCapsules}[$("muscleStyle").value] || MuscleTubes;
        this.muscles = new Style(this.anim.muscles);
        this.muscles.object.visible = $("drawMuscles").checked;
        world.add(this.muscles.object);
    }

    updateMuscles() {
        if (!this.muscles || !this.anim) return;
        const frame = this.anim.frames[this.frame];
        this.muscles.update(frame, this.muscleValues(frame), parseFloat($("muscleScale").value), this.muscleVisible,
            $("drawTendons").checked);
    }

    material(color) {
        if (!this.colorMaterials.has(color)) this.colorMaterials.set(color, new THREE.MeshStandardMaterial({color}));
        return this.colorMaterials.get(color);
    }

    arrow(start, dir, material, shaftRadius, headLength, headRadius) {
        const group = new THREE.Group();
        const length = dir.length();
        group.position.copy(start);
        group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
        const shaftLength = Math.max(0, length - headLength);
        if (shaftLength > 0) {
            const shaft = new THREE.Mesh(this.unit.cylinder, material);
            shaft.scale.set(shaftRadius, shaftLength, shaftRadius);
            group.add(shaft);
        }
        const head = new THREE.Mesh(this.unit.cone, material);
        head.scale.set(headRadius, Math.min(headLength, length), headRadius);
        head.position.y = shaftLength;
        group.add(head);
        group.traverse(o => o.castShadow = true);
        return group;
    }

    updateOverlay(frame) {
        this.overlay.traverse(obj => {
            if (obj.isMesh && !Object.values(this.unit).includes(obj.geometry)) obj.geometry.dispose();
        });
        this.overlay.clear();

        // Task arrows (e.g. commanded vs actual velocity), length in meters
        const legend = new Map();
        if ($("drawGuides").checked) {
            for (const guide of frame.guides || []) {
                const material = this.material(guide.color);
                const dir = new THREE.Vector3(...guide.direction);
                legend.set(guide.color, guide.label);
                if (dir.length() < 0.12) {  // e.g. a standing command: a dot
                    const dot = new THREE.Mesh(this.unit.sphere, material);
                    dot.scale.setScalar(0.04);
                    dot.position.set(...guide.start);
                    this.overlay.add(dot);
                } else {
                    this.overlay.add(this.arrow(new THREE.Vector3(...guide.start), dir, material, 0.02, 0.12, 0.05));
                }
            }
        }
        const legendHtml = [...legend].map(([color, label]) =>
            `<span style="color: #${color.toString(16).padStart(6, "0")}">&#9632; ${label}</span>`).join(" ");
        if (legendHtml !== this.legendHtml) $("guideLegend").innerHTML = this.legendHtml = legendHtml;

        // External forces: fixed length, thickness ~ sqrt(force)
        if ($("drawForces").checked) {
            for (const a of frame.arrows || []) {
                const dir = new THREE.Vector3(...a.direction);
                const force = dir.length();
                if (force < 1e-9) continue;
                const radius = 0.001 * Math.sqrt(force);
                dir.multiplyScalar((0.2 + 4 * radius) / force);
                this.overlay.add(this.arrow(new THREE.Vector3(...a.start), dir, this.materials.force, radius, 4 * radius,
                    2 * radius));
            }
        }

        for (const t of frame.targets || []) {
            const sphere = new THREE.Mesh(this.unit.sphere, t.active ? this.materials.targetActive : this.materials.target);
            sphere.scale.setScalar(t.radius);
            sphere.position.set(...t.pos);
            sphere.castShadow = true;
            this.overlay.add(sphere);
        }

        if ($("drawBeams").checked && frame.beam_points && frame.beam_points.length > 1) {
            const points = frame.beam_points.map(p => new THREE.Vector3(...(p.pos || p)));
            const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 100, 0.01, 8),
                this.materials.beam);
            tube.castShadow = true;
            this.overlay.add(tube);
        }
    }

    // --- Cameras ---
    setLayout(name) {
        const old = this.cameras;
        this.cameras = LAYOUTS[name].map((spec, i) => {
            const prev = old[i];
            const camera = new THREE.PerspectiveCamera(prev ? prev.camera.fov : 50, 1, 0.05, 1000);
            const controls = new OrbitControls(camera, this.renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.15;
            controls.addEventListener("start", () => this.dragging = true);
            controls.addEventListener("change", () => this.pickPending = true);
            controls.addEventListener("end", () => this.dragging = false);
            if (prev && prev.spec.follow === spec.follow) {
                camera.position.copy(prev.camera.position);
                controls.target.copy(prev.controls.target);
            }
            const view = {spec, camera, controls, last: null, placed: !!prev && prev.spec.follow === spec.follow};
            return view;
        });
        old.forEach(v => v.controls.dispose());
        this.resize();
        this.followTargets();
    }

    resetCameras() {
        this.cameras.forEach(v => v.placed = false);
        this.followTargets();
    }

    // Each camera keeps its offset from what it follows (orbiting changes the offset), unless follow is off
    followTargets() {
        if (!this.targets) return;
        const follow = $("follow").checked;
        for (const view of this.cameras) {
            const target = this.targets[view.spec.follow] || this.targets.com;
            if (!view.placed) {
                view.controls.target.copy(target);
                view.camera.position.copy(target).add(new THREE.Vector3(...view.spec.offset));
                view.placed = true;
            } else if (follow && view.last) {
                const delta = target.clone().sub(view.last);
                view.camera.position.add(delta);
                view.controls.target.add(delta);
            }
            view.last = target.clone();
            view.controls.update();
        }
    }

    resize() {
        const w = this.container.clientWidth, h = this.container.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h);
        for (const view of this.cameras) {
            const [, , rw, rh] = view.spec.rect;
            view.camera.aspect = (rw * w) / (rh * h);
            view.camera.updateProjectionMatrix();
        }
    }

    viewAt(clientX, clientY) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        const x = (clientX - rect.left) / rect.width, y = 1 - (clientY - rect.top) / rect.height;
        return this.cameras.find(v => {
            const [rx, ry, rw, rh] = v.spec.rect;
            return x >= rx && x <= rx + rw && y >= ry && y <= ry + rh;
        });
    }

    // --- Muscle picking (hover for the name and value) ---
    setupPicking() {
        const canvas = this.renderer.domElement;
        const tooltip = $("muscleTooltip");
        const raycaster = new THREE.Raycaster();
        canvas.addEventListener("pointermove", event => {
            // Only the camera under the pointer responds to the mouse
            const active = this.viewAt(event.clientX, event.clientY);
            if (!this.dragging) this.cameras.forEach(v => v.controls.enabled = v === active);
            this.pointer = {x: event.clientX, y: event.clientY, view: active};
            this.pickPending = true;
        });
        canvas.addEventListener("pointerleave", () => { this.pointer = null; tooltip.style.display = "none"; });
        const ndc = new THREE.Vector2(), viewport = new THREE.Vector2();
        // Run by tick when the pointer, the frame or a camera has changed (pickPending)
        this.pick = () => {
            const p = this.pointer;
            if (!p || !p.view || !this.muscles || !this.muscles.object.visible || this.dragging) {
                tooltip.style.display = "none";
                return;
            }
            const rect = canvas.getBoundingClientRect();
            const [rx, ry, rw, rh] = p.view.spec.rect;
            ndc.set((((p.x - rect.left) / rect.width - rx) / rw) * 2 - 1,
                (((1 - (p.y - rect.top) / rect.height) - ry) / rh) * 2 - 1);
            raycaster.setFromCamera(ndc, p.view.camera);
            const i = this.muscles.pick(raycaster, viewport.set(rw * rect.width, rh * rect.height));
            if (i < 0) { tooltip.style.display = "none"; return; }
            const frame = this.anim.frames[this.frame];
            const value = this.muscleValues(frame)[i];
            tooltip.textContent = `${this.anim.muscles[i].name}  ${$("muscleColor").selectedOptions[0].text} ${value.toFixed(2)}`;
            tooltip.style.left = `${p.x - rect.left + 14}px`;
            tooltip.style.top = `${p.y - rect.top + 10}px`;
            tooltip.style.display = "block";
        };
    }

    // --- UI ---
    setupUi() {
        const redraw = () => this.setFrame(this.frame, true);
        $("playButton").addEventListener("click", () => this.togglePlay());
        $("timeline").addEventListener("input", e => { this.pause(); this.setFrame(parseInt(e.target.value)); });
        $("stepBack").addEventListener("click", () => { this.pause(); this.setFrame(this.frame - 1); });
        $("stepForward").addEventListener("click", () => { this.pause(); this.setFrame(this.frame + 1); });
        for (const id of ["drawVisuals", "drawMuscles", "drawWrap", "drawColliders", "drawShadows", "boneOpacity"]) {
            $(id).addEventListener("input", () => this.applyVisibility());
        }
        for (const id of ["drawGuides", "drawForces", "drawBeams", "muscleColor", "muscleScale", "drawTendons"]) {
            $(id).addEventListener("input", redraw);
        }
        $("muscleFilter").addEventListener("input", () => { this.applyMuscleFilter(); redraw(); });
        $("muscleStyle").addEventListener("input", () => { if (this.anim) { this.buildMuscles(); redraw(); } });
        $("layout").addEventListener("input", e => this.setLayout(e.target.value));
        $("resetCamera").addEventListener("click", () => this.resetCameras());
        $("groundStyle").innerHTML = Object.entries(GROUND_STYLES)
            .map(([name, s]) => `<option value="${name}">${s.label}</option>`).join("");
        $("groundGrid").innerHTML = Object.entries(GROUND_GRIDS)
            .map(([name, label]) => `<option value="${name}">${label}</option>`).join("");
        for (const id of ["groundStyle", "groundGrid"]) $(id).addEventListener("input", () => this.setGround());
        $("colorBar").style.background = colorGradientCss();
        document.addEventListener("keydown", e => {
            if (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
            if (e.code === "Space") { e.preventDefault(); this.togglePlay(); }
            else if (e.code === "ArrowRight") { this.pause(); this.setFrame(this.frame + (e.shiftKey ? 10 : 1)); }
            else if (e.code === "ArrowLeft") { this.pause(); this.setFrame(this.frame - (e.shiftKey ? 10 : 1)); }
            else if (e.code === "Home") { this.pause(); this.setFrame(0); }
        });
    }

    applyMuscleFilter() {
        const text = $("muscleFilter").value.trim();
        if (!text || !this.anim) { this.muscleVisible = null; return; }
        const patterns = text.split(/[\s,]+/).filter(Boolean).map(p =>
            new RegExp(p.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*"), "i"));
        this.muscleVisible = this.anim.muscles.map(m => patterns.some(p => p.test(m.name)));
    }

    applyVisibility() {
        if (!this.world) return;
        this.visualGroup.visible = $("drawVisuals").checked;
        const bone = this.materials.bone, opacity = parseFloat($("boneOpacity").value);
        bone.opacity = opacity;
        if (bone.transparent !== opacity < 1) {  // switching needs a new shader
            bone.transparent = opacity < 1;
            bone.depthWrite = opacity >= 1;
            bone.needsUpdate = true;
        }
        if (this.muscles) this.muscles.object.visible = $("drawMuscles").checked;
        this.pickPending = true;
        this.wrapGroup.visible = $("drawWrap").checked;
        const colliders = $("drawColliders").checked;
        this.colliderObjects.forEach(o => { if (o) o.visible = colliders || o.userData.alwaysShow; });
        if (this.renderer.shadowMap.enabled !== $("drawShadows").checked) {  // every shader changes
            this.renderer.shadowMap.enabled = $("drawShadows").checked;
            this.scene.traverse(o => { if (o.material) o.material.needsUpdate = true; });
        }
    }

    togglePlay() { this.playing ? this.pause() : this.play(); }

    play() {
        if (!this.anim || this.anim.frames.length < 2) return;
        if (this.frame >= this.anim.frames.length - 1) this.setFrame(0);
        this.playing = true;
        this.playTime = this.frame / this.anim.fps;
        this.lastTick = null;
        $("playButton").textContent = "❚❚";
    }

    pause() {
        this.playing = false;
        $("playButton").textContent = "▶";
    }

    tick(now) {
        if (this.playing && this.anim) {
            if (this.lastTick !== null) this.playTime += (now - this.lastTick) / 1000 * parseFloat($("speed").value);
            this.lastTick = now;
            let index = Math.floor(this.playTime * this.anim.fps + 1e-6);
            if (index >= this.anim.frames.length) {
                if ($("loop").checked) {
                    this.playTime = 0;
                    index = 0;
                } else {
                    index = this.anim.frames.length - 1;
                    this.pause();
                }
            }
            this.setFrame(index);
        }
        this.cameras.forEach(v => v.controls.update());
        if (this.pickPending) {
            this.pickPending = false;
            this.pick();
        }

        const w = this.container.clientWidth, h = this.container.clientHeight;
        const r = this.renderer;
        r.setScissorTest(this.cameras.length > 1);
        for (const view of this.cameras) {
            const [rx, ry, rw, rh] = view.spec.rect;
            const gap = this.cameras.length > 1 ? 1 : 0;
            r.setViewport(rx * w + gap, ry * h + gap, rw * w - 2 * gap, rh * h - 2 * gap);
            r.setScissor(rx * w + gap, ry * h + gap, rw * w - 2 * gap, rh * h - 2 * gap);
            r.render(this.scene, view.camera);
        }
    }

    // --- Loading ---
    async load(url, label) {
        const status = $("loadingStatus");
        status.textContent = `Loading ${label}...`;
        status.style.display = "block";
        this.pause();
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
            const anim = await response.json();
            this.build(anim);
            $("currentTrajectory").textContent = label;
            const info = [`${anim.frames.length} frames at ${anim.fps.toFixed(0)} Hz`, `${anim.muscles.length} muscles`];
            if (anim.wrap_surfaces?.length) info.push(`${anim.wrap_surfaces.length} wrap surfaces`);
            if (anim.source_version === 1) info.push("old format: no wrapping, tendons estimated");
            $("trajectoryInfo").textContent = info.join(" · ");
            status.style.display = "none";
        } catch (err) {
            console.error(err);
            status.textContent = `Could not load ${label}: ${err.message}`;
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const start = () => {
        const container = $("viewer");
        if (!container) { setTimeout(start, 50); return; }  // NiceGUI may add the element after this script runs
        window.mskViewer = new Viewer(container);
        if (window.mskPendingLoad) window.mskViewer.load(...window.mskPendingLoad);
    };
    start();
});
