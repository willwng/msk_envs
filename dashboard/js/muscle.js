import * as THREE from 'three';

const muscleCache = {};

function activationToColor(activation) {
    // 0 is blue. 1 is red.
    const r = activation;
    const g = 0.0;
    const b = 1.0 - activation;
    return new THREE.Color().setRGB(r, g, b);
}

function resetMuscles() {
    for (const key in muscleCache) {
        muscleCache[key].forEach(muscle => {
            muscle.geometry.dispose();
            muscle.material.dispose();
        });
        delete muscleCache[key];
    }
}

function updateCapsule(capsule, p1, p2, radius, color) {
    const direction = new THREE.Vector3().subVectors(p2, p1);
    const length = direction.length();
    const dir = direction.normalize();
    const currentHeight = capsule.geometry.parameters.height;
    if (Math.abs(currentHeight - length) > 0.001) {
        capsule.geometry.dispose();
        capsule.geometry = new THREE.CapsuleGeometry(radius, length, 8, 16);
    }
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    capsule.position.copy(midpoint);
    capsule.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    // Update the material color for this specific capsule
    capsule.material.color.copy(color);
    capsule.material.needsUpdate = true;
}


function drawMuscleLine(name, muscle, addObjectCallback) {
    const color = activationToColor(muscle.activation);
    const muscleRadius = 0.005;

    let capsules = [];
    if (muscleCache[name]) {
        capsules = muscleCache[name];
    } else {
        // Create capsules for each segment
        for (let i = 0; i < muscle.points.length - 1; i++) {
            const muscleMaterial = new THREE.MeshStandardMaterial({
                color: 0xFF8888,
                side: THREE.DoubleSide,
                flatShading: false,
                metalness: 0.0,
                roughness: 1.0
            });

            const muscleGeometry = new THREE.CapsuleGeometry(muscleRadius, 0, 8, 16);
            const capsule = new THREE.Mesh(muscleGeometry, muscleMaterial);

            capsule.castShadow = true;
            capsule.receiveShadow = true;

            capsules.push(capsule);
        }
        muscleCache[name] = capsules;
    }

    // Update all capsules
    for (let i = 0; i < muscle.points.length - 1; i++) {
        const p1 = new THREE.Vector3(...muscle.points[i]);
        const p2 = new THREE.Vector3(...muscle.points[i + 1]);
        const capsule = capsules[i];
        updateCapsule(capsule, p1, p2, muscleRadius, color);
        addObjectCallback(capsule);
    }
}

export {drawMuscleLine, resetMuscles};