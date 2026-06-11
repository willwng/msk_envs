import bolt
import torch
import warp as wp
from bolt import ModelLoadResult, Model

from msk_envs.utils.contact_params import parse_contact_params


class ModelInitializer:

    @staticmethod
    def modify_ground_collider(
            load_result: ModelLoadResult,
            ground_rotation: torch.Tensor,
    ) -> None:
        for collider in load_result.colliders:
            if collider == bolt.GROUND_COLLIDER:
                collider.transform = wp.transform(wp.vec3(), wp.quat(ground_rotation))
        return

    @staticmethod
    def modify_collider_params(
            load_result: ModelLoadResult,
            contact_params_path: str,
            use_specified_contact_params: bool,
    ) -> None:
        # Set collider contact properties
        if use_specified_contact_params:
            contact_params = parse_contact_params(contact_params_path)
            for params in contact_params:
                if params.geom_name not in load_result.collider_id_lookup:
                    print(f"Warning: geometry '{params.geom_name}' not found in geom_id_lookup, skipping.")
                    continue
                geom_id = load_result.collider_id_lookup[params.geom_name]
                load_result.colliders[geom_id].stiffness = params.stiffness
                load_result.colliders[geom_id].dissipation = params.dissipation
                load_result.colliders[geom_id].priority = params.priority
                load_result.colliders[geom_id].friction[0] = params.static_friction
                load_result.colliders[geom_id].friction[1] = params.dynamic_friction
                load_result.colliders[geom_id].friction[2] = params.viscous_friction
                load_result.colliders[geom_id].transition_velocity = params.transition_velocity
        bolt.update_colliders(load_result)
        return

    @staticmethod
    def modify_muscle_activation_dynamics(
            m: Model,
            muscle_activation_dynamics,
            muscle_activation_time_const: float,
            muscle_deactivation_time_const: float,
            muscle_activation_dynamics_smoothing: float,
            muscle_min_activation: float,
            muscle_max_activation: float,
    ) -> None:
        bolt.set_activation_type(m, muscle_activation_dynamics)
        # Individual muscle properties
        for mm in bolt.muscle_metadata(m):
            # Activation dynamics
            mm.activation_time_const = muscle_activation_time_const
            mm.deactivation_time_const = muscle_deactivation_time_const
            mm.activation_dynamics_smoothing = muscle_activation_dynamics_smoothing
            mm.min_activation = muscle_min_activation
            mm.max_activation = muscle_max_activation
        return

    @staticmethod
    def modify_muscle_contraction_dynamics(
            m: Model,
            muscle_contraction_dynamics,
            muscle_multiplier: float,
            muscle_fiber_damping: float,
            muscle_active_force_width_scale: float,
            muscle_v_max: float,
            ignore_short_elastic_tendons: bool,
    ) -> None:
        bolt.set_contraction_type(m, muscle_contraction_dynamics)
        # Individual muscle properties
        for mm in bolt.muscle_metadata(m):
            # Muscle force multiplier
            mm.max_isometric_force *= muscle_multiplier
            # Contraction dynamics
            mm.fiber_damping = muscle_fiber_damping
            mm.active_force_width_scale = muscle_active_force_width_scale
            mm.v_max = muscle_v_max
            # Rigid tendon
            if ignore_short_elastic_tendons:
                mm.ignore_tendon_compliance = \
                    (mm.ignore_tendon_compliance or mm.tendon_slack_length < mm.optimal_fiber_length)
        return

    @staticmethod
    def modify_physics(
            m: Model,
            gravity: float,
            armature: float,
            integrator_accuracy: float,
            integrator_use_inf_norm: bool,
            use_linear_stop: bool,
            enable_drag: bool,
            use_implicit_damping: bool,
            root_free: bool,
    ) -> None:
        # Armature/joint settings
        dof_start = 6 if root_free else 0
        bolt.armature(m)[dof_start:] = armature
        bolt.set_use_linear_stop(m, use_linear_stop)
        # Integrator settings
        bolt.set_implicit_damping(m, use_implicit_damping)
        bolt.set_integrator_accuracy(m, integrator_accuracy)
        bolt.set_integrator_use_inf_norm(m, integrator_use_inf_norm)
        # Physics
        bolt.set_drag_enabled(m, enable_drag)
        bolt.set_gravity(m, gravity)
        return
