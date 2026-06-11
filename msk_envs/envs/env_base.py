import math
import os

import bolt
import torch
import warp as wp

from msk_envs.envs.perturber import Perturber
from msk_envs.utils.contact_params import parse_contact_params
from msk_envs.utils.global_params import UP_IDX
from msk_envs.utils.pose import parse_starting_pose, get_swap_left_right_data
from msk_envs.utils.quat import quat_normalize
from msk_envs.utils.scene_settings import SceneSettings
from msk_envs.utils.transforms import get_position_from_transform, get_rotation_from_transform
from .env_config import EnvConfig


class MSKEnv:
    """ Superclass for MSK environments """

    def _add_colliders(self, env_config: EnvConfig) -> None:
        """ Hook for envs to add colliders """
        return

    def _modify_colliders(self, env_config: EnvConfig) -> None:
        colliders = self.load_result.colliders

        # Rotated contact plane
        self.ground_rotation = torch.tensor(env_config.ground_rotation, dtype=torch.float, device=self.device)
        self.ground_rotation = quat_normalize(self.ground_rotation)
        for collider in colliders:
            if collider == bolt.GROUND_COLLIDER:
                collider.transform = wp.transform(wp.vec3(), wp.quat(self.ground_rotation))

        # Set collider contact properties
        if env_config.use_specified_contact_params:
            contact_params_path = os.path.join(self.curr_path, env_config.contact_params_path)
            contact_params = parse_contact_params(contact_params_path)
            for params in contact_params:
                if params.geom_name not in self.load_result.collider_id_lookup:
                    print(f"Warning: geometry '{params.geom_name}' not found in geom_id_lookup, skipping.")
                    continue
                geom_id = self.load_result.collider_id_lookup[params.geom_name]
                colliders[geom_id].stiffness = params.stiffness
                colliders[geom_id].dissipation = params.dissipation
                colliders[geom_id].priority = params.priority
                colliders[geom_id].friction[0] = params.static_friction
                colliders[geom_id].friction[1] = params.dynamic_friction
                colliders[geom_id].friction[2] = params.viscous_friction
                colliders[geom_id].transition_velocity = params.transition_velocity

        bolt.update_colliders(self.load_result)
        return

    def _setup_model(self, env_config: EnvConfig) -> None:
        """ Modify model parameters here. """
        # Muscles activation and fiber dynamic
        bolt.set_activation_type(self.m, env_config.muscle_activation_dynamics)
        bolt.set_contraction_type(self.m, env_config.muscle_contraction_dynamics)
        muscle_metadata = bolt.muscle_metadata(self.m)
        muscle_idx_to_name = {idx: name for name, idx in self.load_result.muscle_id_lookup.items()}
        for i, mm in enumerate(muscle_metadata):
            muscle_name = muscle_idx_to_name[i]
            # Muscle force multiplier
            if muscle_name in env_config.custom_muscle_multipliers:
                custom_multiplier = env_config.custom_muscle_multipliers[muscle_name]
                mm.max_isometric_force *= custom_multiplier
            else:
                mm.max_isometric_force *= env_config.muscle_multiplier
            # Activation dynamics
            mm.activation_time_const = env_config.muscle_activation_time_const
            mm.deactivation_time_const = env_config.muscle_deactivation_time_const
            mm.activation_dynamics_smoothing = env_config.muscle_activation_dynamics_smoothing
            mm.min_activation = env_config.muscle_min_activation
            mm.max_activation = env_config.muscle_max_activation
            # Contraction dynamics
            mm.fiber_damping = env_config.muscle_fiber_damping
            mm.active_force_width_scale = env_config.muscle_active_force_width_scale
            mm.v_max = env_config.muscle_v_max
            if env_config.ignore_short_elastic_tendons:
                mm.ignore_tendon_compliance = (
                        mm.ignore_tendon_compliance or mm.tendon_slack_length < mm.optimal_fiber_length)

        # Armature/joint settings
        dof_start = 6 if self.root_free else 0
        bolt.armature(self.m)[dof_start:] = env_config.armature
        bolt.set_use_linear_stop(self.m, env_config.use_linear_stop)
        # Integrator settings
        bolt.set_implicit_damping(self.m, env_config.use_implicit_damping)
        bolt.set_integrator_accuracy(self.m, env_config.integrator_accuracy)
        bolt.set_integrator_use_inf_norm(self.m, env_config.integrator_use_inf_norm)
        # Physics
        bolt.set_drag_enabled(self.m, env_config.enable_drag)
        bolt.set_gravity(self.m, env_config.gravity)

        bolt.reinitialize_model(self.m, self.d)
        return

    def _setup_cuda_graphs(self):
        if self.cuda_graph:
            assert torch.cuda.is_available()
            # Step graph
            with wp.ScopedCapture() as capture:
                bolt.step(self.m, self.d)
            self.step_graph = capture.graph

            # FK graph: forward kinematics (positions only)
            with wp.ScopedCapture() as capture:
                bolt.fk(self.m, self.d)
            self.fk_graph = capture.graph

            # Reset graph: call after resetting any the worlds
            with wp.ScopedCapture() as capture:
                bolt.reset(self.m, self.d)
            self.reset_graph = capture.graph

            # Post-step graph: computes things like muscle passive forces, needed for rewards
            with wp.ScopedCapture() as capture:
                bolt.compute_muscle_passive_forces(self.m, self.d)
            self.post_graph = capture.graph

            # Analytics graph: anything else needed for analytics
            with wp.ScopedCapture() as capture:
                bolt.compute_muscle_moments(self.m, self.d)
                bolt.compute_net_joint_moments(self.m, self.d)
                bolt.compute_muscle_force_breakdown(self.m, self.d)
            self.analytics_graph = capture.graph

            # Forward kinematics graph, useful for motion tracking
            with wp.ScopedCapture() as capture:
                bolt.fk(self.m, self.d)
            self.fk_graph = capture.graph
        return

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            cuda_graph: bool,
            debug: bool = False
    ):
        self.num_worlds = num_envs
        self.device = device
        self.debug = debug

        self.curr_path = os.path.abspath(os.path.dirname(__file__))

        # Load msk model
        load_result = bolt.load_model(
            model_path=os.path.join(self.curr_path, env_config.model_path),
            n_worlds=num_envs,
            integrator=env_config.integrator,
            requires_visuals=requires_visuals,
            muscle_fn_path=os.path.join(
                self.curr_path, env_config.muscle_function_path
            ) if env_config.use_function_based_path else None
        )
        self.load_result = load_result
        self.m, self.d = load_result.model, load_result.data
        self.root_free = load_result.root_free
        # Post-load model setup/modifications
        self._add_colliders(env_config)
        self._modify_colliders(env_config)
        self._setup_model(env_config)

        # Store all convenient lookups
        self.body_id_lookup = load_result.body_id_lookup
        self.dof_id_lookup = load_result.dof_id_lookup
        self.qpos_id_lookup = load_result.qpos_id_lookup
        self.limit_id_lookup = load_result.limit_id_lookup
        self.muscle_id_lookup = load_result.muscle_id_lookup
        self.actuator_id_lookup = load_result.actuator_id_lookup
        self.collider_id_lookup = load_result.collider_id_lookup
        self.visuals = load_result.mesh_load_results

        # Model properties
        self.num_qpos = bolt.get_num_qpos(self.m)
        self.num_dofs = bolt.get_num_dofs(self.m)
        self.num_bodies = bolt.get_num_bodies(self.m)
        self.num_muscles = bolt.get_num_muscles(self.m)
        self.num_actuators = bolt.get_num_actuators(self.m)
        self.num_colliders = bolt.get_num_colliders(self.m)
        self.muscle_metadata = bolt.muscle_metadata(self.m)
        self.muscle_max_isometric_forces = bolt.muscle_max_isometric_forces(self.m)  # read-only
        # [num_envs, num_bodies]
        self.body_mass = bolt.body_mass(self.m)
        self.total_mass = self.body_mass.sum()
        self.gravity = bolt.gravity(self.m)
        # [num_envs, num_colliders]
        self.collider_sizes = bolt.get_collider_sizes(self.m)

        # Data properties. The following are all references
        # [num_envs]
        self.time = bolt.time(self.d)
        # [num_envs, num_muscles]
        self.muscle_activations = bolt.muscle_activations(self.d)
        self.muscle_excitations = bolt.muscle_excitations(self.d)
        self.muscle_fiber_lengths = bolt.muscle_fiber_lengths(self.d)
        self.muscle_fiber_velocities = bolt.muscle_fiber_velocities(self.d)
        self.muscle_powers = bolt.muscle_powers(self.d)
        self.muscle_active_length_multiplier = bolt.muscle_active_length_multiplier(self.d)
        self.muscle_active_velocity_multiplier = bolt.muscle_active_velocity_multiplier(self.d)
        # [num_envs, num_muscles, num_qpos]
        self.muscle_moment_arms = bolt.muscle_moment_arms(self.d)
        # [num_envs, num_actuators]
        self.actuator_activations = bolt.actuator_activations(self.d)
        self.actuator_activations_dot = bolt.actuator_activations_dot(self.d)
        self.actuator_excitations = bolt.actuator_excitations(self.d)
        # [num_envs, num_bodies, 7], ignore ground
        self.body_transforms = bolt.body_transforms(self.d)
        # [num_envs, num_bodies, 3], ignore ground
        self.body_positions = bolt.body_com_positions(self.d)
        # [num_envs, num_bodies, 4] (w, x, y, z)
        self.body_rotations = get_rotation_from_transform(self.body_transforms)
        # [num_envs, num_bodies, 6] (ang, lin)
        self.body_velocities = bolt.body_velocities(self.d)
        self.body_accelerations = bolt.body_accelerations(self.d)
        # [num_envs, num_bodies, 6] (frc, trq)
        self.body_user_forces = bolt.body_user_forces(self.d)
        # [num_envs, num_qpos]
        self.joint_positions = bolt.joint_positions(self.d)
        # [num_envs, num_dofs]
        self.joint_velocities = bolt.joint_velocities(self.d)
        # [num_envs, num_dofs]
        self.ufrc_spring = bolt.ufrc_spring(self.d)
        self.ufrc_damper = bolt.ufrc_damper(self.d)
        self.ufrc_limit = bolt.ufrc_limit(self.d)
        self.ufrc_muscle_passive = bolt.ufrc_muscle_passive(self.d)
        # [num_envs, num_colliders]
        self.collider_forces = bolt.collider_forces(self.d)
        # [num_envs, num_colliders, 3]
        self.collider_positions = get_position_from_transform(bolt.get_collider_transforms(self.d))
        # self.collider_self_forces = bolt.collider_self_forces(self.d)
        self.body_self_collision_forces = bolt.body_self_collisions(self.d)
        # [num_envs, 3]
        self.grf = bolt.grf(self.d)
        # [num_envs, num_visuals, 3]
        self.visual_transforms = bolt.get_visual_transforms(self.d)
        self.visual_positions = get_position_from_transform(self.visual_transforms)
        # [num_envs, num_visuals, 4]
        self.visual_rotations = get_rotation_from_transform(self.visual_transforms)

        # RL Environment metadata
        self.action_range = (-1.0, 1.0)
        self.max_episode_duration = env_config.max_episode_duration
        self.delta_t = env_config.delta_t
        self.reset_tensor = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)

        # Simulation steps required to reach env step. if adaptive, just step to desired time
        #  otherwise step in increments of [delta_t_sim]
        self.is_adaptive = bolt.is_adaptive(env_config.integrator)
        if self.is_adaptive:
            self.delta_t_sim = self.delta_t
            self.sim_steps_per_env_step = 1
        else:
            self.delta_t_sim = env_config.delta_t_sim
            self.sim_steps_per_env_step = math.ceil(self.delta_t / self.delta_t_sim)

        # Starting position, load from file
        start_pose_path = os.path.join(self.curr_path, env_config.starting_pose_path)
        q, qv = parse_starting_pose(
            start_pose_path, self.qpos_id_lookup, self.dof_id_lookup, self.num_qpos, self.num_dofs)
        assert len(q) == self.num_qpos and len(qv) == self.num_dofs
        q_torch = torch.tensor(q, dtype=torch.float32, device=device)
        qv_torch = torch.tensor(qv, dtype=torch.float32, device=device)
        self.start_pose_base = q_torch.unsqueeze(0)
        self.start_velocity_base = qv_torch.unsqueeze(0)
        # Repeat for all envs
        self.start_pose = q_torch.unsqueeze(0).repeat(num_envs, 1)
        self.start_velocity = qv_torch.unsqueeze(0).repeat(num_envs, 1)
        # Pose noise/reset settings
        self.noise_start = env_config.noise_start
        self.q_noise = env_config.q_noise
        self.qv_noise = env_config.qv_noise
        self.noise_root = env_config.noise_root
        self.enforce_ground_contact = env_config.enforce_ground_contact
        # Pre-compute left-right swap data
        self.swap_lr = env_config.swap_lr
        if self.swap_lr:
            self.swap_lr_data = get_swap_left_right_data(self.qpos_id_lookup, self.dof_id_lookup)
        else:
            self.swap_lr_data = []

        # Starting muscle activations
        self.noise_act_start = False
        if env_config.default_activation == -1.0:
            self.start_activations = torch.rand((num_envs, self.num_muscles), device=device)
            self.noise_act_start = True
        else:
            self.start_activations = torch.ones(
                (num_envs, self.num_muscles), device=device) * env_config.default_activation

        # Rewards storage
        self.reward_dict = {}
        self.reward_lambdas = env_config.reward_lambdas

        # CUDA Graphs
        self.cuda_graph = cuda_graph
        self._setup_cuda_graphs()

        # Pre-compute useful body IDs and offsets
        self.ground_id = self.body_id_lookup["ground"]
        self.root_id = self.body_id_lookup["pelvis"]
        self.root_pos = self.body_positions[:, self.root_id]

        # Set up random perturber
        self.perturber = Perturber(
            num_envs=num_envs,
            device=self.device,
            perturbation_duration=env_config.perturbation_duration,
            perturbation_frequency=env_config.perturbation_frequency,
            force_std=env_config.force_std,
            delta_t=self.delta_t,
            enabled=env_config.apply_perturbations,
        )
        return

    def noise_start_pose(self, reset_mask: torch.Tensor) -> None:
        """
        Re-noise the starting pose and velocity for envs where reset_mask is 1
        Note: takes effect on next reset or init
        """
        # Repeat for all envs
        q = self.start_pose_base.repeat(self.num_worlds, 1)
        qv = self.start_velocity_base.repeat(self.num_worlds, 1)

        # Noise starting pose
        if self.noise_start:
            if self.noise_root:
                q += torch.randn_like(q) * self.q_noise
            else:
                q[:, 7:] += torch.randn_like(q[:, 7:]) * self.q_noise
            qv += torch.randn_like(qv) * self.qv_noise

        # Swap left/right sides
        if self.swap_lr:
            # Randomly choose to swap left and right
            swap_mask = (torch.rand(self.num_worlds, device=q.device) > 0.5)
            q_old, qv_old = q.clone(), qv.clone()
            for swap_pair in self.swap_lr_data:
                rq, lq = swap_pair.qpos_r, swap_pair.qpos_l
                q[swap_mask, rq] = q_old[swap_mask, lq]
                q[swap_mask, lq] = q_old[swap_mask, rq]

                rdq, ldq = swap_pair.dof_r, swap_pair.dof_l
                qv[swap_mask, rdq] = qv_old[swap_mask, ldq]
                qv[swap_mask, ldq] = qv_old[swap_mask, rdq]

        # Set the new starting pose
        self.start_pose[reset_mask, :] = q[reset_mask, :]
        self.start_velocity[reset_mask, :] = qv[reset_mask, :]

        # Noise starting muscle activations
        if self.noise_act_start:
            random_acts = torch.rand((self.num_worlds, self.num_muscles), device=self.device)
            self.start_activations[reset_mask, :] = random_acts[reset_mask, :]
        return

    def num_obs(self) -> int:
        return self._get_obs().shape[1]

    def num_actions(self) -> int:
        return self._get_actions().shape[1]

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        """ Hook for additional reset behavior in subclasses. Occurs before sim reset """
        return

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        """ Hook for additional reset behavior in subclasses. Occurs after sim reset """
        return

    def _reset_sim(self):
        # Reset time, starting pose, muscle and actuator activations
        reset_mask = self.reset_tensor.squeeze(-1).bool()
        if reset_mask.any():
            self.time[reset_mask] = 0.0
            self.joint_positions[reset_mask, :] = self.start_pose[reset_mask, :]
            self.joint_velocities[reset_mask, :] = self.start_velocity[reset_mask, :]
            self.muscle_activations[reset_mask, :] = self.start_activations[reset_mask, :]
            self.actuator_activations[reset_mask, :] = 0.5

        # Run forward kinematics to ensure contact with ground
        if self.enforce_ground_contact and self.root_free:
            self.fk()
            collider_body_id = bolt.geom_bodyid(self.m)
            non_ground_collider_ids = torch.where(collider_body_id != self.ground_id)[0]

            # Need at least one non-root collider
            if non_ground_collider_ids.size(0) > 1:
                collider_heights = self.collider_positions[:, non_ground_collider_ids, UP_IDX]
                collider_heights -= self.collider_sizes[non_ground_collider_ids, 0]
                lowest_collider_height = collider_heights.min(dim=1).values
                adjustment = -lowest_collider_height
                root_height_q_id = self.qpos_id_lookup["pelvis_ty"] if self.root_free else None
                self.joint_positions[reset_mask, root_height_q_id] += adjustment[reset_mask]

        # Reset sim
        bolt.set_reset(self.d, self.reset_tensor)
        if self.cuda_graph:
            wp.capture_launch(self.reset_graph)
            wp.synchronize()
        else:
            bolt.reset(self.m, self.d)
        return

    # The following are environment-specific and need to be implemented
    def _get_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_actions(self) -> torch.Tensor:
        actions = torch.cat([
            self.muscle_excitations,
            self.actuator_excitations
        ], dim=1)
        return actions.detach().clone()

    def _set_actions(self, raw_action) -> None:
        self._set_muscle_excitations(raw_action[:, :self.num_muscles])
        self._set_actuator_excitations(raw_action[:, self.num_muscles:])
        return

    def _pre_step(self) -> None:
        """ Hook for any pre-step computations """
        return

    def _compute_raw_reward_dict(self):
        """ Guarantee to only run once per step """
        raise NotImplementedError

    def _get_terminated(self):
        raise NotImplementedError

    # Rest is standard
    def get_blank_actions(self) -> torch.Tensor:
        """ Returns actions of all 0 in correct shape """
        return torch.zeros_like(self._get_actions())

    def get_random_actions(self) -> torch.Tensor:
        """ Returns randomly uniform actions in correct shape """
        return (torch.rand_like(self._get_actions()) *
                self.action_range[1] * 2 - self.action_range[1])

    def get_scaled_reward_dict(self) -> dict:
        reward_dict = self.reward_dict
        scaled_reward_dict = {}
        for key, raw_value in reward_dict.items():
            lambda_key = key.replace("rew_", "lambda_")
            lambda_value = self.reward_lambdas[lambda_key]
            scaled_reward_dict[key] = lambda_value * raw_value
        return scaled_reward_dict

    def get_rewards(self) -> torch.Tensor:
        scaled_rew_dict = self.get_scaled_reward_dict()
        total_rewards = torch.zeros(self.num_worlds, device=self.device)
        for key, value in scaled_rew_dict.items():
            total_rewards += value
        return total_rewards.detach()

    def _get_truncated(self):
        truncated = (self.time >= self.max_episode_duration).float()
        return truncated.detach()

    def _set_muscle_excitations(self, raw_action) -> None:
        # Clamp to [-1, 1], then map to [0, 1]
        clamped_action = torch.clamp(raw_action, -1.0, 1.0)
        excitation = (clamped_action + 1.0) / 2.0
        self.muscle_excitations.copy_(excitation)
        return

    def _set_actuator_excitations(self, raw_action) -> None:
        # Clamp to [-1, 1], then map to [0, 1]
        clamped_action = torch.clamp(raw_action, -1.0, 1.0)
        excitations = (clamped_action + 1.0) / 2.0
        self.actuator_excitations.copy_(excitations)
        return

    def _perform_reset(self, reset_mask: torch.Tensor):
        """ Internal reset call, resets only envs where reset_mask is 1 """
        self.reset_tensor.copy_(reset_mask)
        self.noise_start_pose(reset_mask.squeeze(-1).bool())
        self._upon_reset_pre_sim(reset_mask.squeeze(-1).bool())
        self._reset_sim()
        self._upon_reset_post_sim(reset_mask.squeeze(-1).bool())
        self.reset_tensor.fill_(0.0)

    # The following impl of step is kinda jank, but we need this separation for logging
    def pre_step(self, actions) -> None:
        self._pre_step()

        # Set actions
        self._set_actions(actions)

        if self.debug:
            assert not torch.isnan(actions).any(), "Actions contain NaN!"

        # Apply perturbations
        self.perturber.apply(self.root_id, self.body_user_forces)
        return

    def launch_sim_step(self):
        if self.cuda_graph:
            for _ in range(self.sim_steps_per_env_step):
                bolt.increment_next_time(self.m, self.d, self.delta_t_sim)
                wp.capture_launch(self.step_graph)
            wp.capture_launch(self.post_graph)
            wp.synchronize()
        else:
            for _ in range(self.sim_steps_per_env_step):
                bolt.increment_next_time(self.m, self.d, self.delta_t_sim)
                bolt.step(self.m, self.d)
            bolt.post(self.m, self.d)
        return

    def rl_step(self):
        # Only compute reward dict once per step
        self._compute_raw_reward_dict()

        final_obs = self._get_obs()
        rew = self.get_rewards()
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # Reset any worlds that are done
        done = torch.clamp(terminated + truncated, 0.0, 1.0).unsqueeze(-1)
        if done.any():
            self._perform_reset(done)

        # Training requires the observation *after* the reset (for next action)
        obs = self._get_obs()
        # Return raw reward terms for logging
        info = {
            "final_observation": final_obs,
            "raw_rewards": self.reward_dict,
            "scaled_rewards": self.get_scaled_reward_dict(),
        }

        if self.debug:
            assert not torch.isnan(obs).any(), "Observations contain NaN!"
            assert not torch.isnan(rew).any(), "Rewards contain NaN!"

        return obs, rew, terminated, truncated, info

    def step(self, actions):
        """ External step call """
        self.pre_step(actions)
        self.launch_sim_step()
        self.update_metrics()
        return self.rl_step()

    def reset(self):
        """ External reset call, resets all envs """
        self._perform_reset(reset_mask=torch.ones_like(self.reset_tensor))
        obs = self._get_obs()
        return obs

    def fk(self):
        """ Forward kinematics only (only position dependent) """
        if self.cuda_graph:
            wp.capture_launch(self.fk_graph)
            wp.synchronize()
        else:
            bolt.fk(self.m, self.d)

    def scene_settings(self) -> SceneSettings:
        """ Override to provide custom scene settings for viewer/renderer """
        return SceneSettings()

    def additional_metrics(self) -> dict:
        """ Hook for additional metrics to log during training """
        return {}

    def update_metrics(self) -> None:
        """ Hook to update additional metrics """
        return
