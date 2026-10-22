import math

import bolt
import torch
import warp as wp

from msk_envs.geometry import quat_normalize, get_position_from_transform, get_rotation_from_transform
from msk_envs.envs.utils.model_setup import ModelInitializer
from msk_envs.envs.utils.perturber import Perturber
from msk_envs.envs.utils.starting_state import StartingStateHelper
from msk_envs.envs.utils.scene_settings import SceneSettings
from .config import EnvConfig, resolve_model_path


class MSKEnv:
    """
    Base class for MSK environments: a batch of [num_envs] worlds simulated in lockstep by bolt.

    Subclasses define the observation, reward terms and termination by overriding _get_obs,
    _compute_raw_reward_dict and _get_terminated, and can use the hooks below for extra behavior.
    """

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

        self._load_model(env_config, requires_visuals)
        self._bind_state()

        # Pre-compute useful body IDs and offsets
        self.ground_id = self.body_id_lookup["ground"]
        self.root_id = self.body_id_lookup["pelvis"]
        self.root_pos = self.body_positions[:, self.root_id]

        # --- RL Environment metadata ---
        self.action_range = (-1.0, 1.0)
        self.max_episode_duration = env_config.max_episode_duration
        self.delta_t = env_config.delta_t
        self.reset_tensor = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        # Raw reward terms of the current step, keyed "rew_<name>" and scaled by reward_lambdas["lambda_<name>"]
        self.reward_dict = {}
        self.reward_lambdas = env_config.reward_lambdas

        # --- Simulator settings
        # Number of sim steps required to reach RL env step.
        #  If adaptive, just step to desired time
        #  Otherwise, step in increments of [delta_t_sim]
        self.is_adaptive = bolt.is_adaptive(env_config.integrator)
        if self.is_adaptive:
            self.delta_t_sim = self.delta_t
            self.sim_steps_per_env_step = 1
        else:
            self.delta_t_sim = env_config.delta_t_sim
            self.sim_steps_per_env_step = math.ceil(self.delta_t / self.delta_t_sim)
        self.cuda_graph = cuda_graph
        self._setup_cuda_graphs()

        # Random push perturbations
        self.perturber = Perturber(
            num_envs=num_envs,
            device=self.device,
            perturbation_duration=env_config.perturbation_duration,
            perturbation_frequency=env_config.perturbation_frequency,
            force_std=env_config.force_std,
            delta_t=self.delta_t,
            enabled=env_config.apply_perturbations,
        )
        # Random starting pose generator
        self.starting_state_helper = StartingStateHelper(
            starting_pose_path=resolve_model_path(env_config.starting_pose_path),
            qpos_id_lookup=self.qpos_id_lookup,
            dof_id_lookup=self.dof_id_lookup,
            num_qpos=self.num_qpos,
            num_dofs=self.num_dofs,
            num_muscles=self.num_muscles,
            num_envs=num_envs,
            apply_start_noise=env_config.apply_start_noise,
            apply_swap_lr=env_config.apply_swap_lr,
            q_noise=env_config.q_noise,
            qv_noise=env_config.qv_noise,
            default_activation=env_config.default_activation,
            device=self.device,
        )
        self.enforce_ground_contact = env_config.enforce_ground_contact

    def _load_model(self, env_config: EnvConfig, requires_visuals: bool) -> None:
        """ Load the msk model and apply the model modifications from [env_config] """
        load_result = bolt.load_model(
            model_path=resolve_model_path(env_config.model_path),
            n_worlds=self.num_worlds,
            integrator=env_config.integrator,
            requires_visuals=requires_visuals,
            muscle_fn_path=(resolve_model_path(env_config.muscle_function_path)
                            if env_config.use_function_based_path else None),
        )
        self.load_result = load_result
        self.m = load_result.model
        self.d = load_result.data
        self.root_free = load_result.root_free

        # Post-load colliders modifications
        self._add_colliders(env_config)
        self.ground_rotation = quat_normalize(
            torch.tensor(env_config.ground_rotation, dtype=torch.float, device=self.device))
        ModelInitializer.modify_ground_collider(
            load_result=self.load_result,
            ground_rotation=self.ground_rotation,
        )
        ModelInitializer.modify_collider_params(
            load_result=self.load_result,
            contact_params_path=resolve_model_path(env_config.contact_params_path),
            use_specified_contact_params=env_config.use_specified_contact_params,
        )
        # Post-load model modifications
        ModelInitializer.modify_muscle_activation_dynamics(
            m=self.m,
            muscle_activation_dynamics=env_config.muscle_activation_dynamics,
            muscle_activation_time_const=env_config.muscle_activation_time_const,
            muscle_deactivation_time_const=env_config.muscle_deactivation_time_const,
            muscle_activation_dynamics_smoothing=env_config.muscle_activation_dynamics_smoothing,
            muscle_min_activation=env_config.muscle_min_activation,
            muscle_max_activation=env_config.muscle_max_activation,
        )
        ModelInitializer.modify_muscle_contraction_dynamics(
            m=self.m,
            muscle_contraction_dynamics=env_config.muscle_contraction_dynamics,
            muscle_multiplier=env_config.muscle_multiplier,
            muscle_fiber_damping=env_config.muscle_fiber_damping,
            muscle_active_force_width_scale=env_config.muscle_active_force_width_scale,
            muscle_v_max=env_config.muscle_v_max,
            ignore_short_elastic_tendons=env_config.ignore_short_elastic_tendons,
            force_rigid_tendons=env_config.force_rigid_tendons
        )
        ModelInitializer.modify_joints(
            m=self.m,
            armature=env_config.armature,
            use_linear_stop=env_config.use_linear_stop,
            root_free=self.root_free,
        )
        ModelInitializer.modify_physics(
            m=self.m,
            gravity=env_config.gravity,
        )
        ModelInitializer.modify_integrator(
            m=self.m,
            integrator_accuracy=env_config.integrator_accuracy,
            integrator_min_step_size=env_config.integrator_min_step_size,
            integrator_use_inf_norm=env_config.integrator_use_inf_norm,
            use_implicit_damping=env_config.use_implicit_damping,
            use_implicit_friction_damping=env_config.use_implicit_friction_damping,
            step_estimator=env_config.integrator_step_estimator,
        )
        bolt.reinitialize_model(self.m, self.d)

    def _bind_state(self) -> None:
        """ Store name lookups, model properties, and references to the simulator's state tensors """
        load_result = self.load_result
        self.body_id_lookup = load_result.body_id_lookup
        self.dof_id_lookup = load_result.dof_id_lookup
        self.qpos_id_lookup = load_result.qpos_id_lookup
        self.limit_id_lookup = load_result.limit_id_lookup
        self.muscle_id_lookup = load_result.muscle_id_lookup
        self.actuator_id_lookup = load_result.actuator_id_lookup
        self.collider_id_lookup = load_result.collider_id_lookup
        self.visuals = load_result.mesh_load_results

        # --- Model properties
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
        self.collider_body_id = bolt.geom_bodyid(self.m)

        # --- Data properties. The following are all references into the simulator's state
        # [num_envs]
        self.time = bolt.time(self.d)
        # [num_envs, num_muscles]
        self.muscle_activations = bolt.muscle_activations(self.d)
        self.muscle_excitations = bolt.muscle_excitations(self.d)
        self.muscle_norm_fiber_lengths = bolt.muscle_norm_fiber_lengths(self.d)  # in optimal fiber length (ofl)
        self.muscle_norm_fiber_velocities = bolt.muscle_norm_fiber_velocities(self.d)  # in ofl/s
        self.muscle_passive_length_multiplier = bolt.muscle_passive_length_multiplier(self.d)
        self.muscle_active_length_multiplier = bolt.muscle_active_length_multiplier(self.d)
        self.muscle_active_velocity_multiplier = bolt.muscle_active_velocity_multiplier(self.d)
        # [num_envs, num_muscles, num_qpos]
        self.muscle_moment_arms = bolt.muscle_moment_arms(self.d)
        # [num_envs, num_actuators]
        self.actuator_activations = bolt.actuator_activations(self.d)
        self.actuator_activations_dot = bolt.actuator_activations_dot(self.d)
        self.actuator_excitations = bolt.actuator_excitations(self.d)
        # [num_envs, num_bodies, 7]
        self.body_transforms = bolt.body_transforms(self.d)
        # [num_envs, num_bodies, 3]
        self.body_positions = bolt.body_com_positions(self.d)
        # [num_envs, num_bodies, 4] (x, y, z, w)
        self.body_rotations = get_rotation_from_transform(self.body_transforms)
        # [num_envs, num_bodies, 6] (ang, lin)
        self.body_velocities = bolt.body_velocities(self.d)
        self.body_accelerations = bolt.body_accelerations(self.d)
        self.body_user_forces = bolt.body_user_forces(self.d)
        # [num_envs, num_bodies, 3] linear part of body_velocities
        self.body_lin_velocities = self.body_velocities[..., 3:]
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
        self.body_self_collision_forces = bolt.body_self_collisions(self.d)
        # [num_envs, 3]
        self.grf = bolt.grf(self.d)
        # [num_envs, num_visuals, 3]
        self.visual_transforms = bolt.get_visual_transforms(self.d)
        self.visual_positions = get_position_from_transform(self.visual_transforms)
        # [num_envs, num_visuals, 4]
        self.visual_rotations = get_rotation_from_transform(self.visual_transforms)
        # [num_envs, num_sites, 3]
        self.site_positions = bolt.site_positions(self.d)
        self.marker_positions = self.site_positions[:, bolt.get_site_marker_slice(self.m), :]

    # --- Simulator ---

    def build_graph(self, *fns):
        """ Returns the CUDA Graph for executing [fns] """
        if not self.cuda_graph:
            return None
        assert torch.cuda.is_available()
        with wp.ScopedCapture() as capture:
            for fn in fns:
                fn(self.m, self.d)
        return capture.graph

    def _setup_cuda_graphs(self):
        """ Build CUDA graphs for relevant bolt functions """
        self.step_graph = self.build_graph(bolt.step)
        self.fk_graph = self.build_graph(bolt.fk)
        self.reset_graph = self.build_graph(bolt.reset)

    def launch_sim_step(self):
        """ Launch simulator step(s), integrating sim until RL env step size """
        for _ in range(self.sim_steps_per_env_step):
            bolt.increment_next_time(self.m, self.d, self.delta_t_sim)
            if self.cuda_graph:
                wp.capture_launch(self.step_graph)
            else:
                bolt.step(self.m, self.d)
        wp.synchronize()

    def launch_sim_reset(self):
        """ Launch simulator reset """
        bolt.set_reset(self.d, self.reset_tensor)
        if self.cuda_graph:
            wp.capture_launch(self.reset_graph)
            wp.synchronize()
        else:
            bolt.reset(self.m, self.d)

    def fk(self):
        """ Forward kinematics only (only position dependent) """
        if self.cuda_graph:
            wp.capture_launch(self.fk_graph)
            wp.synchronize()
        else:
            bolt.fk(self.m, self.d)

    # --- Hooks for subclasses ---

    def _add_colliders(self, env_config: EnvConfig) -> None:
        """ Add colliders to self.load_result.colliders. Runs before the model is (re)initialized """

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        """ Additional reset behavior. Occurs before sim reset """

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        """ Additional reset behavior. Occurs after sim reset """

    def _pre_step(self) -> None:
        """ Any computations needed before the actions are applied """

    def update_metrics(self) -> None:
        """ Update additional metrics after each sim step """

    def additional_metrics(self) -> dict:
        """ Additional metrics to log during training """
        return {}

    def scene_settings(self) -> SceneSettings:
        """ Scene settings for the viewer/renderer """
        return SceneSettings()

    def _get_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _compute_raw_reward_dict(self) -> None:
        """ Compute self.reward_dict. Guaranteed to run once per step """
        raise NotImplementedError

    def _get_terminated(self) -> torch.Tensor:
        raise NotImplementedError

    # --- Helpers for subclasses ---

    def _qpos_ids_excluding(self, *qpos_names: str) -> torch.Tensor:
        """ Indices of all qpos except [qpos_names] (names missing from the model are ignored) """
        exclude = {self.qpos_id_lookup[name] for name in qpos_names if name in self.qpos_id_lookup}
        keep = [i for i in range(self.num_qpos) if i not in exclude]
        return torch.tensor(keep, device=self.device, dtype=torch.long)

    # --- Actions ---

    def num_obs(self) -> int:
        return self._get_obs().shape[1]

    def num_actions(self) -> int:
        return self._get_actions().shape[1]

    def get_blank_actions(self) -> torch.Tensor:
        """ Returns actions of all 0 in correct shape """
        return torch.zeros_like(self._get_actions())

    def get_random_actions(self) -> torch.Tensor:
        """ Returns randomly uniform actions in correct shape """
        return (torch.rand_like(self._get_actions()) *
                self.action_range[1] * 2 - self.action_range[1])

    def _get_actions(self) -> torch.Tensor:
        actions = torch.cat([
            self.muscle_excitations,
            self.actuator_excitations
        ], dim=1)
        return actions.detach().clone()

    def _set_actions(self, raw_action) -> None:
        """ Actions are excitations of muscles then actuators, mapped from [-1, 1] to [0, 1] """
        excitations = (torch.clamp(raw_action, -1.0, 1.0) + 1.0) / 2.0
        self.muscle_excitations.copy_(excitations[:, :self.num_muscles])
        self.actuator_excitations.copy_(excitations[:, self.num_muscles:])

    # --- Rewards ---

    def get_scaled_reward_dict(self) -> dict:
        """ Scale rewards by their multiplier (defined in hyperparams) """
        return {
            key: self.reward_lambdas[key.replace("rew_", "lambda_")] * raw_value
            for key, raw_value in self.reward_dict.items()
        }

    def get_rewards(self) -> torch.Tensor:
        """ Return total rewards (after scaling) """
        total_rewards = torch.zeros(self.num_worlds, device=self.device)
        for value in self.get_scaled_reward_dict().values():
            total_rewards += value
        return total_rewards.detach()

    def _get_truncated(self):
        truncated = (self.time >= self.max_episode_duration).float()
        return truncated.detach()

    # --- Reset and step ---

    def _sample_start_and_reset_sim(self):
        """ Sample new starting state and reset sim """
        reset_mask = self.reset_tensor.squeeze(-1).bool()
        if not reset_mask.any():
            return

        # Sample new starting state and apply
        self.starting_state_helper.create_new_starting_poses(reset_mask=reset_mask)
        self._apply_starting_state(reset_mask)

        # Ensure contact with ground
        if self.enforce_ground_contact and self.root_free:
            self.fk()
            self.starting_state_helper.adjust_for_ground_contact(
                root_pos=self.root_pos,
                collider_sizes=self.collider_sizes,
                collider_body_id=self.collider_body_id,
                collider_positions=self.collider_positions,
                ground_rotation=self.ground_rotation,
                reset_mask=reset_mask,
                root_height_qpos_id=self.qpos_id_lookup["pelvis_ty"],
                ground_id=self.ground_id,
            )
            self._apply_starting_state(reset_mask)

        self.launch_sim_reset()

    def _apply_starting_state(self, reset_mask: torch.Tensor) -> None:
        self.starting_state_helper.set_starting_state(
            time_out=self.time,
            q_out=self.joint_positions,
            qv_out=self.joint_velocities,
            activations_out=self.muscle_activations,
            actuator_activations_out=self.actuator_activations,
            reset_mask=reset_mask
        )

    def _perform_reset(self, resets: torch.Tensor):
        """ Internal reset call, resets envs where [resets] is 1 """
        reset_mask = resets.squeeze(-1).bool()
        self.reset_tensor.copy_(resets)
        self._upon_reset_pre_sim(reset_mask=reset_mask)
        self._sample_start_and_reset_sim()
        self._upon_reset_post_sim(reset_mask=reset_mask)
        self.reset_tensor.fill_(0.0)

    def pre_sim_step(self, actions) -> None:
        self._pre_step()
        self._set_actions(actions)
        if self.debug:
            assert not torch.isnan(actions).any(), "Actions contain NaN!"
        self.perturber.apply(self.root_id, self.body_user_forces)

    def rl_step(self):
        self._compute_raw_reward_dict()

        final_obs = self._get_obs()
        rew = self.get_rewards()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        done = torch.clamp(terminated + truncated, 0.0, 1.0).unsqueeze(-1)

        if done.any():  # Reset any worlds that are done
            self._perform_reset(done)

        # Training requires the observation *after* the reset (for next action). This is recomputed even without
        # resets, since _get_terminated may update task state that the observation depends on (e.g. carioca phase)
        obs = self._get_obs()

        if self.debug:
            assert not torch.isnan(final_obs).any(), "Observations contain NaN!"
            assert not torch.isnan(obs).any(), "Observations contain NaN!"
            assert not torch.isnan(rew).any(), "Rewards contain NaN!"

        # Reward terms for logging
        info = {
            "final_observation": final_obs,
            "raw_rewards": self.reward_dict,
            "scaled_rewards": self.get_scaled_reward_dict(),
        }
        return obs, rew, terminated, truncated, info

    def step(self, actions):
        """ External step call """
        self.pre_sim_step(actions)
        self.launch_sim_step()
        self.update_metrics()
        return self.rl_step()

    def reset(self):
        """ External reset call, resets all envs """
        self._perform_reset(resets=torch.ones_like(self.reset_tensor))
        return self._get_obs()
