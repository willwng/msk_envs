import torch
import msk_warp
import warp as wp
import os

from .env_config import EnvConfig
from msk_envs.utils.pose import parse_starting_pose, get_swap_left_right_data, parse_starting_activations
from msk_envs.utils.joint_limits import get_exp_limit_curves, get_joint_limits
from msk_envs.utils.global_params import UP_IDX, SIDE_IDX, FWD_IDX, build_axis


class MSKEnv:
    """ Superclass for MSK environments """

    def _setup_model(self, env_config: EnvConfig) -> None:
        """ We're going to modify some model parameters here. """
        # Reset stiffness, damping, armature to zero
        stiffness = msk_warp.stiffness(self.m)  # [num_bodies]
        damping = msk_warp.damping(self.m)  # [num_dofs]
        armature = msk_warp.armature(self.m)  # [num_dofs]
        damping[:] = 0.0
        stiffness[:] = 0.0
        armature[:] = 0.0

        # Joint damping
        damping[6:] = env_config.joint_damping

        # Joint armature
        armature[6:] = env_config.joint_armature

        # Torso damping
        torso_id = self.lookup_body_id("torso")
        dof_adr = msk_warp.get_dof_adr(self.m, torso_id)
        dof_num = msk_warp.get_dof_num(self.m, torso_id)
        damping[dof_adr:dof_adr + dof_num] = env_config.torso_damping

        # Foot stiffness and damping
        for toe in ["toes_l", "toes_r"]:
            toe_id = self.lookup_body_id(toe)
            stiffness[toe_id] = env_config.toes_stiffness

            dof_adr = msk_warp.get_dof_adr(self.m, toe_id)
            dof_num = msk_warp.get_dof_num(self.m, toe_id)
            damping[dof_adr:dof_adr + dof_num] = env_config.toes_damping

        # Muscles
        for mm in msk_warp.muscle_metadata(self.m):
            mm.max_isometric_force *= env_config.muscle_multiplier
            mm.fiber_damping = env_config.muscle_fiber_damping
            mm.min_activation = env_config.muscle_min_activation
            mm.max_activation = env_config.muscle_max_activation
            mm.v_max = env_config.muscle_v_max
        msk_warp.set_muscle_dynamics_substeps(
            self.m, env_config.muscle_dynamics_substeps)

        # Joint limits: override if specified
        if not env_config.use_default_joint_limits:
            joint_limit_ranges = msk_warp.joint_limit_ranges(self.m)
            joint_limits_path = os.path.join(self.curr_path, env_config.joint_limits_path)
            joint_limits = get_joint_limits(joint_limits_path, self.limit_id_lookup)
            for joint_limit in joint_limits:
                limit_id = joint_limit.limit_id
                joint_limit_ranges[limit_id][0] = joint_limit.lower
                joint_limit_ranges[limit_id][1] = joint_limit.upper

        # Contact model (Hunt-Crossley or MuJoCo)
        msk_warp.set_contact_type(self.m, env_config.contact_type)

        # Joint limit model (Exponential or MuJoCo)
        msk_warp.set_limit_type(self.m, env_config.limit_type)

        # Load exponential-spring force curve parameters
        if env_config.limit_type == msk_warp.LimitType.EXPONENTIAL:
            msk_warp.use_exponential_limit(self.m)
            exp_limit_forces = msk_warp.exp_limit_forces(self.m)
            exp_limit_shapes = msk_warp.exp_limit_shapes(self.m)
            limit_curves_path = os.path.join(self.curr_path, env_config.limit_force_curves_path)
            limit_curves = get_exp_limit_curves(limit_curves_path, self.limit_id_lookup)
            for limit_curve in limit_curves:
                limit_id = limit_curve.limit_id
                exp_limit_forces[limit_id][0] = limit_curve.limit_force[0]
                exp_limit_forces[limit_id][1] = limit_curve.limit_force[1]
                exp_limit_shapes[limit_id][0] = limit_curve.shape_param[0]
                exp_limit_shapes[limit_id][1] = limit_curve.shape_param[1]

        # MuJoCo-parameters
        msk_warp.set_solref(self.m, env_config.solref)

        # Newton solver faster for GPU, CG for CPU
        if self.device.type == "cuda":
            msk_warp.set_solver_type(self.m, msk_warp.SolverType.NEWTON)
        else:
            msk_warp.set_solver_type(self.m, msk_warp.SolverType.CG)

        # Integrator type
        msk_warp.set_integrator_type(self.m, env_config.integrator)

        # Toggle drag forces
        msk_warp.set_drag_enabled(self.m, env_config.enable_drag)

        msk_warp.reinitialize_model(self.m, self.d)

    def _setup_cuda_graphs(self):
        if self.cuda_graph:
            assert torch.cuda.is_available()
            with wp.ScopedCapture() as capture:
                msk_warp.step_to(self.m, self.d, self.delta_t, self.delta_t_sim)
            self.step_graph = capture.graph
            with wp.ScopedCapture() as capture:
                msk_warp.reset(self.m, self.d)
            self.reset_graph = capture.graph
            with wp.ScopedCapture() as capture:
                msk_warp.fk(self.m, self.d)
            self.fk_graph = capture.graph
        return

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            render: bool,
            cuda_graph: bool
    ):
        self.debug = True
        self.num_worlds = num_envs
        self.device = device

        # Load model
        self.curr_path = os.path.abspath(os.path.dirname(__file__))
        self.model_path = os.path.join(self.curr_path, env_config.model_path)
        function_path = os.path.join(
            self.curr_path, env_config.muscle_function_path) if env_config.use_function_based_path else None
        load_result = msk_warp.load_model(self.model_path, num_envs, function_path)
        self.m, self.d = load_result.model, load_result.data
        self.body_id_lookup = load_result.body_id_lookup
        self.dof_id_lookup = load_result.dof_id_lookup
        self.limit_id_lookup = load_result.limit_id_lookup
        self.muscle_id_lookup = load_result.muscle_id_lookup
        self.actuator_id_lookup = load_result.actuator_id_lookup
        self.visuals = load_result.visuals
        self._setup_model(env_config)

        # Model properties
        self.num_qpos = msk_warp.get_num_qpos(self.m)
        self.num_dofs = msk_warp.get_num_dofs(self.m)
        self.num_muscles = msk_warp.get_num_muscles(self.m)
        self.num_actuators = msk_warp.get_num_actuators(self.m)
        # [num_envs, num_bodies]
        self.body_mass = msk_warp.body_mass(self.m)
        self.total_mass = self.body_mass.sum()

        # [num_envs]
        self.time = msk_warp.time(self.d)
        # [num_envs, num_muscles]
        self.muscle_activations = msk_warp.muscle_activations(self.d)
        self.muscle_excitations = msk_warp.muscle_excitations(self.d)
        self.muscle_fiber_lengths = msk_warp.muscle_fiber_lengths(self.d)
        self.muscle_fiber_velocities = msk_warp.muscle_fiber_velocities(self.d)
        # [num_envs, num_actuators]
        self.actuator_activations = msk_warp.actuator_activations(self.d)
        self.actuator_excitations = msk_warp.actuator_excitations(self.d)
        # [num_envs, num_bodies, 3]
        self.body_positions = msk_warp.body_com_positions(self.d)
        # [num_envs, num_bodies, 4] (w, x, y, z)
        self.body_rotations = msk_warp.body_rotations(self.d)
        # [num_envs, num_bodies, 6] (ang, lin)
        self.body_velocities = msk_warp.body_com_velocities(self.d)
        # [num_envs, num_qpos]
        self.joint_positions = msk_warp.joint_positions(self.d)
        # [num_envs, num_dofs]
        self.joint_velocities = msk_warp.joint_velocities(self.d)

        # [num_envs, ]
        self.grf = msk_warp.grf(self.d)
        # [num_envs, num_joint_limits]
        self.limit_torques = msk_warp.limit_torques(self.d)
        # [num_envs, num_dofs]
        self.joint_moments = msk_warp.joint_moments(self.d)

        # [num_envs, num_visuals, 3]
        self.visual_positions = msk_warp.get_visual_positions(self.d)
        # [num_envs, num_visuals, 4]
        self.visual_rotations = msk_warp.get_visual_rotations(self.d)

        # RL Environment metadata
        self.action_range = (-1.0, 1.0)
        self.max_episode_duration = env_config.max_episode_duration
        self.delta_t = env_config.delta_t
        self.delta_t_sim = env_config.delta_t_sim
        self.reset_tensor = torch.zeros(
            (num_envs, 1), dtype=torch.float32, device=device)

        # Starting position, load from file
        start_pose_path = os.path.join(self.curr_path, env_config.starting_pose)
        q, qv = parse_starting_pose(start_pose_path, self.dof_id_lookup, self.num_qpos, self.num_dofs)
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
        self.swap_lr = env_config.swap_lr
        if self.swap_lr:
            self.swap_lr_data = get_swap_left_right_data(self.m, self.body_id_lookup)
        else:
            self.swap_lr_data = []

        # Starting muscle activations
        if env_config.use_prescribed_starting_activations:
            start_activations_path = os.path.join(self.curr_path, env_config.starting_activations)
            start_activations = parse_starting_activations(
                start_activations_path, self.muscle_id_lookup, env_config.default_activation)
            self.start_activations = torch.tensor(start_activations, device=device).unsqueeze(0).repeat(num_envs, 1)
        else:
            self.start_activations = torch.ones((num_envs, self.num_muscles),
                                                device=device) * env_config.default_activation

        # Rewards storage
        self.reward_dict = {}
        self.reward_lambdas = env_config.reward_lambdas

        self.render = render
        if render:
            self.renderer = msk_warp.create_renderer(
                load_result=load_result,
                renderer_type=msk_warp.RendererType.TILED,
                draw_visuals=True,
                draw_colliders=False,
                draw_muscles=True
            )
            if self.renderer.viewer_type == msk_warp.RendererType.TILED:
                self.renderer.setup_tiled_renderer(list(range(min(num_envs, 4))))

        # CUDA Graphs
        self.cuda_graph = cuda_graph
        self._setup_cuda_graphs()

        # Pre-compute useful body IDs and offsets
        self.root_id = self.lookup_body_id("pelvis")
        self.torso_id = self.lookup_body_id("torso")
        self.root_pos = self.body_positions[:, self.root_id]
        self.torso_pos = self.body_positions[:, self.torso_id]
        self.torso_rot = self.body_rotations[:, self.torso_id]
        head_offset = build_axis(axis=UP_IDX, scale=0.215)
        self.head_offset = torch.tensor(head_offset, device=self.device).unsqueeze(0).repeat(num_envs, 1)

        # Finally, set initial pose
        reset_ind = torch.ones_like(self.reset_tensor, dtype=torch.bool)
        self.set_start_pose(reset_ind.ravel())
        return

    def set_start_pose(self, reset_mask: torch.Tensor) -> None:
        """
        Re-noise the starting pose and velocity for envs where reset_mask is 1
        Note: takes effect on next reset or init
        """
        # Repeat for all envs
        q = self.start_pose_base.repeat(self.num_worlds, 1)
        qv = self.start_velocity_base.repeat(self.num_worlds, 1)

        # Noise starting pose
        if self.noise_start:
            q[7:] += torch.randn_like(q[7:]) * self.q_noise
            qv += torch.randn_like(qv) * self.qv_noise

        if self.swap_lr:
            # Determine which envs to swap the starting pose
            swap_mask = (torch.rand(self.num_worlds, device=q.device) > 0.5)
            q_old, qv_old = q.clone(), qv.clone()
            for swap_pair in self.swap_lr_data:
                rq, lq, nq = swap_pair.start_qpos_r, swap_pair.start_qpos_l, swap_pair.num_qpos
                q[swap_mask, rq:rq + nq] = q_old[swap_mask, lq:lq + nq]
                q[swap_mask, lq:lq + nq] = q_old[swap_mask, rq:rq + nq]

                rv, lv, nv = swap_pair.start_dof_r, swap_pair.start_dof_l, swap_pair.num_dof
                qv[swap_mask, rv:rv + nv] = qv_old[swap_mask, lv:lv + nv]
                qv[swap_mask, lv:lv + nv] = qv_old[swap_mask, rv:rv + nv]

        self.start_pose[reset_mask, :] = q[reset_mask, :]
        self.start_velocity[reset_mask, :] = qv[reset_mask, :]
        return

    def num_obs(self) -> int:
        return self._get_obs().shape[1]

    def num_actions(self) -> int:
        return self._get_actions().shape[1]

    def get_time(self) -> torch.Tensor:
        return msk_warp.time(self.d)

    def _upon_reset(self, reset_mask: torch.Tensor) -> None:
        """ Hook for additional reset behavior in subclasses """
        return

    def _handle_reset(self):
        msk_warp.set_reset(self.d, self.reset_tensor)  # Inform sim of resets

        # Reset time, starting pose, muscle and actuator activations
        reset_mask = self.reset_tensor.squeeze(-1).bool()
        if reset_mask.any():
            self.time[reset_mask] = 0.0  # should this be in env itself?
            self.joint_positions[reset_mask, :] = self.start_pose[reset_mask, :]
            self.joint_velocities[reset_mask, :] = self.start_velocity[
                                                   reset_mask, :]
            self.muscle_activations[reset_mask, :] = self.start_activations[reset_mask, :]
            self.actuator_activations[reset_mask, :] = 0.5

        # Reset sim
        if self.cuda_graph:
            wp.capture_launch(self.reset_graph)
            wp.synchronize()
        else:
            msk_warp.reset(self.m, self.d)

        self.reset_tensor.fill_(0.0)
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
        truncated = (self.get_time() >= self.max_episode_duration).float()
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

    def step(self, actions):
        # Sim step
        self._set_actions(actions)
        if self.cuda_graph:
            wp.capture_launch(self.step_graph)
            wp.synchronize()
        else:
            msk_warp.step_to(self.m, self.d, self.delta_t, self.delta_t_sim)

        # Only compute reward dict once per step
        self._compute_raw_reward_dict()

        final_obs = self._get_obs()
        rew = self.get_rewards()
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # Reset any worlds that are done
        done = torch.clamp(terminated + truncated, 0.0, 1.0).unsqueeze(-1)
        self.reset_tensor.copy_(done)
        self.set_start_pose(done.squeeze(-1).bool())
        self._upon_reset(done.squeeze(-1).bool())
        self._handle_reset()

        # Training requires the observation *after* the reset
        obs = self._get_obs()
        # Return raw reward terms for logging
        info = {
            "final_observation": final_obs,
            "raw_rewards": self.reward_dict,
            "scaled_rewards": self.get_scaled_reward_dict(),
        }

        assert not torch.isnan(obs).any(), "Observations contain NaN!"
        assert not torch.isnan(actions).any(), "Actions contain NaN!"

        if self.render and hasattr(self.renderer, 'meshes') and len(
                self.renderer.meshes) > 0:
            self.renderer.render(self.m, self.d)

        return obs, rew, terminated, truncated, info

    def reset(self):
        self.reset_tensor.fill_(1.0)
        self._upon_reset(self.reset_tensor.squeeze(-1).bool())
        self._handle_reset()
        obs = self._get_obs()
        return obs

    def fk(self):
        """ Forward kinematics only (only position dependent) """
        if self.cuda_graph:
            wp.capture_launch(self.fk_graph)
        else:
            msk_warp.fk(self.m, self.d)

    def lookup_body_id(self, body_name: str) -> int:
        return self.body_id_lookup[body_name]
