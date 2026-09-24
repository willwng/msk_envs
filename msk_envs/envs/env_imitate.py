import torch
import bolt
import warp as wp

from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.parse_mot import parse_mot
from msk_envs.utils.reward_lib import has_fallen


class ImitateEnv(MSKEnv):
    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            cuda_graph: bool,
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            cuda_graph=cuda_graph
        )

        # TODO: load motion proper
        m, d = self.load_result.model, self.load_result.data
        motion_file = "msk_envs/msk_models/gt/motions/maxVerticalJump_3step.mot"
        motion = parse_mot(motion_file, self.load_result, in_degrees=False)
        motion = torch.tensor(motion, device=device, dtype=torch.float32)
        ref_time, ref_q = motion[:, 0].contiguous(), motion[:, 1:]  # (N_frames), (N_frames, N_qpos)
        n_frames = len(ref_q)
        self.n_frames = n_frames

        # Adjust height TODO proper
        ref_q[:, self.qpos_id_lookup["pelvis_ty"]] -= 0.05

        # Reference velocity via finite differences.
        # this is dq/dt in qpos space (N_qpos), we need to convert to u space
        ref_vel_dq = torch.gradient(ref_q, spacing=(ref_time,), dim=0)[0]  # (N_frames, N_qpos)
        ref_u = torch.zeros((n_frames, self.num_dofs), device=device)  # (N_frames, N_u)
        for i, dq in enumerate(ref_vel_dq):
            dq = wp.from_torch(dq.to(torch.float32).repeat(num_envs, 1))
            ref_u[i, :] = bolt.map_dq_to_u(m, d, dq)[0]
        self.ref_time, self.ref_q, self.ref_u = ref_time, ref_q, ref_u

        # Run FK on the reference motion to finish processing
        body_positions, body_rotations = [], []
        vis_positions, vis_rotations = [], []
        marker_positions = []
        joint_positions = bolt.joint_positions(d)
        for i, q in enumerate(ref_q):
            joint_positions[0, :] = q
            self.fk()
            body_positions.append(self.body_positions[0, :, :].clone())
            body_rotations.append(self.body_rotations[0, :, :].clone())
            vis_positions.append(self.visual_positions[0, :, :].clone())
            vis_rotations.append(self.visual_rotations[0, :, :].clone())
            marker_positions.append(self.marker_positions[0, :, :].clone())
        # [N_frames, N_bodies, 3|4]
        self.ref_bp = torch.stack(body_positions, dim=0).to(device)
        self.ref_br = torch.stack(body_rotations, dim=0).to(device)
        # [N_frames, N_visuals, 3|4]
        self.ref_vp = torch.stack(vis_positions, dim=0).to(device)
        self.ref_vr = torch.stack(vis_rotations, dim=0).to(device)
        # [N_frames, N_markers, 3|4]
        self.ref_mp = torch.stack(marker_positions, dim=0).to(device)

        # Update RL starting position, episode duration
        self.starting_state_helper.update_starting_pose(self.ref_q[0, :])
        self.starting_state_helper.update_starting_velocity(self.ref_u[0, :])
        self.max_episode_duration = self.ref_time[-1]

        # Target frame index per env
        self.ref_frame = torch.zeros(num_envs, dtype=torch.long, device=device)
        return

    def _set_target(self):
        frame_indices = torch.searchsorted(self.ref_time, self.time)
        frame_indices = torch.clamp(frame_indices, 0, self.n_frames - 1)
        self.ref_frame.copy_(frame_indices)
        return

    def _pre_step(self):
        self._set_target()
        return

    def _get_ref(self):
        ref_q = self.ref_q[self.ref_frame]
        ref_u = self.ref_u[self.ref_frame]
        ref_bp = self.ref_bp[self.ref_frame]
        ref_mp = self.ref_mp[self.ref_frame]
        return ref_q, ref_u, ref_bp, ref_mp

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         - Current time
         - Muscle activations, fiber lengths
         - Actuator activations
         - Joint positions (q)
         - Joint velocities (qv)
         - Reference targets
        """
        ref_q, ref_u, ref_bp, ref_mp = self._get_ref()

        obs = torch.cat([
            self.time.unsqueeze(1),
            self.muscle_activations,
            self.muscle_norm_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            ref_q.view(self.num_worlds, -1),
            ref_u.view(self.num_worlds, -1),
            ref_bp.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        ref_q, ref_u, ref_bp, ref_mp = self._get_ref()

        q_err = torch.abs(self.joint_positions - ref_q)
        u_err = torch.abs(self.joint_velocities - ref_u)
        bp_err = torch.linalg.norm(self.body_positions - ref_bp, dim=-1)
        mp_err = torch.linalg.norm(self.marker_positions - ref_mp, dim=-1)

        q_err_avg = torch.mean(q_err, dim=-1)
        u_err_avg = torch.mean(u_err, dim=-1)
        bp_err_avg = torch.mean(bp_err, dim=-1)
        mp_err_avg = torch.mean(mp_err, dim=-1)

        rew_track_q = torch.exp(-0.1 * q_err_avg)
        rew_track_u = torch.exp(-0.01 * u_err_avg)
        rew_track_bp = torch.exp(-3.0 * bp_err_avg)
        rew_track_mp = torch.exp(-3.0 * mp_err_avg)

        self.reward_dict = {
            "rew_track_q": rew_track_q,
            "rew_track_u": rew_track_u,
            "rew_track_bp": rew_track_bp,
            "rew_track_mp": rew_track_mp,
        }

    def _get_terminated(self):
        root_diff = torch.linalg.norm(self.root_pos - self.ref_bp[self.ref_frame, self.root_id], dim=-1)
        root_too_far = root_diff > 0.5
        terminated = (root_too_far).float()
        return terminated.detach()

    def get_references(self):
        ref_visual_positions = self.ref_vp[self.ref_frame, :]
        ref_visual_rotations = self.ref_vr[self.ref_frame, :]
        ref_q = self.ref_q[self.ref_frame]
        return ref_visual_positions, ref_visual_rotations, ref_q
