""" Provides a wrapper around an MSKEnv to log simulation data """
from msk_envs.envs.env_base import MSKEnv
from msk_envs.utils.frame_parser import parse_frame, add_reference_visuals, add_ext_forces_to_frame, add_target
from msk_envs.utils.animation_builder import create_animation_json
from msk_envs.utils.pdf_log_builder import create_pdf_output

import math
import bolt
import os
import gzip
import json
import numpy as np
import torch
import warp as wp


class LoggedSim:
    def __init__(
            self,
            envs: MSKEnv,
            device: torch.device,
            delta_t_log: float = 0.01,  # default record at 100Hz
    ):
        self.envs = envs
        self.worlds_to_save = list(range(envs.num_worlds))
        self.delta_t_log = delta_t_log
        n_worlds = envs.num_worlds
        n_worlds_to_save = len(self.worlds_to_save)
        assert n_worlds_to_save <= n_worlds
        assert min(self.worlds_to_save) >= 0
        assert max(self.worlds_to_save) < n_worlds

        # Build storage for things to track
        max_env_steps = int(envs.max_episode_duration / envs.delta_t) + 1
        self.finished = torch.zeros((n_worlds,), dtype=torch.bool, device=device)
        self.rewards = torch.zeros((n_worlds_to_save, max_env_steps), dtype=torch.float32, device=device)
        self.episode_length = torch.zeros((n_worlds,), dtype=torch.int32, device=device)
        self.reward_data = [[] for _ in range(n_worlds_to_save)]
        self.max_env_steps = max_env_steps

        # Frame data: logged every delta_t_log seconds
        self.is_adaptive = self.envs.is_adaptive
        if self.is_adaptive:
            self.num_sim_steps_per_log = 1
        else:
            self.num_sim_steps_per_log = math.ceil(self.delta_t_log / self.envs.delta_t_sim)
        self.frame_data = [[] for _ in range(n_worlds_to_save)]

        self.n_worlds = n_worlds
        self.n_worlds_to_save = n_worlds_to_save
        self.curr_policy_step = 0
        self.curr_sim_step = 0
        self.device = device

        # Build lookups helpers: qpos idx to name
        self.body_idx_to_name = {v: k for k, v in self.envs.body_id_lookup.items()}
        # qpos idx to name
        self.qpos_idx_to_name = {v: k for k, v in self.envs.qpos_id_lookup.items()}
        # dof idx to name
        self.dof_idx_to_name = {v: k for k, v in self.envs.dof_id_lookup.items()}
        # muscle idx to name
        self.muscle_idx_to_name = {v: k for k, v in self.envs.muscle_id_lookup.items()}
        # actuator idx to name
        self.actuator_idx_to_name = {v: k for k, v in self.envs.actuator_id_lookup.items()}
        # collider idx to name
        self.collider_idx_to_name = {v: k for k, v in self.envs.collider_id_lookup.items()}

        max_episode_duration = self.envs.max_episode_duration
        self.log_times = torch.arange(0, max_episode_duration + self.delta_t_log, self.delta_t_log, device=device)

    def add_to_rl_log(self):
        # Track rewards
        ind_not_finished = torch.where(self.finished == 0)[0]
        if len(ind_not_finished) == 0:
            return
        # Total rewards, episode length
        rew = self.envs.get_rewards()
        self.rewards[ind_not_finished, self.curr_policy_step] = rew[ind_not_finished]
        self.episode_length[ind_not_finished] += 1

        # Reward breakdown
        reward_dict = self.envs.get_scaled_reward_dict()
        for i in range(self.n_worlds_to_save):
            idx_world = self.worlds_to_save[i]
            if self.finished[idx_world]:
                continue
            reward_info = {k: float(v[idx_world].item()) for k, v in reward_dict.items()}
            self.reward_data[i].append(reward_info)

        self.curr_policy_step += 1
        return

    def add_to_log(self):
        times = self.envs.time
        scene_settings = self.envs.scene_settings()

        # If we are using the ImitateEnv, add reference visuals
        add_reference, ref_joint_angles = False, None
        if hasattr(self.envs, "get_reference_visuals") and hasattr(self.envs, "get_reference_times") \
                and hasattr(self.envs, "get_reference_joint_angles"):
            ref_joint_angles = self.envs.get_reference_joint_angles()
            ref_visuals_pos, ref_visuals_rot = self.envs.get_reference_visuals()
            ref_times = self.envs.get_reference_times()
            add_reference = True

        for i in range(len(self.worlds_to_save)):
            idx_world = self.worlds_to_save[i]
            if self.finished[idx_world]:
                continue
            frame_time = float(times[idx_world].item())
            frame = parse_frame(
                m=self.envs.m,
                d=self.envs.d,
                body_name_to_idx=self.envs.body_id_lookup,
                qpos_idx_to_name=self.qpos_idx_to_name,
                qpos_name_to_idx=self.envs.qpos_id_lookup,
                dof_idx_to_name=self.dof_idx_to_name,
                limit_id_lookup=self.envs.limit_id_lookup,
                muscle_idx_to_name=self.muscle_idx_to_name,
                actuation_idx_to_name=self.actuator_idx_to_name,
                collider_idx_to_name=self.collider_idx_to_name,
                visual_load_results=self.envs.visuals,
                world_id=idx_world,
                frame_time=frame_time,
                scene_settings=scene_settings,
                ref_joint_angles=ref_joint_angles,
            )

            if add_reference:
                # find time in ref_times closest to frame_time
                time_diffs = torch.abs(ref_times - frame_time)
                closest_idx = torch.argmin(time_diffs).item()
                add_reference_visuals(
                    frame,
                    ref_visuals_pos[closest_idx],
                    ref_visuals_rot[closest_idx],
                )

            add_ext_forces_to_frame(
                frame,
                self.envs.m,
                self.envs.d,
                idx_world
            )

            self.frame_data[i].append(frame)
        return

    def perform_post(self):
        """ Run post-processing and analytics task graphs """
        if self.envs.cuda_graph:
            wp.capture_launch(self.envs.post_graph)
            wp.capture_launch(self.envs.analytics_graph)
        else:
            bolt.compute_muscle_passive_forces(self.envs.m, self.envs.d)
            bolt.compute_muscle_moments(self.envs.m, self.envs.d)
            bolt.compute_net_joint_moments(self.envs.m, self.envs.d)
        wp.synchronize()
        return

    def check_post_step_log(self):
        """ Check if we need to log, if so, run post and add to log """
        if (self.curr_sim_step % self.num_sim_steps_per_log) == 0:
            self.perform_post()
            self.add_to_log()
        self.curr_sim_step += 1
        return

    def perform_step(self, dt_step: float, force_no_log: bool = False):
        bolt.increment_next_time(self.envs.m, self.envs.d, dt_step)
        if self.envs.cuda_graph:
            wp.capture_launch(self.envs.step_graph)
        else:
            bolt.step(self.envs.m, self.envs.d)

        if not force_no_log:
            self.check_post_step_log()
        return

    def step(self, actions: torch.Tensor):
        if self.finished.all():
            return True, None

        # We're overriding typical env step to allow for logging

        # Start with the pre-step
        self.envs.pre_sim_step(actions)

        # We need to report at the correct logging event times
        #  For fixed time-stepping, we assume [delta_t_sim] is small and just check after every sim step
        #  For adaptive time-stepping, we integrate to all logging events within this env step
        if not self.is_adaptive:  # fixed time-stepping
            for _ in range(self.envs.sim_steps_per_env_step):
                self.perform_step(self.envs.delta_t_sim)
            self.perform_post()
        else:  # adaptive time-stepping
            # we should only look at worlds that aren't finished, they should all be at the same time
            idx_not_finished = torch.where(self.finished == 0)[0][0].item()
            # find all logging events in (curr_time, curr_time + delta_t)
            delta_t = self.envs.delta_t
            curr_time = self.envs.time[idx_not_finished].item()
            end_time = curr_time + delta_t

            # Determine which logging times are in (curr_time, end_time]
            logging_times = self.log_times
            times_in_interval = (logging_times > curr_time) & (logging_times <= end_time)
            logging_times_in_interval = logging_times[times_in_interval]
            # integrate to each logging time
            for i in range(len(logging_times_in_interval)):
                curr_time = self.envs.time[idx_not_finished].item()
                delta_t = logging_times_in_interval[i] - curr_time
                self.perform_step(float(delta_t))
            # check if we need to add an extra step to reach end_time, don't log
            if len(logging_times_in_interval) == 0 or not np.isclose(logging_times_in_interval[-1].item(), end_time):
                curr_time = self.envs.time[idx_not_finished].item()
                dt_step = end_time - curr_time
                self.perform_step(float(dt_step), force_no_log=True)

            self.perform_post()

        # Now the policy step is done
        obs, rew, terminated, truncated, info = self.envs.rl_step()
        done = (terminated + truncated).bool()
        self.finished = self.finished | done
        self.add_to_rl_log()
        return self.finished, obs

    def reset(self):
        self.curr_policy_step = 0
        self.curr_sim_step = 0
        self.finished[:] = 0
        self.rewards[:] = 0
        self.episode_length[:] = 0
        self.reward_data = [[] for _ in range(self.n_worlds_to_save)]
        self.frame_data = [[] for _ in range(self.n_worlds_to_save)]
        obs = self.envs.reset()

        self.add_to_log()
        return obs

    def get_rewards_mean(self):
        return self.rewards.sum(dim=1).float().mean()

    def get_episode_length_mean(self):
        return self.episode_length.float().mean()

    def save_frame_data(self, out_folder: str, base_filename: str, use_gzip: bool = False):
        """ Save the raw frame data into a JSON file for each world """
        os.makedirs(out_folder, exist_ok=True)
        for idx_world in self.worlds_to_save:
            frame_data = self.frame_data[idx_world]
            frame_data = [frame.to_dict() for frame in frame_data]
            # Output file path
            if use_gzip:
                out_file = os.path.join(out_folder, f"{base_filename}_{idx_world}.json.gz")
            else:
                out_file = os.path.join(out_folder, f"{base_filename}_{idx_world}.json")

            if use_gzip:
                with gzip.open(out_file, "wt") as f:
                    json.dump(frame_data, f, indent=4)
            else:
                with open(out_file, "w") as f:
                    json.dump(frame_data, f, indent=4)
            print("Saved frame data to", out_file)
        return

    def save_animation(self, out_folder: str, base_filename: str, use_gzip: bool = False):
        """ Create the animation-ready JSON file for each world """
        for idx_world in self.worlds_to_save:
            frame_data = self.frame_data[idx_world]

            if use_gzip:
                out_file = os.path.join(out_folder, f"{base_filename}_{idx_world}.json.gz")
            else:
                out_file = os.path.join(out_folder, f"{base_filename}_{idx_world}.json")

            create_animation_json(frame_data, out_file, use_gzip)
            print("Saved animation to", out_file)
        return

    def save_analytics(self, out_folder: str, base_filename: str):
        """ Create PDF analytics files """
        for i in range(self.n_worlds_to_save):
            idx_world = self.worlds_to_save[i]
            out_file = os.path.join(out_folder, f"{base_filename}_{idx_world}.pdf")
            create_pdf_output(self.frame_data[i], self.reward_data[i], out_file)
            print("Saved analytics to", out_file)
        return
