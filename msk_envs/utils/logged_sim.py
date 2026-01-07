""" Provides a wrapper around an MSKEnv to log simulation data """
from msk_envs.envs.env_base import MSKEnv
from msk_envs.utils.checkpoint_parser import parse_frame, add_reference_visuals
from msk_envs.utils.animation_builder import create_animation_json
from msk_envs.utils.pdf_log_builder import create_pdf_output

import os
import json
import torch


class LoggedSim:
    def __init__(
            self,
            envs: MSKEnv,
            max_episode_length: int,
            device: torch.device,
    ):
        self.envs = envs
        self.worlds_to_save = list(range(envs.num_worlds))
        n_worlds = envs.num_worlds
        n_worlds_to_save = len(self.worlds_to_save)
        assert n_worlds_to_save <= n_worlds
        assert min(self.worlds_to_save) >= 0
        assert max(self.worlds_to_save) < n_worlds

        # Build storage for things to track
        max_ep_len = max_episode_length
        self.finished = torch.zeros((n_worlds,),
                                    dtype=torch.bool, device=device)
        self.rewards = torch.zeros((n_worlds_to_save, max_ep_len),
                                   dtype=torch.float32, device=device)
        self.frame_data = [[] for _ in range(n_worlds_to_save)]
        self.episode_length = torch.zeros((n_worlds,),
                                          dtype=torch.int32, device=device)

        self.n_worlds = n_worlds
        self.n_worlds_to_save = n_worlds_to_save
        self.max_episode_length = max_ep_len
        self.curr_step = 0
        self.device = device

        # Build lookups helpers: qpos idx to name
        qpos_idx_to_name = {v[0]: k for k, v in self.envs.dof_id_lookup.items()}
        self.qpos_idx_to_name = qpos_idx_to_name
        # dof idx to name
        dof_idx_to_name = {v[1]: k for k, v in self.envs.dof_id_lookup.items()}
        self.dof_idx_to_name = dof_idx_to_name
        # muscle idx to name
        self.muscle_idx_to_name = {v: k for k, v in self.envs.muscle_id_lookup.items()}
        # actuator idx to name
        self.actuator_idx_to_name = {v: k for k, v in self.envs.actuator_id_lookup.items()}

    def add_to_log(self):
        # Track rewards
        rew = self.envs.get_rewards()
        ind_not_finished = torch.where(self.finished == 0)[0]
        if len(ind_not_finished) == 0:
            return
        self.rewards[ind_not_finished, self.curr_step] = rew[ind_not_finished]
        self.episode_length[ind_not_finished] += 1

        times = self.envs.get_time()
        reward_dict = self.envs.get_scaled_reward_dict()

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
            reward_data = {k: v[idx_world].item() for k, v in reward_dict.items()}
            frame = parse_frame(
                m=self.envs.m,
                d=self.envs.d,
                qpos_idx_to_name=self.qpos_idx_to_name,
                dof_idx_to_name=self.dof_idx_to_name,
                muscle_idx_to_name=self.muscle_idx_to_name,
                actuation_idx_to_name=self.actuator_idx_to_name,
                visual_load_results=self.envs.visuals,
                world_id=idx_world,
                frame_time=frame_time,
                reward_data=reward_data,
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

            self.frame_data[i].append(frame)
        self.curr_step += 1
        return

    def step(self, actions: torch.Tensor):
        if self.finished.all():
            return True, None

        obs, rew, terminated, truncated, _ = self.envs.step(actions)
        done = (terminated + truncated).bool()
        self.finished = self.finished | done

        self.add_to_log()
        return False, obs

    def reset(self):
        self.curr_step = 0
        self.finished[:] = 0
        self.rewards[:] = 0
        self.episode_length[:] = 0
        self.frame_data = [[] for _ in range(self.n_worlds_to_save)]
        obs = self.envs.reset()
        return obs

    def get_rewards_mean(self):
        return self.rewards.sum(dim=1).float().mean()

    def get_episode_length_mean(self):
        return self.episode_length.float().mean()

    def save_frame_data(self, out_folder: str, base_filename: str):
        """ Save the raw frame data as json files """
        os.makedirs(out_folder, exist_ok=True)
        for idx_world in self.worlds_to_save:
            out_file = os.path.join(out_folder, f"{base_filename}_{idx_world}.json")
            frame_data = self.frame_data[idx_world]
            frame_data = [frame.to_dict() for frame in frame_data]
            with open(out_file, 'w') as f:
                json.dump(frame_data, f)
            print("Saved frame data to", out_file)
        return

    def save_animation(self, out_folder: str, base_filename):
        """ Create the animation-ready json files """
        for idx_world in self.worlds_to_save:
            out_file = os.path.join(out_folder, f"{base_filename}_{idx_world}.json")
            frame_data = self.frame_data[idx_world]
            create_animation_json(frame_data, out_file)
            print("Saved animation to", out_file)
        return

    def save_analytics(self, out_folder: str, base_filename: str):
        """ Create PDF analytics files """
        for i in range(self.n_worlds_to_save):
            idx_world = self.worlds_to_save[i]
            out_file = os.path.join(out_folder, f"{base_filename}_{idx_world}.pdf")
            create_pdf_output(self.frame_data[i], out_file)
            print("Saved analytics to", out_file)
        return
