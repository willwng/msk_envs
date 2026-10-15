""" Training loop shared by the off-policy algorithms (TD3, SAC, QFlex) """
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn
import tqdm
from loguru import logger
from torch.amp import autocast

from msk_envs.envs.base import MSKEnv
from msk_envs.train.common import RunPaths, evaluate_policy, make_logging_helper, raw_reward_means
from msk_envs.train.nets.buffer import SimpleReplayBuffer, collect_experience, sample_and_prepare_batches
from msk_envs.train.utils import TensorAverageMeterDict, mark_step


class OffPolicyTrainer:
    """
    Training loop shared by the off-policy algorithms. Each env step:
     1. act with explore() and store the transition in the replay buffer
     2. once the buffer holds learning_starts steps, sample num_updates batches, and for each run update_critic(),
        update_actor() (every policy_frequency updates) and after_update()
     3. log, checkpoint and evaluate at their intervals

    Subclasses build their networks in __init__ (after calling super().__init__) and implement the hooks below.
    """

    include_final_step: bool = False
    """ Whether to also run step num_learning_iterations (so the last log/checkpoint lands exactly on it) """

    def __init__(self, cfg, envs: MSKEnv, eval_envs: MSKEnv, paths: RunPaths, device: torch.device):
        self.cfg = cfg
        self.envs = envs
        self.eval_envs = eval_envs
        self.paths = paths
        self.device = device

        self.amp_enabled = getattr(cfg, "amp", False)
        self.amp_dtype = torch.bfloat16 if getattr(cfg, "amp_dtype", "bf16") == "bf16" else torch.float16

        self.logging_helper = make_logging_helper(
            paths, device, num_envs=cfg.num_envs, num_steps_per_log=cfg.logging_interval,
            num_learning_iterations=cfg.num_learning_iterations)
        self.training_metrics = TensorAverageMeterDict()

        self.n_obs = envs.num_obs()
        self.n_act = envs.num_actions()
        self.action_low, self.action_high = envs.action_range
        self.obs_normalizer: nn.Module = nn.Identity()
        self.global_step = 0
        self.actor_metrics = {}
        """ Metrics of the latest actor update, logged on every update until the next one """

    def _make_replay_buffer(self) -> SimpleReplayBuffer:
        return SimpleReplayBuffer(
            n_env=self.cfg.num_envs,
            buffer_size=self.cfg.buffer_size,
            n_obs=self.n_obs,
            n_act=self.n_act,
            n_steps=self.cfg.num_steps,
            gamma=self.cfg.gamma,
            device=self.device,
        )

    @contextmanager
    def amp(self):
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            yield

    @torch._dynamo.disable
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_normalizer(obs)

    # --- Hooks ---

    def explore(self, norm_obs: torch.Tensor, dones: torch.Tensor | None) -> torch.Tensor:
        """ Actions to collect experience with, from normalized observations """
        raise NotImplementedError

    def after_env_step(self, rewards: torch.Tensor, prev_dones: torch.Tensor | None) -> None:
        """ Called after each env step, with the dones of the previous step """

    def update_critic(self, data) -> dict:
        raise NotImplementedError

    def update_actor(self, data) -> dict:
        raise NotImplementedError

    def update_metrics(self) -> dict:
        """ Extra metrics logged after every update """
        return {}

    def after_update(self) -> None:
        """ Called after each update (e.g. to update target networks) """

    def on_step_end(self) -> None:
        """ Called after each env step, once global_step has been incremented """

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> int:
        """ Restore training state from [path] and return its global step """
        raise NotImplementedError

    def eval_policy(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """ The policy to evaluate (raw observations -> actions) """
        raise NotImplementedError

    # --- Loop ---

    def _update(self, info: dict) -> None:
        cfg = self.cfg
        batch_size = max(cfg.batch_size // cfg.num_envs, 1)
        with self.logging_helper.record_learn_time():
            # Sample once, normalize once, split into updates
            prepared_batches = sample_and_prepare_batches(
                rb=self.rb, obs_normalizer=self.normalize_obs,
                num_updates=cfg.num_updates, target_batch_size=batch_size
            )
            for i, data in enumerate(prepared_batches):
                critic_metrics = self.update_critic(data)
                if cfg.num_updates > 1:
                    if i % cfg.policy_frequency == 1:
                        self.actor_metrics = self.update_actor(data)
                elif self.global_step % cfg.policy_frequency == 0:
                    self.actor_metrics = self.update_actor(data)

                self.training_metrics.add({**critic_metrics, **self.actor_metrics, **self.update_metrics()})
                self.after_update()

    def train(self) -> None:
        cfg, envs = self.cfg, self.envs
        if getattr(cfg, "checkpoint_path", ""):
            self.global_step = self.load_checkpoint(cfg.checkpoint_path)

        obs = envs.reset()
        dones = None
        pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=self.global_step)
        while self.global_step < cfg.num_learning_iterations or (
                self.include_final_step and self.global_step == cfg.num_learning_iterations):
            mark_step()
            with self.logging_helper.record_collection_time():
                with torch.no_grad(), self.amp():
                    actions = self.explore(self.normalize_obs(obs), dones)

                next_obs, rewards, terminated, truncations, info = envs.step(actions)
                collect_experience(
                    rb=self.rb, obs=obs, actions=actions, next_obs=next_obs, rewards=rewards,
                    terminated=terminated, truncations=truncations, info=info,
                )
                self.after_env_step(rewards, dones)

                dones = (terminated + truncations).bool()
                self.logging_helper.update_episode_stats(rewards, dones)
                obs = next_obs

            # Wait until the replay buffer has collected enough transitions before learning
            if self.rb.ptr >= cfg.learning_starts:
                self._update(info)
                step = self.global_step

                if step % cfg.logging_interval == 0:
                    loss_dict = self.training_metrics.get_metrics_and_clear()
                    loss_dict["env_rewards"] = rewards.mean().item()
                    extra_log_dicts = {
                        "raw_rewards": raw_reward_means(info),
                        "additional_metrics": envs.additional_metrics(),
                    }
                    self.logging_helper.post_epoch_logging(it=step, loss_dict=loss_dict,
                                                           extra_log_dicts=extra_log_dicts)

                if cfg.save_interval > 0 and step > 0 and step % cfg.save_interval == 0:
                    logger.info(f"Saving model at global step {step}")
                    self.save(self.paths.checkpoint(step))

                if step % cfg.eval_freq == 0:
                    logger.info(f"Evaluating at global step {step}")
                    evaluate_policy(self.eval_envs, self.eval_policy(), self.paths, step, self.device)

            if self.global_step >= cfg.num_learning_iterations:
                break
            self.global_step += 1
            self.on_step_end()
            pbar.update(1)
