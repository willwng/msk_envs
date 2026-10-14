import torch
import torch.nn as nn
import tqdm
from loguru import logger

from msk_envs.envs.base import MSKEnv
from msk_envs.train.common import RunPaths, evaluate_policy, make_logging_helper, module_state, save_checkpoint
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.nets.optimizer import make_optimizer
from msk_envs.train.nets.ppo_networks import PPOActor, PPOCritic
from msk_envs.train.utils import TensorAverageMeterDict, mark_step
from .config import PPOConfig
from .rollout_storage import RolloutStorage


class PPOTrainer:
    """ PPO with clipped surrogate and value losses, GAE, and an optional KL-adaptive learning rate """

    def __init__(self, cfg: PPOConfig, envs: MSKEnv, eval_envs: MSKEnv, paths: RunPaths, device: torch.device):
        self.cfg = cfg
        self.envs = envs
        self.eval_envs = eval_envs
        self.paths = paths
        self.device = device

        self.logging_helper = make_logging_helper(
            paths, device, num_envs=cfg.num_envs,
            # samples collected between two logging calls
            num_steps_per_log=cfg.num_steps_per_env * cfg.logging_interval,
            num_learning_iterations=cfg.num_learning_iterations)
        self.training_metrics = TensorAverageMeterDict()

        n_obs = envs.num_obs()
        n_act = envs.num_actions()

        self.actor = PPOActor(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dims=cfg.hidden_dims,
            init_noise_std=cfg.init_noise_std,
            activation=cfg.activation,
            dropout_prob=cfg.dropout_prob,
            use_layer_norm=cfg.use_layer_norm,
            min_noise_std=cfg.min_noise_std,
            min_mean_noise_std=cfg.min_mean_noise_std,
            device=device,
        )
        self.critic = PPOCritic(
            n_obs=n_obs,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            dropout_prob=cfg.dropout_prob,
            use_layer_norm=cfg.use_layer_norm,
            device=device,
        )

        self.actor_optimizer = make_optimizer(
            model=self.actor, lr=cfg.actor_learning_rate, betas=(0.9, 0.95), weight_decay=0.0,
        )
        self.critic_optimizer = make_optimizer(
            model=self.critic, lr=cfg.critic_learning_rate, betas=(0.9, 0.95), weight_decay=0.0,
        )
        self.actor_lr = cfg.actor_learning_rate
        self.critic_lr = cfg.critic_learning_rate

        # Adaptive-LR bounds
        self.min_actor_lr = cfg.min_actor_learning_rate if cfg.min_actor_learning_rate is not None else 1e-5
        self.max_actor_lr = cfg.max_actor_learning_rate if cfg.max_actor_learning_rate is not None else 1e-2
        self.min_critic_lr = cfg.min_critic_learning_rate if cfg.min_critic_learning_rate is not None else 1e-5
        self.max_critic_lr = cfg.max_critic_learning_rate if cfg.max_critic_learning_rate is not None else 1e-2

        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device) if cfg.empirical_normalization \
            else nn.Identity()

        self.storage = RolloutStorage(
            num_envs=cfg.num_envs,
            num_steps=cfg.num_steps_per_env,
            n_obs=n_obs,
            n_act=n_act,
            device=device,
        )

        self._compute_losses = self.compute_losses
        if cfg.compile:
            self._compute_losses = torch.compile(self.compute_losses, mode=cfg.compile_mode,
                                                 backend=cfg.compile_backend)
        self.global_step = 0

    def normalize(self, x, update):
        if isinstance(self.obs_normalizer, nn.Identity):
            return x
        return self.obs_normalizer(x, update=update)

    def compute_losses(self, data):
        cfg = self.cfg
        old_values = data["values"]
        old_mu = data["old_mu"]
        old_sigma = data["old_sigma"]

        log_probs, entropy, mu, sigma = self.actor.evaluate_actions(data["observations"], data["actions"])
        values = self.critic(data["observations"])

        # Clipped surrogate objective
        advantages = data["advantages"]
        ratio = torch.exp(log_probs - data["old_log_probs"])
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Clipped value loss
        returns = data["returns"]
        value_clipped = old_values + (values - old_values).clamp(-cfg.clip_param, cfg.clip_param)
        value_losses = (values - returns).pow(2)
        value_losses_clipped = (value_clipped - returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        entropy_loss = entropy.mean()

        # Approximate KL for logging / adaptive schedule (Gaussian, closed form)
        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
                - 0.5,
                dim=-1,
            )
            kl_mean = kl.mean()

        actor_loss = surrogate_loss - cfg.entropy_coef * entropy_loss
        critic_loss = cfg.value_loss_coef * value_loss
        return actor_loss, critic_loss, surrogate_loss, value_loss, entropy_loss, kl_mean

    def adapt_learning_rate(self, kl_mean):
        """ rsl_rl-style KL adaptive learning-rate schedule """
        cfg = self.cfg
        if cfg.schedule != "adaptive":
            return
        if kl_mean > cfg.desired_kl * 2.0:
            self.actor_lr = max(self.min_actor_lr, self.actor_lr / 1.5)
            self.critic_lr = max(self.min_critic_lr, self.critic_lr / 1.5)
        elif 0.0 < kl_mean < cfg.desired_kl / 2.0:
            self.actor_lr = min(self.max_actor_lr, self.actor_lr * 1.5)
            self.critic_lr = min(self.max_critic_lr, self.critic_lr * 1.5)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.critic_lr

    def update(self, data):
        actor_loss, critic_loss, surrogate_loss, value_loss, entropy_loss, kl_mean = self._compute_losses(data)

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        if self.cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "surrogate_loss": surrogate_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "kl": kl_mean.detach(),
        }

    def evaluate(self) -> tuple[float, float]:
        self.actor.eval()
        result = evaluate_policy(
            self.eval_envs, lambda obs: self.actor.act_inference(self.normalize(obs, update=False)),
            self.paths, self.global_step, self.device)
        self.actor.train()
        return result

    def save(self, path: str) -> None:
        state = {
            "actor_state_dict": module_state(self.actor),
            "critic_state_dict": module_state(self.critic),
            "obs_normalizer_state": module_state(self.obs_normalizer),
        }
        save_checkpoint(path, self.global_step, self.cfg, state)

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if checkpoint.get("obs_normalizer_state") is not None:
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        return checkpoint.get("global_step", 0)

    def train(self) -> None:
        cfg, envs, storage = self.cfg, self.envs, self.storage
        if cfg.checkpoint_path:
            self.global_step = self.load_checkpoint(cfg.checkpoint_path)

        obs = envs.reset()
        pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=self.global_step)
        while self.global_step < cfg.num_learning_iterations:
            mark_step()

            # Rollout collection
            storage.clear()
            with self.logging_helper.record_collection_time():
                self.actor.eval()
                for _ in range(cfg.num_steps_per_env):
                    with torch.no_grad():
                        norm_obs = self.normalize(obs, update=True)  # updates running stats while training
                        actions, log_probs, mu, sigma = self.actor.act(norm_obs)
                        values = self.critic(norm_obs)

                    next_obs, rewards, terminated, truncations, info = envs.step(actions)
                    dones = (terminated + truncations).bool()

                    storage.add(
                        obs=norm_obs, actions=actions, rewards=rewards, dones=dones,
                        values=values, log_probs=log_probs, mus=mu, sigmas=sigma,
                    )
                    self.logging_helper.update_episode_stats(rewards, dones)
                    obs = next_obs

                # Bootstrap value for the final state, then GAE
                with torch.no_grad():
                    last_values = self.critic(self.normalize(obs, update=False))
                storage.compute_returns(last_values, gamma=cfg.gamma, lam=cfg.lam)
                raw_rewards_dict = {f"{name}_raw": t.mean() for name, t in info["raw_rewards"].items()}

            # Learning
            self.actor.train()
            with self.logging_helper.record_learn_time():
                for data in storage.mini_batch_generator(num_mini_batches=cfg.num_mini_batches,
                                                         num_epochs=cfg.num_learning_epochs):
                    metrics = self.update(data)
                    self.adapt_learning_rate(metrics["kl"])
                    self.training_metrics.add(metrics)

            step = self.global_step
            if step % cfg.logging_interval == 0 and step > 0:
                loss_metrics = self.training_metrics.get_metrics_and_clear()
                loss_metrics["env_rewards"] = storage.rewards.mean().item()
                loss_metrics["actor_lr"] = self.actor_lr
                loss_metrics["critic_lr"] = self.critic_lr
                extra_log_dicts = {
                    "raw_rewards": raw_rewards_dict,
                    "additional_metrics": envs.additional_metrics(),
                }
                self.logging_helper.post_epoch_logging(it=step, loss_dict=loss_metrics,
                                                       extra_log_dicts=extra_log_dicts)

            if cfg.save_interval > 0 and step > 0 and step % cfg.save_interval == 0:
                logger.info(f"Saving model at global step {step}")
                self.save(self.paths.checkpoint(step))

            if step % cfg.eval_freq == 0 and step > 0:
                logger.info(f"Evaluating at global step {step}")
                self.evaluate()

            self.global_step += 1
            pbar.update(1)
