import torch
import torch.nn as nn
import tqdm
from loguru import logger

from msk_envs.envs.base import MSKEnv
from msk_envs.train.common import RunPaths, evaluate_policy, make_logging_helper, module_state, save_checkpoint
from msk_envs.train.nets.normalizers import EmpiricalNormalization, RewardNormalizer
from msk_envs.train.nets.optimizer import make_optimizer
from msk_envs.train.nets.ppo_networks import PPOActor, PPOCritic
from msk_envs.train.utils import TensorAverageMeterDict, mark_step
from .config import PPOConfig
from .rollout_storage import RolloutStorage


class PPOTrainer:

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
            hidden_dims=cfg.actor_hidden_dims,
            init_std=cfg.init_std,
            learnable_std=cfg.learnable_std,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
            device=device,
        )
        self.critic = PPOCritic(
            n_obs=n_obs,
            hidden_dims=cfg.critic_hidden_dims,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
            device=device,
        )
        # Separate actor and critic optimizers: Muon for the weight matrices and AdamW for the rest, or AdamW for all
        optimizer_kwargs = dict(
            lr=cfg.learning_rate, betas=tuple(cfg.adam_betas), weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer, eps=cfg.adam_eps, fused=False, muon_momentum=cfg.muon_momentum,
            muon_nesterov=cfg.muon_nesterov, muon_ns_steps=cfg.muon_ns_steps,
            muon_adjust_lr_fn=cfg.muon_adjust_lr_fn)
        self.actor_optimizer = make_optimizer(self.actor, **optimizer_kwargs)
        self.critic_optimizer = make_optimizer(self.critic, **optimizer_kwargs)
        self.lr = cfg.learning_rate

        # Obs and reward normalizers
        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device, eps=cfg.obs_norm_eps) \
            if cfg.obs_normalization else nn.Identity()
        self.reward_normalizer = RewardNormalizer(gamma=cfg.gamma, device=device, g_max=cfg.reward_norm_g_max) \
            if cfg.reward_normalization else None

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
        x = self.obs_normalizer(x, update=update)
        if self.cfg.obs_norm_clip is not None:
            x = x.clamp(-self.cfg.obs_norm_clip, self.cfg.obs_norm_clip)
        return x

    def compute_losses(self, data):
        cfg = self.cfg
        log_probs, entropy, mu, sigma = self.actor.evaluate_actions(data["observations"], data["actions"])
        values = self.critic(data["observations"])

        # Clipped surrogate objective (advantages are normalized over the rollout)
        advantages = data["advantages"]
        ratio = torch.exp(log_probs - data["old_log_probs"])
        surrogate = ratio * advantages
        surrogate_clipped = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantages
        surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

        # Clipped value loss
        old_values, returns = data["values"], data["returns"]
        value_clipped = old_values + (values - old_values).clamp(-cfg.value_clip_param, cfg.value_clip_param)
        value_loss = 0.5 * torch.max((values - returns).pow(2), (value_clipped - returns).pow(2)).mean()

        entropy = entropy.mean()
        actor_loss = surrogate_loss - cfg.entropy_coef * entropy
        critic_loss = cfg.value_loss_coef * value_loss

        # Closed-form KL between the rollout and current Gaussians, for logging and the adaptive schedule
        with torch.no_grad():
            old_mu, old_sigma = data["old_mu"], data["old_sigma"]
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
                - 0.5,
                dim=-1,
            ).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.clip_param).float().mean()
        return actor_loss, critic_loss, surrogate_loss, value_loss, entropy, kl, clip_fraction

    def update(self, data):
        actor_loss, critic_loss, surrogate_loss, value_loss, entropy, kl, clip_fraction = self._compute_losses(data)

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        max_norm = self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf")
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return {
            "surrogate_loss": surrogate_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "kl": kl,
            "clip_fraction": clip_fraction,
            "actor_grad_norm": actor_grad_norm.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
        }

    def set_learning_rate(self, lr: float) -> None:
        self.lr = lr
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = lr

    def adapt_learning_rate(self, mean_kl: float) -> None:
        """ rsl_rl-style adaptive schedule: lower the learning rate when the minibatch KL is above twice the target,
        raise it when below half """
        cfg = self.cfg
        if mean_kl > cfg.desired_kl * 2.0:
            self.set_learning_rate(max(cfg.min_learning_rate, self.lr / 1.5))
        elif 0.0 < mean_kl < cfg.desired_kl / 2.0:
            self.set_learning_rate(min(cfg.max_learning_rate, self.lr * 1.5))

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
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        save_checkpoint(path, self.global_step, self.cfg, state)

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        else:  # saved with a single optimizer over the actor and critic
            logger.warning("Checkpoint has no per-network optimizer state; starting the optimizers fresh")
        return checkpoint.get("global_step", 0)

    def train(self) -> None:
        cfg, envs, storage = self.cfg, self.envs, self.storage
        if cfg.checkpoint_path:
            self.global_step = self.load_checkpoint(cfg.checkpoint_path)

        obs = envs.reset()
        pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=self.global_step)
        while self.global_step < cfg.num_learning_iterations:
            mark_step()
            if cfg.schedule == "linear":  # decays to min_lr_ratio * learning_rate at the end of training
                frac = max(0.0, 1.0 - self.global_step / cfg.num_learning_iterations)
                self.set_learning_rate(cfg.learning_rate * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * frac))

            # Rollout collection
            storage.clear()
            raw_reward_sum = torch.zeros((), device=self.device)
            with self.logging_helper.record_collection_time():
                self.actor.eval()
                for _ in range(cfg.num_steps_per_env):
                    with torch.no_grad():
                        norm_obs = self.normalize(obs, update=True)  # updates running stats while training
                        actions, log_probs, mu, sigma = self.actor.act(norm_obs)
                        values = self.critic(norm_obs)

                    next_obs, rewards, terminated, truncations, info = envs.step(actions)
                    dones = (terminated + truncations).bool()
                    with torch.no_grad():
                        train_rewards = rewards
                        if self.reward_normalizer is not None:
                            self.reward_normalizer.update_stats(rewards, dones.float())
                            train_rewards = self.reward_normalizer(rewards)

                        # Time-limit truncation is not a true terminal: bootstrap from the pre-reset obs
                        if cfg.bootstrap_truncations:
                            timeouts = truncations.bool() & ~terminated.bool()
                            if timeouts.any():
                                idx = timeouts.nonzero(as_tuple=True)[0]
                                final_obs = self.normalize(info["final_observation"][idx], update=False)
                                train_rewards = train_rewards.index_add(0, idx, cfg.gamma * self.critic(final_obs))

                    storage.add(obs=norm_obs, actions=actions, rewards=train_rewards, dones=dones,
                                values=values, log_probs=log_probs, mus=mu, sigmas=sigma)
                    self.logging_helper.update_episode_stats(rewards, dones)
                    raw_reward_sum += rewards.mean()
                    obs = next_obs

                # Bootstrap value for the final state, then GAE
                with torch.no_grad():
                    last_values = self.critic(self.normalize(obs, update=False))
                storage.compute_returns(last_values, gamma=cfg.gamma, lam=cfg.lam)
                raw_rewards_dict = {f"{name}_raw": t.mean() for name, t in info["raw_rewards"].items()}

            # Learning
            self.actor.train()
            lr_used = self.lr
            with self.logging_helper.record_learn_time():
                for data in storage.mini_batch_generator(num_mini_batches=cfg.num_mini_batches,
                                                         num_epochs=cfg.num_learning_epochs):
                    metrics = self.update(data)
                    if cfg.schedule == "adaptive":
                        self.adapt_learning_rate(metrics["kl"].item())
                    self.training_metrics.add(metrics)

            step = self.global_step
            if step % cfg.logging_interval == 0 and step > 0:
                loss_metrics = self.training_metrics.get_metrics_and_clear()
                loss_metrics["env_rewards"] = (raw_reward_sum / cfg.num_steps_per_env).item()
                loss_metrics["learning_rate"] = lr_used
                loss_metrics["action_std"] = self.actor.log_std.exp().mean().item()
                # storage.rewards are normalized + bootstrapped; env_rewards above is the raw env reward
                loss_metrics["train_rewards"] = storage.rewards.mean().item()
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
