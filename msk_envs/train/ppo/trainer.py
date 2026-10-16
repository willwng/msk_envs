import torch
import torch.nn as nn
import tqdm
from loguru import logger

from msk_envs.envs.base import MSKEnv
from msk_envs.train.common import RunPaths, evaluate_policy, make_logging_helper, module_state, save_checkpoint
from msk_envs.train.nets.normalizers import EmpiricalNormalization, RewardNormalizer
from msk_envs.train.nets.ppo_networks import PPOActor, PPOCritic
from msk_envs.train.utils import TensorAverageMeterDict, mark_step
from .config import PPOConfig
from .rollout_storage import RolloutStorage


def make_optimizers(params: list[nn.Parameter], cfg: PPOConfig) -> list[torch.optim.Optimizer]:
    """ Muon for the weight matrices and AdamW for everything else (biases, LayerNorm, log std), or AdamW for all """
    adamw_kwargs = dict(lr=cfg.learning_rate, eps=1e-5, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(params, **adamw_kwargs)]
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon needs PyTorch >= 2.9; use --ppo-config.optimizer adamw")
    matrices = [p for p in params if p.ndim == 2]
    others = [p for p in params if p.ndim != 2]
    muon = torch.optim.Muon(
        matrices,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        momentum=cfg.muon_momentum,
        nesterov=cfg.muon_nesterov,
        ns_steps=cfg.muon_ns_steps,
        adjust_lr_fn="match_rms_adamw",  # update RMS comparable to AdamW, so both share the learning rate
    )
    return [muon, torch.optim.AdamW(others, **adamw_kwargs)]


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
        self.params = [*self.actor.parameters(), *self.critic.parameters()]
        self.optimizers = make_optimizers(self.params, cfg)
        self.lr = cfg.learning_rate

        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device) if cfg.obs_normalization \
            else nn.Identity()
        # Scale rewards by the running std of the discounted return (g_max=inf: no floor on the scale)
        self.reward_normalizer = RewardNormalizer(gamma=cfg.gamma, device=device, g_max=float("inf")) \
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
        return self.obs_normalizer(x, update=update)

    def compute_losses(self, data):
        cfg = self.cfg
        log_probs, entropy = self.actor.evaluate_actions(data["observations"], data["actions"])
        values = self.critic(data["observations"])

        # Clipped surrogate objective, with advantages normalized over the minibatch
        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = torch.exp(log_probs - data["old_log_probs"])
        surrogate = ratio * advantages
        surrogate_clipped = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantages
        surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

        # Clipped value loss
        old_values, returns = data["values"], data["returns"]
        value_clipped = old_values + (values - old_values).clamp(-cfg.value_clip_param, cfg.value_clip_param)
        value_loss = 0.5 * torch.max((values - returns).pow(2), (value_clipped - returns).pow(2)).mean()

        entropy = entropy.mean()
        total_loss = surrogate_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy

        # Sample-based approximate KL, for logging and the adaptive schedule
        with torch.no_grad():
            kl = (data["old_log_probs"] - log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.clip_param).float().mean()
        return total_loss, surrogate_loss, value_loss, entropy, kl, clip_fraction

    def update(self, data):
        total_loss, surrogate_loss, value_loss, entropy, kl, clip_fraction = self._compute_losses(data)

        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        max_norm = self.cfg.max_grad_norm if self.cfg.max_grad_norm > 0 else float("inf")
        grad_norm = nn.utils.clip_grad_norm_(self.params, max_norm)
        if torch.isfinite(grad_norm):  # skip updates with non-finite gradients
            for optimizer in self.optimizers:
                optimizer.step()

        return {
            "surrogate_loss": surrogate_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "kl": kl,
            "clip_fraction": clip_fraction,
            "grad_norm": grad_norm.detach(),
            "skipped_update": (~torch.isfinite(grad_norm)).float(),
        }

    def set_learning_rate(self, lr: float) -> None:
        self.lr = lr
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = lr

    def adapt_learning_rate(self, mean_kl: float) -> None:
        """ Adaptive schedule: lower the learning rate when the update's mean KL is above twice the target, raise it
        when below half """
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
            "optimizer_state_dicts": [optimizer.state_dict() for optimizer in self.optimizers],
        }
        save_checkpoint(path, self.global_step, self.cfg, state)

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        for optimizer, state in zip(self.optimizers, checkpoint["optimizer_state_dicts"]):
            optimizer.load_state_dict(state)
        return checkpoint.get("global_step", 0)

    def train(self) -> None:
        cfg, envs, storage = self.cfg, self.envs, self.storage
        if cfg.checkpoint_path:
            self.global_step = self.load_checkpoint(cfg.checkpoint_path)

        obs = envs.reset()
        pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=self.global_step)
        while self.global_step < cfg.num_learning_iterations:
            mark_step()
            if cfg.schedule == "linear":  # decays to 0 at the end of training
                self.set_learning_rate(cfg.learning_rate * (1.0 - self.global_step / cfg.num_learning_iterations))

            # Rollout collection
            storage.clear()
            raw_reward_sum = torch.zeros((), device=self.device)
            with self.logging_helper.record_collection_time():
                self.actor.eval()
                for _ in range(cfg.num_steps_per_env):
                    with torch.no_grad():
                        norm_obs = self.normalize(obs, update=True)  # updates running stats while training
                        actions, log_probs = self.actor.act(norm_obs)
                        values = self.critic(norm_obs)

                    next_obs, rewards, terminated, truncations, info = envs.step(actions)
                    dones = (terminated + truncations).bool()
                    train_rewards = rewards
                    if self.reward_normalizer is not None:
                        self.reward_normalizer.update_stats(rewards, dones.float())
                        train_rewards = self.reward_normalizer(rewards)

                    storage.add(obs=norm_obs, actions=actions, rewards=train_rewards, dones=dones,
                                values=values, log_probs=log_probs)
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
            kls = []
            with self.logging_helper.record_learn_time():
                for data in storage.mini_batch_generator(num_mini_batches=cfg.num_mini_batches,
                                                         num_epochs=cfg.num_learning_epochs):
                    metrics = self.update(data)
                    kls.append(metrics["kl"])
                    self.training_metrics.add(metrics)
            lr_used = self.lr
            if cfg.schedule == "adaptive":
                self.adapt_learning_rate(torch.stack(kls).mean().item())

            step = self.global_step
            if step % cfg.logging_interval == 0 and step > 0:
                loss_metrics = self.training_metrics.get_metrics_and_clear()
                loss_metrics["env_rewards"] = (raw_reward_sum / cfg.num_steps_per_env).item()
                loss_metrics["learning_rate"] = lr_used
                loss_metrics["action_std"] = self.actor.log_std.exp().mean().item()
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
