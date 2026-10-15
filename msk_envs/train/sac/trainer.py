import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler

from msk_envs.train.common import clip_grad_norm, module_state, save_checkpoint, soft_update
from msk_envs.train.off_policy import OffPolicyTrainer
from msk_envs.train.nets.distributional_critic import DistributionalCritic
from msk_envs.train.inference import policy_from_checkpoint
from msk_envs.train.nets.gaussian_policy import GaussianPolicy
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.nets.optimizer import make_optimizer
from .config import SACConfig


class SACTrainer(OffPolicyTrainer):
    """ FastSAC: squashed Gaussian actor with a learned entropy temperature, distributional critic ensemble """

    # Also run the final step, so the last log/checkpoint lands exactly on num_learning_iterations
    include_final_step = True

    def __init__(self, cfg: SACConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        device = self.device
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        if cfg.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=self.n_obs, device=device)

        action_scale = torch.ones(self.n_act, device=device) * (self.action_high - self.action_low) / 2.0
        action_bias = torch.zeros(self.n_act, device=device) + (self.action_high + self.action_low) / 2.0
        self.actor = GaussianPolicy(
            n_obs=self.n_obs,
            n_act=self.n_act,
            num_envs=cfg.num_envs,
            hidden_dim=cfg.actor_hidden_dim,
            log_std_max=cfg.log_std_max,
            log_std_min=cfg.log_std_min,
            use_tanh=cfg.use_tanh,
            use_layer_norm=cfg.use_layer_norm,
            device=device,
            action_scale=action_scale,
            action_bias=action_bias,
        )
        critic_kwargs = dict(
            n_obs=self.n_obs,
            n_act=self.n_act,
            num_atoms=cfg.num_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            hidden_dim=cfg.critic_hidden_dim,
            use_layer_norm=cfg.use_layer_norm,
            num_q_networks=cfg.num_q_networks,
            device=device,
        )
        self.qnet = DistributionalCritic(**critic_kwargs)
        self.qnet_target = DistributionalCritic(**critic_kwargs)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.q_optimizer = make_optimizer(
            model=self.qnet,
            lr=cfg.critic_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )
        self.actor_optimizer = make_optimizer(
            model=self.actor,
            lr=cfg.actor_learning_rate,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )

        self.target_entropy = -self.n_act * cfg.target_entropy_ratio
        self.log_alpha = torch.tensor([math.log(cfg.alpha_init)], requires_grad=True, device=device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=cfg.alpha_learning_rate, fused=True,
                                           betas=(0.9, 0.95))
        self.rb = self._make_replay_buffer()

        self._explore = self.actor.explore
        self._update_critic = self._critic_step
        self._update_actor = self._actor_step
        if cfg.compile:
            self._update_critic = torch.compile(self._update_critic)
            self._update_actor = torch.compile(self._update_actor)
            self._explore = torch.compile(self._explore)

        self.actor_metrics = {
            "actor_loss": torch.tensor(0.0, device=device),
            "actor_grad_norm": torch.tensor(0.0, device=device),
            "policy_entropy": torch.tensor(0.0, device=device),
            "action_std": torch.tensor(0.0, device=device),
        }

    def explore(self, norm_obs, dones):
        return self._explore(obs=norm_obs, dones=dones).float()

    def _critic_step(self, data):
        cfg = self.cfg
        with self.amp():
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = self.actor.get_actions_and_log_probs(next_observations)
                discount = cfg.gamma ** data["next"]["effective_n_steps"]

                target_distributions = self.qnet_target.projection(
                    next_observations,
                    next_state_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = self.qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = self.qnet(observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        self.q_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(qf_loss).backward()
        self.scaler.unscale_(self.q_optimizer)
        critic_grad_norm = clip_grad_norm(self.qnet.parameters(), cfg.max_grad_norm, self.device)
        self.scaler.step(self.q_optimizer)
        self.scaler.update()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if cfg.use_autotune:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            with self.amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.unscale_(self.alpha_optimizer)
            self.scaler.step(self.alpha_optimizer)
            self.scaler.update()

        return {
            "qf_loss": qf_loss.detach(),
            "qf_max": target_value_max.detach(),
            "qf_min": target_value_min.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "buffer_rewards": rewards.mean(),
            "alpha_loss": alpha_loss.detach(),
        }

    def _actor_step(self, data):
        with self.amp():
            observations = data["observations"]
            actions, log_probs = self.actor.get_actions_and_log_probs(observations)
            # For logging only
            with torch.no_grad():
                _, _, log_std = self.actor(observations)
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()

            q_outputs = self.qnet(observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = self.qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        actor_grad_norm = clip_grad_norm(self.actor.parameters(), self.cfg.max_grad_norm, self.device)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "policy_entropy": policy_entropy.detach(),
            "action_std": action_std.detach(),
        }

    def update_critic(self, data):
        return self._update_critic(data)

    def update_actor(self, data):
        return self._update_actor(data)

    def update_metrics(self):
        return {"alpha_value": self.log_alpha.exp().detach().mean()}

    def after_update(self):
        soft_update(self.qnet, self.qnet_target, self.cfg.tau)

    def save(self, path):
        state = {
            "actor_state_dict": module_state(self.actor),
            "qnet_state_dict": module_state(self.qnet),
            "qnet_target_state_dict": module_state(self.qnet_target),
            "log_alpha": self.log_alpha.detach().cpu(),
            "obs_normalizer_state": module_state(self.obs_normalizer),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "grad_scaler_state_dict": self.scaler.state_dict(),
            "iteration": int(self.global_step),
        }
        save_checkpoint(path, self.global_step, self.cfg, state, logging_helper=self.logging_helper)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.qnet.load_state_dict(checkpoint["qnet_state_dict"])
        self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        self.qnet_target.load_state_dict(checkpoint["qnet_target_state_dict"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        return checkpoint["global_step"]

    def eval_policy(self):
        # The inference policy for the current weights, built the same way as when loading a checkpoint
        return policy_from_checkpoint({
            "args": vars(self.cfg),
            "actor_state_dict": module_state(self.actor),
            "obs_normalizer_state": module_state(self.obs_normalizer),
        }).to(device=self.device)
