import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from msk_envs.train.common import module_state, save_checkpoint
from msk_envs.train.off_policy import OffPolicyTrainer
from msk_envs.train.nets.crossq_critic import CrossQCritic
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.nets.optimizer import make_optimizer
from msk_envs.train.nets.qflex_networks import ReferencePolicy, QFlexActor, VelocityField
from .config import QFlexConfig


class QFlexTrainer(OffPolicyTrainer):
    """
    QFlex: a CrossQ critic, an entropy-regularized Gaussian reference policy, and a velocity field trained by flow
    matching toward Q-gradient-ascent targets. The actor samples from the reference and integrates the flow.
    """

    def __init__(self, cfg: QFlexConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        device = self.device
        n_obs, n_act = self.n_obs, self.n_act
        action_scale = torch.ones(n_act, device=device) * (self.action_high - self.action_low) / 2.0
        action_bias = torch.zeros(n_act, device=device) + (self.action_high + self.action_low) / 2.0

        self.critic = CrossQCritic(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=cfg.critic_hidden_dim,
            num_q_networks=cfg.num_q_networks,
            device=device,
        )
        self.reference = ReferencePolicy(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=cfg.actor_hidden_dim,
            log_std_max=cfg.log_std_max,
            log_std_min=cfg.log_std_min,
            device=device,
            action_scale=action_scale,
            action_bias=action_bias,
        )
        self.log_alpha = torch.tensor([math.log(cfg.alpha_init)], requires_grad=True, device=device)
        self.velocity_field = VelocityField(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dim=cfg.velocity_hidden_dim,
            device=device
        )
        self.actor = QFlexActor(
            reference=self.reference,
            velocity_field=self.velocity_field,
            num_timesteps=cfg.num_flow_steps,
            device=device,
            action_low=self.action_low,
            action_high=self.action_high,
        ).to(device)

        def network_optimizer(module):
            return make_optimizer(module, lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay,
                                  optimizer=cfg.optimizer, fused=False)

        self.q_optimizer = network_optimizer(self.critic)
        self.ref_optimizer = network_optimizer(self.reference)
        self.vel_optimizer = network_optimizer(self.velocity_field)
        # no decay of the entropy temperature
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=cfg.learning_rate, betas=cfg.betas, weight_decay=0.0)

        if cfg.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        self.rb = self._make_replay_buffer()
        self.target_entropy = -cfg.target_entropy_ratio * n_act
        self.max_update = 2.0 * math.sqrt(n_act)  # bound on the Q-gradient-ascent step

        self._update_critic = self._critic_step
        self._update_reference = self._reference_step
        if cfg.compile:
            self._update_critic = torch.compile(self._update_critic, mode=cfg.compile_mode,
                                                backend=cfg.compile_backend)
            self._update_reference = torch.compile(self._update_reference, mode=cfg.compile_mode,
                                                   backend=cfg.compile_backend)

    def explore(self, norm_obs, dones):
        if self.global_step < self.cfg.learning_starts:
            return torch.rand((self.cfg.num_envs, self.n_act), device=self.device) * 2.0 - 1.0
        self.actor.eval()
        return self.actor.explore(norm_obs)

    def _critic_step(self, data):
        cfg = self.cfg
        self.critic.train()
        self.reference.eval()

        observations = data["observations"]
        next_observations = data["next"]["observations"]
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        with torch.no_grad():
            next_actions, next_log_probs = self.reference.get_actions_and_log_probs(next_observations)
            discount = cfg.gamma ** data["next"]["effective_n_steps"]
            alpha = self.log_alpha.exp()

        qf_current, qf_next = self.critic.forward_joint(observations, actions, next_observations, next_actions)
        qf1, qf2 = qf_current.squeeze(-1)
        qf1_next_value, qf2_next_value = qf_next.detach().squeeze(-1)
        # Soft target: entropy bonus keeps the reference policy from collapsing.
        qf_next_value = torch.minimum(qf1_next_value, qf2_next_value) - alpha * next_log_probs
        qf_next_target = rewards + bootstrap * discount * qf_next_value

        qf1_loss = F.mse_loss(qf1, qf_next_target)
        qf2_loss = F.mse_loss(qf2, qf_next_target)
        q_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        return {
            "q_loss": q_loss.detach().item(),
            "q_min": qf_next_value.min().detach().item(),
            "q_max": qf_next_value.max().detach().item(),
            "alpha": alpha.detach().item(),
        }

    def _reference_step(self, data):
        self.critic.eval()
        self.reference.train()

        actions, log_probs = self.reference.get_actions_and_log_probs(data["observations"])
        alpha = self.log_alpha.exp()
        q_value = self.critic.q_value(data["observations"], actions)
        ref_loss = (alpha.detach() * log_probs - q_value).mean()

        self.ref_optimizer.zero_grad(set_to_none=True)
        ref_loss.backward()
        self.ref_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return {
            "reference_loss": ref_loss.detach().item(),
            "reference_entropy": -log_probs.mean().detach().item(),
            "alpha_loss": alpha_loss.detach().item(),
        }

    def _velocity_step(self, data):
        cfg = self.cfg
        self.critic.eval()
        self.reference.eval()
        self.velocity_field.train()

        next_obs = data["next"]["observations"]
        with torch.no_grad():
            action_init, _ = self.reference.get_actions_and_log_probs(next_obs)

        # Bounded Q-gradient ascent in action space builds the flow target.
        y = action_init.clone()
        for _ in range(cfg.grad_step_num):
            y = y.detach().requires_grad_(True)
            q = self.critic.q_value(next_obs, y)
            grad_y = torch.autograd.grad(q.sum(), y)[0]
            grad_norm = grad_y.norm(dim=1, keepdim=True)
            step = torch.minimum(
                torch.full_like(grad_norm, cfg.grad_step_size),
                self.max_update / (grad_norm + 1e-6),
            )
            y = (y + step * grad_y).detach()
        action_flow_update = y

        # Conditional flow matching toward the straight line init -> updated
        t = torch.rand(next_obs.shape[0], 1, device=next_obs.device)
        temp_action = (1.0 - t) * action_init + t * action_flow_update
        temp_velocity = self.velocity_field(next_obs, temp_action, t)
        target_velocity = action_flow_update - action_init
        vel_loss = ((temp_velocity - target_velocity) ** 2).mean()

        self.vel_optimizer.zero_grad(set_to_none=True)
        vel_loss.backward()
        self.vel_optimizer.step()

        q_init = self.critic.q_value(next_obs, action_init).detach()
        q_flow = self.critic.q_value(next_obs, action_flow_update).detach()
        return {
            "velocity_loss": vel_loss.detach().item(),
            "q_init": q_init.mean().detach().item(),
            "q_flow_update": q_flow.mean().detach().item(),
            "q_flow_diff": (q_flow - q_init).mean().detach().item(),
            "target_velocity_norm": target_velocity.norm(dim=1).mean().detach().item(),
        }

    def update_critic(self, data):
        return self._update_critic(data)

    def update_actor(self, data):
        return {**self._update_reference(data), **self._velocity_step(data)}

    def save(self, path):
        state = {
            "actor_state_dict": module_state(self.actor),
            "critic_state_dict": module_state(self.critic),
            "obs_normalizer_state": module_state(self.obs_normalizer),
        }
        save_checkpoint(path, self.global_step, self.cfg, state)

    def eval_policy(self):
        self.actor.eval()
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            # Normalize with the training statistics, without updating them with eval observations
            return lambda obs: self.actor.act(self.obs_normalizer(obs, update=False))
        return lambda obs: self.actor.act(self.obs_normalizer(obs))
