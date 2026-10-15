import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler

from msk_envs.train.common import clip_grad_norm, module_state, save_checkpoint, soft_update
from msk_envs.train.off_policy import OffPolicyTrainer
from msk_envs.train.inference import policy_from_checkpoint
from msk_envs.train.nets.deterministic_policy import DeterministicPolicy
from msk_envs.train.nets.distributional_critic import DistributionalCritic
from msk_envs.train.nets.normalizers import EmpiricalNormalization, RewardNormalizer
from msk_envs.train.nets.optimizer import make_optimizer
from .config import TD3Config


class TD3Trainer(OffPolicyTrainer):
    """ FastTD3: deterministic actor with exploration noise, distributional twin critic """


    def __init__(self, cfg: TD3Config, **kwargs):
        super().__init__(cfg, **kwargs)
        device = self.device
        self.scaler = GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        if cfg.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=self.n_obs, device=device)
        if cfg.reward_normalization:
            self.reward_normalizer = RewardNormalizer(
                gamma=cfg.gamma,
                device=device,
                g_max=min(abs(cfg.v_min), abs(cfg.v_max)),
            )

        self.actor = DeterministicPolicy(
            n_obs=self.n_obs,
            n_act=self.n_act,
            num_envs=cfg.num_envs,
            hidden_dim=cfg.actor_hidden_dim,
            std_min=cfg.std_min,
            std_max=cfg.std_max,
            use_layer_norm=cfg.use_layer_norm,
            device=device,
            use_gsde=cfg.use_gsde,
            gsde_steps=cfg.gsde_steps,
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
        self.q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.q_optimizer,
            T_max=cfg.num_learning_iterations,
            eta_min=cfg.critic_learning_rate_end,
        )
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=cfg.num_learning_iterations,
            eta_min=cfg.actor_learning_rate_end,
        )
        self.rb = self._make_replay_buffer()

        self._explore = self.actor.explore
        self._update_critic = self._critic_step
        self._update_actor = self._actor_step
        self._update_reward_stats = self.reward_normalizer.update_stats if cfg.reward_normalization else None
        if cfg.compile:
            compile_mode, backend = cfg.compile_mode, cfg.compile_backend
            self._update_critic = torch.compile(self._update_critic, mode=compile_mode, backend=backend)
            self._update_actor = torch.compile(self._update_actor, mode=compile_mode, backend=backend)
            self._explore = torch.compile(self._explore, mode=None, backend=backend)
            if cfg.reward_normalization:
                self._update_reward_stats = torch.compile(self._update_reward_stats, mode=None, backend=backend)

        self.actor_metrics = {
            "actor_loss": torch.tensor(0.0, device=device),
            "actor_grad_norm": torch.tensor(0.0, device=device),
        }

    def explore(self, norm_obs, dones):
        return self._explore(obs=norm_obs, dones=dones)

    def after_env_step(self, rewards, prev_dones):
        if self.cfg.reward_normalization:
            # Discounted-return estimate, restarted for worlds whose episode ended on the previous step
            if prev_dones is None:
                prev_dones = torch.zeros_like(rewards, dtype=torch.bool)
            self._update_reward_stats(rewards, prev_dones.float())

    def _critic_step(self, data):
        cfg = self.cfg
        with self.amp():
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            if cfg.reward_normalization:
                rewards = self.reward_normalizer(rewards)
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                clipped_noise = torch.randn_like(actions)
                clipped_noise = clipped_noise.mul(cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
                next_state_actions = (self.actor(next_observations) + clipped_noise).clamp(
                    self.action_low, self.action_high)
                discount = cfg.gamma ** data["next"]["effective_n_steps"]

                qf1_next_target_projected, qf2_next_target_projected = self.qnet_target.projection(
                    next_observations,
                    next_state_actions,
                    rewards,
                    bootstrap,
                    discount,
                )
                qf1_next_target_value = self.qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = self.qnet_target.get_value(qf2_next_target_projected)
                if cfg.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1) < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = qf1_next_target_projected, qf2_next_target_projected

            qf1, qf2 = self.qnet(observations, actions)
            qf1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
            qf2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
            qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(qf_loss).backward()
        self.scaler.unscale_(self.q_optimizer)
        critic_grad_norm = clip_grad_norm(self.qnet.parameters(), cfg.max_grad_norm, self.device)
        self.scaler.step(self.q_optimizer)
        self.scaler.update()

        return {
            "qf_loss": qf_loss.mean(),
            "qf_max": qf1_next_target_value.max(),
            "qf_min": qf1_next_target_value.min(),
            "critic_grad_norm": critic_grad_norm.mean(),
            "buffer_rewards": rewards.mean(),
        }

    def _actor_step(self, data):
        with self.amp():
            ref_actions = self.actor(data["observations"])
            q_value = self.qnet.q_value(data["observations"], ref_actions, self.cfg.use_cdq)
            actor_loss = -q_value.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        actor_grad_norm = clip_grad_norm(self.actor.parameters(), self.cfg.max_grad_norm, self.device)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        return {
            "actor_loss": actor_loss.detach().item(),
            "actor_grad_norm": actor_grad_norm.detach().item(),
        }

    def update_critic(self, data):
        return self._update_critic(data)

    def update_actor(self, data):
        return self._update_actor(data)

    def after_update(self):
        soft_update(self.qnet, self.qnet_target, self.cfg.tau)

    def on_step_end(self):
        self.actor_scheduler.step()
        self.q_scheduler.step()

    def save(self, path):
        state = {
            "actor_state_dict": module_state(self.actor),
            "qnet_state_dict": module_state(self.qnet),
            "qnet_target_state_dict": module_state(self.qnet_target),
            "obs_normalizer_state": module_state(self.obs_normalizer),
        }
        if self.cfg.save_optimizer_state:
            state.update({
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
                "q_scheduler_state_dict": self.q_scheduler.state_dict(),
                "grad_scaler_state_dict": self.scaler.state_dict(),
            })
        save_checkpoint(path, self.global_step, self.cfg, state)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        self.qnet.load_state_dict(checkpoint["qnet_state_dict"])
        self.qnet_target.load_state_dict(checkpoint["qnet_target_state_dict"])

        # Optimizer and scheduler states are only saved with save_optimizer_state
        optional_states = {
            "actor_optimizer_state_dict": self.actor_optimizer,
            "q_optimizer_state_dict": self.q_optimizer,
            "actor_scheduler_state_dict": self.actor_scheduler,
            "q_scheduler_state_dict": self.q_scheduler,
            "grad_scaler_state_dict": self.scaler,
        }
        for key, target in optional_states.items():
            if key in checkpoint:
                target.load_state_dict(checkpoint[key])
        return checkpoint.get("global_step", 0)

    def eval_policy(self):
        # The inference policy for the current weights, built the same way as when loading a checkpoint
        return policy_from_checkpoint({
            "args": vars(self.cfg),
            "actor_state_dict": module_state(self.actor),
            "obs_normalizer_state": module_state(self.obs_normalizer),
        }).to(device=self.device)
