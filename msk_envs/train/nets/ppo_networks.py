""" Based on Holosoma """
import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "ELU",
        dropout_prob: float = 0.0,
        use_layer_norm: bool = False,
        device: torch.device = None,
) -> nn.Sequential:
    act_cls = getattr(nn, activation)
    layers: list[nn.Module] = []

    if len(hidden_dims) == 0:
        layers.append(nn.Linear(input_dim, output_dim, device=device))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(input_dim, hidden_dims[0], device=device))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dims[0], device=device))
    layers.append(act_cls())
    if dropout_prob > 0:
        layers.append(nn.Dropout(dropout_prob))

    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], device=device))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dims[i + 1], device=device))
        layers.append(act_cls())
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

    layers.append(nn.Linear(hidden_dims[-1], output_dim, device=device))
    return nn.Sequential(*layers)


class PPOActor(nn.Module):
    """Gaussian policy with a raw, state-independent std parameter """

    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dims: list[int],
            init_noise_std: float,
            activation: str = "ELU",
            dropout_prob: float = 0.0,
            use_layer_norm: bool = False,
            min_noise_std: float | None = None,
            min_mean_noise_std: float | None = None,
            device: torch.device = None,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.device = device
        self.min_noise_std = min_noise_std
        self.min_mean_noise_std = min_mean_noise_std

        self.actor = build_mlp(
            n_obs, hidden_dims, n_act, activation, dropout_prob, use_layer_norm, device
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(n_act, device=device))

    def _clamped_std(self) -> torch.Tensor:
        if self.min_noise_std:
            return torch.clamp(self.std, min=self.min_noise_std)
        elif self.min_mean_noise_std:
            current_mean = self.std.mean()
            if current_mean < self.min_mean_noise_std:
                scale_up = self.min_mean_noise_std / (current_mean + 1e-6)
                return self.std * scale_up
            return self.std
        return self.std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (distribution mean)."""
        return self.actor(obs)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self._clamped_std()
        return Normal(mean, mean * 0.0 + std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """Sample an action; returns (action, log_prob, mean, std)."""
        dist = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist.mean, dist.stddev

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Re-evaluate stored actions; returns (log_prob, entropy, mean, std)."""
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, dist.mean, dist.stddev

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        # return self.actor(obs)
        return self.act(obs)[0]


class PPOCritic(nn.Module):
    """State-value function V(s)."""

    def __init__(
            self,
            n_obs: int,
            hidden_dims: list[int],
            activation: str = "ELU",
            dropout_prob: float = 0.0,
            use_layer_norm: bool = False,
            device: torch.device = None,
    ):
        super().__init__()
        self.critic = build_mlp(
            n_obs, hidden_dims, 1, activation, dropout_prob, use_layer_norm, device
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
