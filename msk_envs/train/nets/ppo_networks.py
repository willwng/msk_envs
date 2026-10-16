import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "SiLU",
        use_layer_norm: bool = True,
        output_gain: float = 0.01,
        device: torch.device = None,
) -> nn.Sequential:
    """
    MLP of Linear -> (LayerNorm) -> activation blocks and a linear output. Weights are orthogonally initialized
    (gain sqrt(2) for hidden layers, [output_gain] for the output) with zero biases.
    """
    act_cls = getattr(nn, activation)
    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_dims]
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        linear = nn.Linear(d_in, d_out, device=device)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if use_layer_norm:
            layers.append(nn.LayerNorm(d_out, device=device))
        layers.append(act_cls())

    output = nn.Linear(dims[-1], output_dim, device=device)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class PPOActor(nn.Module):
    """ Gaussian policy with a state-independent log std (actions are not squashed) """

    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dims: list[int],
            init_std: float,
            learnable_std: bool = True,
            activation: str = "SiLU",
            use_layer_norm: bool = True,
            device: torch.device = None,
    ):
        super().__init__()
        self.actor = build_mlp(n_obs, hidden_dims, n_act, activation, use_layer_norm, output_gain=0.01, device=device)
        log_std = torch.full((n_act,), math.log(init_std), device=device)
        if learnable_std:
            self.log_std = nn.Parameter(log_std)
        else:
            self.register_buffer("log_std", log_std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """ Deterministic action (distribution mean) """
        return self.actor(obs)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Sample an action; returns (action, log_prob) """
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Re-evaluate stored actions; returns (log_prob, entropy) """
        dist = self.distribution(obs)
        return dist.log_prob(actions).sum(dim=-1), dist.entropy().sum(dim=-1)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """ Sampled action, as used for evaluation """
        return self.act(obs)[0]


class PPOCritic(nn.Module):
    """ State-value function V(s) """

    def __init__(
            self,
            n_obs: int,
            hidden_dims: list[int],
            activation: str = "SiLU",
            use_layer_norm: bool = True,
            device: torch.device = None,
    ):
        super().__init__()
        self.critic = build_mlp(n_obs, hidden_dims, 1, activation, use_layer_norm, output_gain=0.01, device=device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
