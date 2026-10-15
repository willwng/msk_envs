from typing import Optional

import torch
import torch.nn as nn


class DeterministicPolicy(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_envs: int,
            hidden_dim: int,
            std_min: float,
            std_max: float,
            use_layer_norm: bool = True,
            use_gsde: bool = False,
            gsde_steps: int = 10,
            device: torch.device = None,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_envs = num_envs
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        # This will be overridden in subclasses if needed
        self.setup_network()

        # Exploration noise scales per environment
        noise_scales = (torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min)
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.register_buffer("noise", torch.zeros(num_envs, n_act, device=device))
        self.register_buffer("gsde_step_count", torch.zeros(1, device=device, dtype=torch.int32))
        self.use_gsde = use_gsde
        self.gsde_steps = gsde_steps

    def setup_network(self) -> None:
        """Setup the network architecture. Can be overridden by subclasses."""
        self._setup_network_with_input_dim(self.n_obs)

    def _setup_network_with_input_dim(self, input_dim: int) -> None:
        """Setup network with specific input dimension."""
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, device=self.device),
            nn.LayerNorm(self.hidden_dim, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, device=self.device),
            nn.LayerNorm(self.hidden_dim // 2, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4, device=self.device),
            nn.LayerNorm(self.hidden_dim // 4, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device),
            nn.Tanh(),
        )
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x_net = self.net(obs)
        x_head = self.fc_head(x_net)
        action = self.fc_mu(x_head)
        return action

    def _sample_new_noise(self, dones: Optional[torch.Tensor]):
        """ Generate new exploration noise """
        # Generate new noise scales for done environments
        if dones is not None and dones.sum() > 0:
            new_scales = (torch.rand(self.n_envs, 1, device=self.device) *
                          (self.std_max - self.std_min) + self.std_min)
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(torch.where(dones_view, new_scales, self.noise_scales))

        # Sample noise every gsde_steps or every step
        if self.use_gsde:
            resample_noise = (self.gsde_step_count % self.gsde_steps) == 0
            new_noise = torch.randn_like(self.noise) * self.noise_scales
            self.noise.copy_(torch.where(resample_noise, new_noise, self.noise))
            self.gsde_step_count += 1
        else:
            self.noise.copy_(torch.randn_like(self.noise) * self.noise_scales)
        return

    def explore(self, obs: torch.Tensor, dones: Optional[torch.Tensor]) -> torch.Tensor:
        self._sample_new_noise(dones)
        act = self(obs)
        return act + self.noise
