import torch
import torch.nn as nn
import torch.nn.functional as F



class DistributionalQNetwork(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_atoms: int,
            v_min: float,
            v_max: float,
            hidden_dim: int,
            use_layer_norm: bool,
            device: torch.device,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )

        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )

        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        x = self.fc_head(x)
        return x

    def projection(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            bootstrap: torch.Tensor,
            discount: torch.Tensor,
            q_support: torch.Tensor,
            device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)

        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        # Additional safety check for indices
        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1

        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class DistributionalCritic(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_atoms: int,
            v_min: float,
            v_max: float,
            hidden_dim: int,
            use_layer_norm: bool,
            device: torch.device,
            num_q_networks: int = 2,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.n_act = n_act
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.num_q_networks = num_q_networks
        self.device = device

        # Setup Q-networks - this will be overridden in subclasses if needed
        assert num_q_networks >= 1, "Number of Q networks must be at least 1"
        self.setup_qnetworks()
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))
        return

    def setup_qnetworks(self) -> None:
        """Setup Q-networks. Can be overridden by subclasses."""
        self._setup_qnetworks_with_obs_dim(self.n_obs)

    def _setup_qnetworks_with_obs_dim(self, n_obs: int) -> None:
        """Setup Q-networks with specific observation dimension."""
        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    n_obs=n_obs,
                    n_act=self.n_act,
                    num_atoms=self.num_atoms,
                    v_min=self.v_min,
                    v_max=self.v_max,
                    hidden_dim=self.hidden_dim,
                    use_layer_norm=self.use_layer_norm,
                    device=self.device,
                )
                for _ in range(self.num_q_networks)
            ]
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """ Returns the distribution of Q(s,a) over atoms for each Q-network """
        x = obs
        outputs = [qnet(x, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def projection(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            bootstrap: torch.Tensor,
            discount: torch.Tensor,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        x = obs
        projections = [
            qnet.projection(
                x,
                actions,
                rewards,
                bootstrap,
                discount,
                self.q_support,
                self.q_support.device,
            )
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=-1)

    def q_value(self, obs: torch.Tensor, actions: torch.Tensor, use_cdq: bool) -> torch.Tensor:
        """ Returns Q(s,a) """
        qfs = self.forward(obs, actions)  # (num_q_networks, batch, num_atoms)
        q_values = self.get_value(F.softmax(qfs, dim=-1))  # (num_q_networks, batch)
        return q_values.amin(dim=0) if use_cdq else q_values.mean(dim=0)  # (batch)
