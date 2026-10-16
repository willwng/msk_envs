import torch


class RolloutStorage:
    """On-policy storage for a single PPO rollout, batched over environments.

    All tensors are shaped ``[num_steps, num_envs, ...]``. A rollout is filled
    step-by-step via :meth:`add`, then :meth:`compute_returns` fills the GAE
    advantages/returns, and :meth:`mini_batch_generator` yields flattened
    minibatches for the update.
    """

    def __init__(
            self,
            num_envs: int,
            num_steps: int,
            n_obs: int,
            n_act: int,
            device: torch.device,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        self.observations = torch.zeros(num_steps, num_envs, n_obs, device=device)
        self.actions = torch.zeros(num_steps, num_envs, n_act, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.mus = torch.zeros(num_steps, num_envs, n_act, device=device)
        self.sigmas = torch.zeros(num_steps, num_envs, n_act, device=device)

        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0

    def add(self, obs, actions, rewards, dones, values, log_probs, mus, sigmas):
        t = self.step
        self.observations[t].copy_(obs)
        self.actions[t].copy_(actions)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones.float())
        self.values[t].copy_(values)
        self.log_probs[t].copy_(log_probs)
        self.mus[t].copy_(mus)
        self.sigmas[t].copy_(sigmas)
        self.step += 1

    def clear(self):
        self.step = 0

    @torch.no_grad()
    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        """ Generalized Advantage Estimation (truncations treated as done; see the trainer's truncation bootstrap).
        Advantages are normalized once over the whole rollout """
        advantage = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
            next_not_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_not_terminal - self.values[t]
            advantage = delta + gamma * lam * next_not_terminal * advantage
            self.advantages[t] = advantage
        self.returns.copy_(self.advantages + self.values)
        self.advantages.copy_((self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8))

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        batch_size = self.num_steps * self.num_envs
        mini_batch_size = batch_size // num_mini_batches

        # Flatten [T, N, ...] -> [T*N, ...]
        observations = self.observations.reshape(batch_size, -1)
        actions = self.actions.reshape(batch_size, -1)
        values = self.values.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        log_probs = self.log_probs.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        mus = self.mus.reshape(batch_size, -1)
        sigmas = self.sigmas.reshape(batch_size, -1)

        for _ in range(num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                idx = perm[i * mini_batch_size:(i + 1) * mini_batch_size]
                yield {
                    "observations": observations[idx],
                    "actions": actions[idx],
                    "values": values[idx],
                    "returns": returns[idx],
                    "advantages": advantages[idx],
                    "old_log_probs": log_probs[idx],
                    "old_mu": mus[idx],
                    "old_sigma": sigmas[idx],
                }
