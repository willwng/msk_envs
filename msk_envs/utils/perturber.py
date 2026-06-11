import torch


class Perturber:
    def __init__(
            self,
            num_envs: int,
            device: torch.device,
            perturbation_duration: tuple,
            perturbation_frequency: tuple,
            force_std: float,
            delta_t: float,
            enabled: bool,
    ):
        self.num_envs = num_envs
        self.device = device
        self.enabled = enabled

        # How long perturbations last
        self.perturbation_duration = perturbation_duration
        # How often to wait between perturbations
        self.perturbation_frequency = perturbation_frequency

        # Whether to apply perturbations this step
        self.perturbation_enabled = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

        # Timers to track perturbation durations and wait times
        self.timer_target_duration = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.timer = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        # Standard deviation of force to apply
        self.force_std = force_std

        # Duration between calls
        self.delta_t = delta_t
        return

    def sample_range(self, range_tuple: tuple) -> torch.Tensor:
        return torch.rand(self.num_envs, device=self.device) * (range_tuple[1] - range_tuple[0]) + range_tuple[0]

    def apply(self, root_id: int, body_user_forces: torch.Tensor) -> None:
        if not self.enabled:
            return

        # If any world are currently applying perturbations *and* have exceeded their perturbation duration
        worlds_done = (self.perturbation_enabled & (self.timer >= self.timer_target_duration))
        if torch.any(worlds_done):
            # Disable perturbations, reset timer
            self.perturbation_enabled[worlds_done] = False
            self.timer[worlds_done] = 0.0
            # Sample time to wait until next perturbation
            wait_times = self.sample_range(self.perturbation_frequency)
            self.timer_target_duration[worlds_done] = wait_times[worlds_done]

        # If any worlds are *not* currently applying perturbations *and* have exceeded their wait time
        worlds_start = (~self.perturbation_enabled & (self.timer >= self.timer_target_duration))
        if torch.any(worlds_start):
            # Enable perturbations, reset timer
            self.perturbation_enabled[worlds_start] = True
            self.timer[worlds_start] = 0.0
            # Sample perturbation durations
            perturb_durations = self.sample_range(self.perturbation_duration)
            self.timer_target_duration[worlds_start] = perturb_durations[worlds_start]

            # Sample a new random external force for these worlds
            num_perturb = torch.sum(worlds_start).item()
            force_magnitudes = torch.randn(num_perturb, device=self.device) * self.force_std
            force_directions = torch.randn((num_perturb, 3), device=self.device)
            force_directions = force_directions / torch.norm(force_directions, dim=1, keepdim=True)
            external_forces = force_directions * force_magnitudes.unsqueeze(1)
            body_user_forces[worlds_start, root_id, 3:6] = external_forces

        # Make sure we reset forces for worlds not applying perturbations
        no_perturb_mask = ~self.perturbation_enabled
        body_user_forces[no_perturb_mask, :, :] = 0.0

        # Increment timers
        self.timer += self.delta_t
        return
