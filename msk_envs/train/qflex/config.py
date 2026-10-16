from dataclasses import dataclass
from typing import Literal


@dataclass
class QFlexConfig:
    num_envs: int = 4096
    """number of parallel environments"""

    num_learning_iterations: int = 150000
    """total timesteps of the experiments"""

    learning_rate: float = 3e-4
    """learning rate"""
    betas: tuple[float, float] = (0.5, 0.999)
    """Adam betas"""
    optimizer: Literal["adamw", "muon"] = "adamw"
    """adamw, or muon: Muon for weight matrices and AdamW for the other parameters"""

    weight_decay: float = 0.0
    """weight decay"""

    buffer_size: int = 256 * 20
    """the replay memory buffer size per environment"""
    num_steps: int = 1
    """the number of steps to use for the multi-step return"""
    gamma: float = 0.99
    """the discount factor"""
    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""
    num_updates: int = 8
    """the number of updates to perform per step"""
    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    actor_hidden_dim: int = 256
    """the hidden dimension of the actor network"""
    velocity_hidden_dim: int = 256
    """hidden dimension of velocity network"""
    hidden_num: int = 3
    """number of hidden layers in each MLP (uniform width)"""
    log_std_max: float = 2.0
    """the maximum value of the log std"""
    log_std_min: float = -20.0
    """the minimum value of the log std"""
    target_entropy_ratio: float = 0.1
    """the ratio of the target entropy to the number of actions"""
    alpha_init: float = 0.001
    """the initial value of the alpha"""

    critic_hidden_dim: int = 256
    """the hidden dimension of the critic network"""
    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    obs_normalization: bool = False
    """ whether to normalize observations """

    num_flow_steps: int = 20
    """Euler steps for sampling from the flow (``diffusion_steps``)."""
    grad_step_size: float = 1e-2
    """step size of the Q-gradient ascent that builds the flow target."""
    grad_step_num: int = 20
    """number of Q-gradient ascent steps."""

    compile: bool = True
    """whether to use torch.compile."""
    compile_mode: str = "reduce-overhead"  # "max-autotune" can fail on some GPU architectures
    compile_backend: str = "inductor"  # "eager" is slower but safer

    save_interval: int = 1000
    """ the interval to save the model """
    logging_interval: int = 100
    """ the interval to log the metrics """

    """ Evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 1000
