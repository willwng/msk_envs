from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PPOConfig:
    """ Hyperparameter defaults follow musclemimic's fullbody setup """

    num_envs: int = 8192
    """number of parallel environments"""

    num_learning_iterations: int = 12500
    """number of policy updates (each collects a rollout of num_envs x num_steps_per_env steps)"""

    num_steps_per_env: int = 20
    """number of environment steps collected per env per rollout"""

    num_learning_epochs: int = 1
    """number of passes over the collected rollout per update"""

    num_mini_batches: int = 128
    """number of minibatches the rollout is split into per epoch"""

    # ------------------------------------------------------------------ optim
    learning_rate: float = 4e-4
    """the learning rate of the actor and critic optimizers"""

    schedule: Literal["linear", "fixed", "adaptive"] = "linear"
    """learning-rate schedule: linear decay to min_lr_ratio * learning_rate over training, constant, or KL-adaptive
    (rsl_rl-style, adjusted after every minibatch)"""

    min_lr_ratio: float = 0.0
    """'linear' schedule: final learning rate as a fraction of learning_rate"""

    desired_kl: float = 0.01
    """target KL divergence for the adaptive schedule"""

    min_learning_rate: float = 1e-5
    """lower bound for the adaptive learning rate"""

    max_learning_rate: float = 1e-2
    """upper bound for the adaptive learning rate"""

    optimizer: Literal["muon", "adamw"] = "muon"
    """muon: Muon for weight matrices and AdamW for the other parameters; adamw: AdamW for all"""

    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    """number of Newton-Schulz iterations"""

    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] = "original"
    """Muon per-matrix lr scale: 'original' sqrt(max(1, rows / cols)) as in optax and torch.optim.Muon,
    or 'match_rms_adamw' 0.2 * sqrt(max(rows, cols)) (6.4x larger steps for 1024 x 1024 matrices)"""

    adam_betas: tuple[float, float] = (0.9, 0.999)
    """AdamW betas (also used for the AdamW part of muon)"""

    adam_eps: float = 1e-8
    """AdamW epsilon"""

    weight_decay: float = 0.0
    """weight decay for all optimizers"""

    max_grad_norm: float = 1.0
    """maximum gradient norm, clipped separately for the actor and the critic (<= 0 disables clipping)"""

    # ------------------------------------------------------------- ppo losses
    gamma: float = 0.99
    """the discount factor gamma"""

    lam: float = 0.95
    """the GAE lambda"""

    clip_param: float = 0.2
    """the PPO surrogate clipping coefficient"""

    value_clip_param: float = 0.2
    """the value function clipping coefficient"""

    entropy_coef: float = 0.0
    """coefficient of the entropy bonus"""

    value_loss_coef: float = 0.5
    """coefficient of the value loss (which is 0.5 * squared error)"""

    # ------------------------------------------------------------- networks
    actor_hidden_dims: list[int] = field(default_factory=lambda: [1024] * 16)
    """hidden layer sizes of the actor MLP"""

    critic_hidden_dims: list[int] = field(default_factory=lambda: [1024] * 16)
    """hidden layer sizes of the critic MLP"""

    activation: str = "SiLU"
    """activation function (name of a torch.nn module)"""

    use_layer_norm: bool = True
    """whether to apply LayerNorm after each hidden linear layer"""

    init_std: float = 3.0
    """initial standard deviation of the action noise"""

    learnable_std: bool = True
    """whether the action noise standard deviation is learned"""

    # ------------------------------------------------------------ normalization
    obs_normalization: bool = True
    """whether to normalize observations by running mean/variance"""

    obs_norm_eps: float = 1e-3
    """epsilon added to the running obs std before dividing"""

    obs_norm_clip: float | None = 10.0
    """clip normalized observations to [-clip, clip] (None disables)"""

    reward_normalization: bool = True
    """whether to scale rewards by the running std of the discounted return"""

    reward_norm_g_max: float = 10.0
    """the reward scale is at least max|G| / g_max (bounds normalized returns; inf disables the floor)"""

    bootstrap_truncations: bool = True
    """on time-limit truncation, add gamma * V(final_obs) to the reward instead of treating it as terminal"""

    # ------------------------------------------------------------ misc/train
    compile: bool = True
    """whether to torch.compile the loss computation"""
    compile_mode: str | None = None
    compile_backend: str = "inductor"

    save_interval: int = 100
    """the interval (in iterations) to save the model"""

    logging_interval: int = 5
    """the interval (in iterations) to log the metrics"""

    checkpoint_path: str = ""

    """ Evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 100
