from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PPOConfig:
    """ Defaults follow musclemimic's fullbody setup (fullbody/conf_fullbody.yaml) """

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
    """the learning rate (one optimizer for the actor and critic)"""

    schedule: Literal["linear", "fixed", "adaptive"] = "linear"
    """learning-rate schedule: linear decay to 0 over training, constant, or KL-adaptive"""

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

    weight_decay: float = 0.0
    """weight decay for all optimizers"""

    max_grad_norm: float = 1.0
    """maximum global gradient norm over the actor and critic (<= 0 disables clipping)"""

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

    reward_normalization: bool = True
    """whether to scale rewards by the running std of the discounted return"""

    # ------------------------------------------------------------ misc/train
    compile: bool = False
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
