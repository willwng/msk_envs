from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    num_envs: int = 4096
    """number of parallel environments"""

    num_learning_iterations: int = 1000000
    """total number of policy-update iterations (each collects a full rollout)"""

    num_steps_per_env: int = 24
    """number of environment steps collected per env per rollout"""

    num_learning_epochs: int = 8
    """number of passes over the collected rollout per iteration"""

    num_mini_batches: int = 4
    """number of minibatches the rollout is split into per epoch"""

    # ------------------------------------------------------------------ optim
    actor_learning_rate: float = 1e-5
    """the learning rate of the actor"""

    critic_learning_rate: float = 1e-5
    """the learning rate of the critic"""

    max_actor_learning_rate: float | None = None
    """upper bound for the adaptive actor learning rate (None -> 1e-2)"""

    min_actor_learning_rate: float | None = None
    """lower bound for the adaptive actor learning rate (None -> 1e-5)"""

    max_critic_learning_rate: float | None = None
    """upper bound for the adaptive critic learning rate (None -> 1e-2)"""

    min_critic_learning_rate: float | None = None
    """lower bound for the adaptive critic learning rate (None -> 1e-5)"""

    schedule: str = "adaptive"
    """learning-rate schedule: 'adaptive' (KL based) or 'fixed'"""

    desired_kl: float = 0.01
    """target KL divergence used by the adaptive schedule"""

    # ------------------------------------------------------------- ppo losses
    gamma: float = 0.99
    """the discount factor gamma"""

    lam: float = 0.95
    """the GAE lambda"""

    clip_param: float = 0.2
    """the PPO surrogate (and value) clipping coefficient"""

    entropy_coef: float = 0.01
    """coefficient of the entropy bonus"""

    value_loss_coef: float = 1.0
    """coefficient of the value loss"""

    max_grad_norm: float = 1.0
    """maximum gradient norm for clipping (<= 0 disables)"""

    # ------------------------------------------------------------- networks
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    """hidden layer sizes shared by the actor and critic MLPs"""

    activation: str = "ELU"
    """activation function (name of a torch.nn module, e.g. 'ELU')"""

    use_layer_norm: bool = False
    """whether to use layer normalization in the networks"""

    dropout_prob: float = 0.0
    """dropout probability between hidden layers (0 disables)"""

    init_noise_std: float = 0.8
    """initial standard deviation of the (state-independent) action noise"""

    min_noise_std: float | None = None
    """if set, clamp the action std to this minimum value"""

    min_mean_noise_std: float | None = None
    """if set, scale std up when its mean falls below this threshold"""

    # ------------------------------------------------------------ misc/train
    empirical_normalization: bool = False
    """whether to normalize observations with running empirical statistics"""

    compile: bool = False
    """whether to torch.compile the loss computation"""
    compile_mode: str | None = None
    compile_backend: str = "inductor"

    save_interval: int = 100
    """the interval (in iterations) to save the model"""

    logging_interval: int = 1
    """the interval (in iterations) to log the metrics"""

    checkpoint_path: str = ""

    """ Evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 100
