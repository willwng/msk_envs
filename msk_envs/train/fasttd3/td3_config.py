from dataclasses import dataclass


@dataclass
class TD3Config:
    agent: str = "fasttd3"
    """the type of the agent (fasttd3, simbav2)"""

    num_envs: int = 4096
    """number of parallel environments"""

    num_learning_iterations: int = 150000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    critic_learning_rate_end: float = 3e-5
    """the end learning rate of the critic"""

    actor_learning_rate_end: float = 3e-5
    """the end learning rate for the actor"""

    buffer_size: int = 256 * 25
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.99
    """the discount factor gamma"""

    tau: float = 0.1
    """the soft update coefficient """

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    policy_noise: float = 0.001
    """the scale of target action noise"""

    noise_clip: float = 0.5
    """the clip range of target action noise"""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -5.0
    """the minimum value of the support"""

    v_max: float = 15.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    critic_num_blocks: int = 2
    """ SimbaV2 only, the number of blocks in the critic network """

    actor_num_blocks: int = 1
    """ SimbaV2 only, the number of blocks in the actor network """

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_cdq: bool = True
    """whether to use Clipped Double Q-learning"""

    std_min: float = 0.001
    """minimum scale of exploration noise"""

    std_max: float = 0.4
    """maximum scale of exploration noise"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    use_synergistic_noise: bool = False
    """whether to use synergistic noise"""

    use_gsde: bool = False
    """ whether to use gSDE for exploration """
    gsde_steps: int = 10
    """ number of steps to sample new noise for gSDE """

    compile: bool = True
    """whether to use torch.compile."""
    compile_mode: str = "reduce-overhead"  # "max-autotune" can fail on some GPU architectures
    compile_backend: str = "inductor"  # "eager" is slower but safer

    obs_normalization: bool = True
    """ whether to normalize observations """

    reward_normalization: bool = False
    """ whether to normalize rewards """

    use_layer_norm: bool = False
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """ maximum gradient norm for clipping """

    amp: bool = True
    """ wether to use amp """

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    use_soap: bool = False
    """ whether to use soap optimizer """

    weight_decay: float = 0.001
    """ weight decay for optimizers """

    save_interval: int = 1000
    """ the interval to save the model """

    logging_interval: int = 100
    """ the interval to log the metrics """

    """ device/torch settings """
    checkpoint_path: str = ""

    # Checkpoint details
    save_optimizer_state: bool = False

    """ Evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 1000

    @staticmethod
    def pretty_print(td3_config):
        print("TD3 Configuration:")
        for field in td3_config.__dataclass_fields__:
            value = getattr(td3_config, field)
            print(f"  {field}: {value}")
