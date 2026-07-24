from dataclasses import dataclass


@dataclass
class QFlexRefConfig:
    """Config for the reference (JAX) Qflex / FlowExp algorithm. """

    num_envs: int = 256
    """parallel environments (reference uses 8)"""
    num_learning_iterations: int = 5_000_000
    """total transitions to collect"""
    start_env_steps: int = 30_000
    """warmup transitions of random actions before learning"""
    buffer_size: int = 1_000_000
    """replay buffer capacity in transitions"""
    batch_size: int = 256
    """transitions per gradient update"""
    num_updates: int = 1
    """gradient updates per env step"""

    learning_rate: float = 3e-4
    """lr for Q / velocity / reference policy"""
    alpha_lr: float = 3e-4
    """lr for the entropy temperature"""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 1.0
    """target soft-update coefficient (1.0 since we are using CrossQ)"""
    reward_scale: float = 1.0
    """reward scaling"""

    # --- networks ---
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    """hidden widths for Q / velocity / reference-policy MLPs"""
    num_flow_steps: int = 20
    """Euler steps for the flow policy"""
    use_bn: bool = True
    """BatchNorm inside networks; only True is supported upstream"""
    learn_reference_gn: bool = True
    """jointly learn the reference Gaussian policy"""

    # --- Q-guided flow target construction ---
    grad_step_size: float = 1e-2
    """Q-gradient ascent step size"""
    grad_step_num: int = 20
    """number of Q-gradient ascent steps"""

    # --- checkpoint / logging / eval ---
    save_interval: int = 250_000
    """interval in transitions to save the policy"""
    logging_interval: int = 100
    num_eval_envs: int = 1
    eval_freq: int = 2000

    @staticmethod
    def pretty_print(cfg: "QFlexRefConfig"):
        print("QFlexRef (JAX) Configuration:")
        for f in cfg.__dataclass_fields__:
            print(f"  {f}: {getattr(cfg, f)}")
