"""
Inference policies (observations -> actions) from training checkpoints, for TD3, SAC and PPO.

    policy = load_policy("models/<exp>/<exp>_<step>.pt")  # deterministic=True for the mean action
    actions = policy(obs)  # sampled, for the stochastic policies (SAC, PPO)
"""
import torch
import torch.nn as nn

from msk_envs.train.nets.deterministic_policy import DeterministicPolicy
from msk_envs.train.nets.gaussian_policy import GaussianPolicy
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.nets.ppo_networks import PPOActor


class InferencePolicy(nn.Module):
    """
    An actor with the observation normalizer it was trained with, in eval mode. Stochastic policies (SAC, PPO) sample
    actions unless [deterministic], in which case they use the mean; TD3's policy is deterministic.
    """

    def __init__(self, actor: nn.Module, obs_normalizer: nn.Module, algo: str, deterministic: bool = False):
        super().__init__()
        self.actor = actor
        self.obs_normalizer = obs_normalizer
        self.algo = algo
        self.deterministic = deterministic
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(obs)
        if self.algo == "sac":
            return self.actor.explore(x, deterministic=self.deterministic)
        if self.algo == "ppo" and not self.deterministic:
            return self.actor.act_inference(x)
        return self.actor(x)


def _detect_algo(actor_state: dict) -> str:
    if "log_std" in actor_state:
        return "ppo"
    if "fc_logstd.weight" in actor_state:
        return "sac"
    if "fc_mu.0.weight" in actor_state:
        return "td3"
    raise ValueError("Unsupported checkpoint: only TD3, SAC and PPO policies can be loaded for inference")


def _build_actor(algo: str, actor_state: dict, args: dict) -> nn.Module:
    if algo == "td3":
        return DeterministicPolicy(
            n_obs=actor_state["net.0.weight"].shape[-1],
            n_act=actor_state["fc_mu.0.weight"].shape[0],
            num_envs=args["num_envs"],
            device="cpu",
            hidden_dim=args["actor_hidden_dim"],
            std_min=args["std_min"],
            std_max=args["std_max"],
            use_layer_norm=args["use_layer_norm"],
        )
    if algo == "sac":
        return GaussianPolicy(
            n_obs=actor_state["net.0.weight"].shape[-1],
            n_act=actor_state["fc_mu.0.weight"].shape[0],
            num_envs=args["num_envs"],
            hidden_dim=args["actor_hidden_dim"],
            log_std_max=args["log_std_max"],
            log_std_min=args["log_std_min"],
            use_tanh=args["use_tanh"],
            use_layer_norm=args["use_layer_norm"],
            device="cpu",
            action_scale=args.get("action_scale"),  # restored from the state dict either way
            action_bias=args.get("action_bias"),
        )
    return PPOActor(
        n_obs=actor_state["actor.0.weight"].shape[-1],
        n_act=actor_state["log_std"].shape[0],
        hidden_dims=list(args["actor_hidden_dims"]),
        init_std=args["init_std"],
        learnable_std=args["learnable_std"],
        activation=args["activation"],
        use_layer_norm=args["use_layer_norm"],
        device="cpu",
    )


def policy_from_checkpoint(checkpoint: dict, deterministic: bool = False) -> InferencePolicy:
    """
    Inference policy (on the CPU) from a checkpoint's "args" (training config), "actor_state_dict" and
    "obs_normalizer_state", e.g. a loaded checkpoint file or a trainer's current state
    """
    actor_state = checkpoint["actor_state_dict"]
    algo = _detect_algo(actor_state)
    actor = _build_actor(algo, actor_state, checkpoint["args"])
    actor.load_state_dict(actor_state)

    normalizer_state = checkpoint["obs_normalizer_state"]
    if not normalizer_state:  # trained without observation normalization
        obs_normalizer = nn.Identity()
    else:
        obs_normalizer = EmpiricalNormalization(shape=normalizer_state["_mean"].shape[-1], device="cpu")
        obs_normalizer.load_state_dict(normalizer_state)
    return InferencePolicy(actor, obs_normalizer, algo, deterministic=deterministic)


def load_policy(checkpoint_path: str, deterministic: bool = False) -> InferencePolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return policy_from_checkpoint(checkpoint, deterministic=deterministic)
