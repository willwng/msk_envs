""" Building blocks shared by the training algorithms """
import os
from typing import Callable

import torch
import torch.nn as nn
import wandb
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from msk_envs.envs.base import MSKEnv
from msk_envs.recording import LoggedSim
from msk_envs.train.utils import LoggingHelper, cpu_state

torch.set_float32_matmul_precision("high")


class RunPaths:
    """ Where a training run writes its outputs (relative to the working directory) """

    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.log_dir = f"models/{exp_name}"
        """ TensorBoard logs and checkpoints """
        self.traj_out_folder = f"dashboard/trajectories/{exp_name}"
        """ Eval animations, for the dashboard """
        self.analytics_out_folder = f"models/frame_data/{exp_name}"
        """ Eval frame data and analytics PDFs """

    def checkpoint(self, global_step: int) -> str:
        return f"{self.log_dir}/{self.exp_name}_{global_step}.pt"


def make_logging_helper(paths: RunPaths, device: torch.device, num_envs: int, num_steps_per_log: int,
                        num_learning_iterations: int) -> LoggingHelper:
    writer = SummaryWriter(log_dir=paths.log_dir, flush_secs=10)
    return LoggingHelper(
        writer,
        log_dir=paths.log_dir,
        device=device,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_log,
        num_learning_iterations=num_learning_iterations,
    )


def module_state(module: nn.Module) -> dict:
    return cpu_state(module.state_dict())


def save_checkpoint(path: str, global_step: int, cfg, state: dict, logging_helper: LoggingHelper = None) -> None:
    """
    Save [state] along with the step and training config. The config is stored as a dict under "args", which
    inference loaders use to rebuild the networks. With [logging_helper], the checkpoint is also uploaded to wandb.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {**state, "args": vars(cfg), "global_step": global_step}
    if wandb.run is not None:
        checkpoint["wandb_run_id"] = wandb.run.id
    if logging_helper is not None:
        logging_helper.save_checkpoint_artifact(checkpoint, path)
    else:
        torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def clip_grad_norm(parameters, max_grad_norm: float, device: torch.device) -> torch.Tensor:
    """ Clip gradients to [max_grad_norm] (if > 0) and return their norm (0 if not clipped) """
    if max_grad_norm > 0:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
    return torch.tensor(0.0, device=device)


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """ target <- (1 - tau) * target + tau * source """
    src_ps = [p.data for p in source.parameters()]
    tgt_ps = [p.data for p in target.parameters()]
    torch._foreach_mul_(tgt_ps, 1.0 - tau)
    torch._foreach_add_(tgt_ps, src_ps, alpha=tau)


def raw_reward_means(info: dict) -> dict:
    """ Mean of each reward term before its lambda multiplier, for logging """
    return {f"{name}_raw": value.mean() for name, value in info["raw_rewards"].items()}


@torch.no_grad()
@torch.compiler.disable
def evaluate_policy(
        eval_envs: MSKEnv,
        policy: Callable[[torch.Tensor], torch.Tensor],
        paths: RunPaths,
        global_step: int,
        device: torch.device,
) -> tuple[float, float]:
    """
    Run one recorded episode with [policy] (observations -> actions), save its animation, frame data and analytics,
    and return the mean return and episode length
    """
    sim = LoggedSim(eval_envs, device=device)
    eval_obs = sim.reset()
    for _ in range(sim.max_env_steps):
        eval_actions = policy(eval_obs)
        finished, eval_obs = sim.step(eval_actions)
        if torch.as_tensor(finished).all():  # every eval world's episode has ended
            break
    rewards_mean = sim.get_rewards_mean()
    episode_length_mean = sim.get_episode_length_mean()

    sim.save_animation(paths.traj_out_folder, str(global_step), use_gzip=True)
    sim.save_frame_data(paths.analytics_out_folder, f"frame_data_{global_step}", use_gzip=True)
    sim.save_analytics(paths.analytics_out_folder, f"analytics_{global_step}")

    eval_avg_return, eval_avg_length = rewards_mean.item(), episode_length_mean.item()
    logger.info(f"Eval Average Return: {eval_avg_return}, Eval Average Length: {eval_avg_length}")
    return eval_avg_return, eval_avg_length


def print_config(cfg, title: str) -> None:
    print(f"{title}:")
    for field in cfg.__dataclass_fields__:
        print(f"  {field}: {getattr(cfg, field)}")
