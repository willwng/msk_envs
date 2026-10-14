import os
import pathlib
import random
import re
import statistics
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple

import numpy as np
import torch
import wandb
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from torch.utils.tensorboard import SummaryWriter

console = Console()


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def init_wandb_run(args):
    if not getattr(args, "use_wandb", False):
        return None

    wandb_resume = None
    wandb_run_id = None

    if getattr(args, "resume", False):
        checkpoint_path = ""
        algo = getattr(args, "algo", "").lower()
        if algo == "sac" and getattr(args.sac_config, "checkpoint_path", ""):
            checkpoint_path = args.sac_config.checkpoint_path
        elif algo == "td3" and getattr(args.td3_config, "checkpoint_path", ""):
            checkpoint_path = args.td3_config.checkpoint_path

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "wandb_run_id" in checkpoint:
                wandb_run_id = checkpoint["wandb_run_id"]
                wandb_resume = "must"

        if wandb_resume is None:
            wandb_resume = "allow"

    wandb_kwargs = {
        "project": args.project,
        "name": args.exp_name,
        "save_code": True,
    }
    if wandb_resume:
        wandb_kwargs["resume"] = wandb_resume
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id

    if getattr(args, "resume", False) and getattr(args, "override_wandb_config", False):
        run = wandb.init(**wandb_kwargs)
        run.config.update(vars(args), allow_val_change=True)
        return run

    wandb_kwargs["config"] = vars(args)
    return wandb.init(**wandb_kwargs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def find_latest_checkpoint(exp_prefix: str, models_dir: str = "models") -> Tuple[
    Optional[str], Optional[str], Optional[int]]:
    """
    Find the latest checkpoint for an experiment prefix.
    
    Searches models/ directory for subdirectories matching {exp_prefix}_* pattern,
    then finds the checkpoint with the highest global_step.
    
    Args:
        exp_prefix: Experiment prefix to search for
        models_dir: Directory containing model checkpoints (default: "models")
    
    Returns:
        Tuple of (checkpoint_path, exp_name, global_step) if found, else (None, None, None)
    """
    if not exp_prefix:
        return None, None, None

    models_path = pathlib.Path(models_dir)
    if not models_path.exists():
        return None, None, None

    latest_checkpoint = None
    latest_exp_name = None
    latest_global_step = -1

    # Pattern to match directories starting with exp_prefix_
    prefix_pattern = re.compile(f"^{re.escape(exp_prefix)}_.*")

    # Search for matching directories
    for exp_dir in models_path.iterdir():
        if not exp_dir.is_dir():
            continue

        dir_name = exp_dir.name
        if not prefix_pattern.match(dir_name):
            continue

        # Search for checkpoint files in this directory
        # Pattern: {exp_name}_{global_step}.pt
        checkpoint_pattern = re.compile(rf"^{re.escape(dir_name)}_(\d+)\.pt$")

        for checkpoint_file in exp_dir.iterdir():
            if not checkpoint_file.is_file() or not checkpoint_file.suffix == ".pt":
                continue

            match = checkpoint_pattern.match(checkpoint_file.name)
            if match:
                global_step = int(match.group(1))
                if global_step > latest_global_step:
                    latest_global_step = global_step
                    latest_checkpoint = str(checkpoint_file)
                    latest_exp_name = dir_name

    if latest_checkpoint is None:
        return None, None, None

    return latest_checkpoint, latest_exp_name, latest_global_step


@torch.no_grad()
def mark_step():
    # call this once per iteration *before* any compiled function
    torch.compiler.cudagraph_mark_step_begin()


class TensorAverageMeter:
    def __init__(self):
        self.values = []

    def add(self, x):
        if isinstance(x, (int, float)):
            self.values.append(float(x))
        else:
            if len(x.shape) == 0:
                x = x.unsqueeze(0)
            self.values.append(x.item() if x.numel() == 1 else x.mean().item())

    def mean(self):
        if len(self.values) == 0:
            return 0
        return sum(self.values) / len(self.values)

    def clear(self):
        self.values = []

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean


class TensorAverageMeterDict:
    def __init__(self):
        self.data = {}

    def add(self, data_dict):
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].add(v)

    def mean(self):
        return {k: v.mean() for k, v in self.data.items()}

    def clear(self):
        self.data = {}

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean

    def get_metrics_and_clear(self):
        accumulated_metrics = self.mean_and_clear()
        # Convert tensor values to float for logging
        metrics = {}
        for key, value in accumulated_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            else:
                metrics[key] = float(value)
        return metrics


class LoggingHelper:
    """ Tracks episode returns/lengths and timing, and logs metrics to TensorBoard, wandb and the console """

    def __init__(
            self,
            writer: SummaryWriter,
            log_dir: str | pathlib.Path,
            num_envs: int,
            num_steps_per_env: int,
            num_learning_iterations: int,
            device: torch.device,
            title: str = "Training Log",
    ):
        """
        num_steps_per_env: number of steps per environment between each call to post_epoch_logging
        num_learning_iterations: total number of learning iterations
        """
        self.writer: SummaryWriter = writer
        self.log_dir: str = str(log_dir)
        self.device: torch.device = device
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.collection_time: float = 0.0
        self.learn_time: float = 0.0
        self.num_envs: int = num_envs
        self.num_steps_per_env: int = num_steps_per_env
        self.num_learning_iterations: int = num_learning_iterations
        self.title: str = title

        # Episode returns and lengths, over the last 100 finished episodes
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.cur_reward_sum: torch.Tensor = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length: torch.Tensor = torch.zeros(num_envs, dtype=torch.float, device=self.device)

    @contextmanager
    def record_collection_time(self) -> Generator[None, None, None]:
        """Record the time taken for collection."""
        start_time = time.perf_counter()
        yield
        self.collection_time += time.perf_counter() - start_time

    @contextmanager
    def record_learn_time(self) -> Generator[None, None, None]:
        """Record the time taken for learning."""
        start_time = time.perf_counter()
        yield
        self.learn_time += time.perf_counter() - start_time

    def update_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """ Accumulate this step's rewards, and record the return and length of episodes that just finished """
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        if len(new_ids) > 0:
            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0

    def post_epoch_logging(
            self,
            it: int,
            loss_dict: dict[str, float],
            extra_log_dicts: dict[str, dict[str, float]],
            width: int = 80,
            pad: int = 35,
    ) -> None:
        """
        Log [loss_dict] and [extra_log_dicts] ({section_name: {metric_name: value}}) along with episode and timing
        stats, then reset the timers
        """
        self.tot_timesteps += self.num_steps_per_env * self.num_envs
        self.tot_time += self.collection_time + self.learn_time
        iteration_time = self.collection_time + self.learn_time
        fps = int(self.num_steps_per_env * self.num_envs / (self.collection_time + self.learn_time + 1e-8))

        self._logging_to_writer(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts, fps=fps)
        log_string = self._create_console_output(
            it=it,
            loss_dict=loss_dict,
            extra_log_dicts=extra_log_dicts,
            width=width,
            pad=pad,
            iteration_time=iteration_time,
            fps=fps,
        )
        with Live(Panel(log_string, title=self.title), refresh_per_second=4, console=console):
            pass

        self.learn_time = 0.0
        self.collection_time = 0.0

    def _logging_to_writer(
            self,
            it: int,
            loss_dict: dict[str, float],
            extra_log_dicts: dict[str, dict[str, float]],
            fps: int,
    ) -> None:
        scalars_to_log: dict[str, float] = {f"Loss/{k}": v for k, v in loss_dict.items()}

        # Log extra metrics
        for section_name, section_dict in extra_log_dicts.items():
            for key, value in section_dict.items():
                scalars_to_log[f"{section_name}/{key}"] = value

        # Log performance metrics
        scalars_to_log["Perf/total_fps"] = fps
        scalars_to_log["Perf/collection_time"] = self.collection_time
        scalars_to_log["Perf/learning_time"] = self.learn_time

        # Log reward metrics if available
        if len(self.rewbuffer) > 0:
            scalars_to_log["Train/mean_reward"] = statistics.mean(self.rewbuffer)
            scalars_to_log["Train/mean_reward/time"] = statistics.mean(self.rewbuffer)
        if len(self.lenbuffer) > 0:
            scalars_to_log["Train/mean_episode_length"] = statistics.mean(self.lenbuffer)
            scalars_to_log["Train/mean_episode_length/time"] = statistics.mean(self.lenbuffer)

        scalars_to_log["Train/num_samples"] = self.tot_timesteps

        for k, v in scalars_to_log.items():
            self.writer.add_scalar(k, v, global_step=it)
        if wandb.run is not None:
            wandb.log(dict(scalars_to_log, global_step=it), step=it)

    def _create_console_output(
            self,
            it: int,
            loss_dict: dict[str, float],
            extra_log_dicts: dict[str, dict[str, float]],
            width: int,
            pad: int,
            iteration_time: float,
            fps: int,
    ) -> str:
        header = f" \033[1m Learning iteration {it}/{self.num_learning_iterations} \033[0m "

        # Base log string with computation info
        log_string = (
            f"""{header.center(width, " ")}\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s """
            f"""(Collection: {self.collection_time:.3f}s, Learning {self.learn_time:.3f}s)\n"""
        )

        # Add training metrics if available
        if len(self.rewbuffer) > 0:
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
        if len(self.lenbuffer) > 0:
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

        # Add loss metrics
        for key, value in loss_dict.items():
            log_string += f"{f'{key}:':>{pad}} {value:.4f}\n"

        # Add extra metrics
        for section_name, section_dict in extra_log_dicts.items():
            for key, value in section_dict.items():
                log_string += f"{f'{section_name}/{key}:':>{pad}} {value:.4f}\n"

        eta = self.tot_time / (it + 1) * (self.num_learning_iterations - it)

        # Add timing info
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {eta:.1f}s\n"""
        )
        log_string += f"Logging Directory: {self.log_dir}"

        return log_string

    def save_checkpoint_artifact(self, state_dict: dict[str, Any], path: str) -> None:
        if not path.startswith(self.log_dir):
            raise ValueError(f"Path {path} is not in the logging directory {self.log_dir}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving checkpoint to {path}")
        torch.save(state_dict, path)
        self.save_to_wandb(path)

    def save_to_wandb(self, file_path: str) -> None:
        """Saves file to wandb if run is initialized."""
        if wandb.run is None:
            return
        wandb.save(file_path, base_path=self.log_dir)
