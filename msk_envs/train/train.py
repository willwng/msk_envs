"""
Train a policy on a task env:

    python -m msk_envs.train.train <task> --algo <td3|sac|qflex|ppo> [options] env-config:<model> [env options]

See hyperparams.py for the tasks and their defaults.
"""
import torch
import warp as wp

from msk_envs.envs import make_env
from msk_envs.train.common import RunPaths
from msk_envs.train.hyperparams import BaseArgs, algo_config, get_args, pretty_print_base_args
from msk_envs.train.ppo.trainer import PPOTrainer
from msk_envs.train.qflex.trainer import QFlexTrainer
from msk_envs.train.sac.trainer import SACTrainer
from msk_envs.train.td3.trainer import TD3Trainer
from msk_envs.train.utils import init_wandb_run, set_seed

TRAINERS = {
    "td3": TD3Trainer,
    "sac": SACTrainer,
    "qflex": QFlexTrainer,
    "ppo": PPOTrainer,
}


def run(args: BaseArgs, device: torch.device) -> None:
    """ Build the training and eval envs, and train with the algo selected by args.algo """
    cfg = algo_config(args)
    env_kwargs = dict(env_config=args.env_config, device=device, cuda_graph=device.type == "cuda")
    envs = make_env(num_envs=cfg.num_envs, requires_visuals=False, **env_kwargs)
    eval_envs = make_env(num_envs=cfg.num_eval_envs, requires_visuals=True, **env_kwargs)

    trainer = TRAINERS[args.algo](cfg, envs=envs, eval_envs=eval_envs, paths=RunPaths(args.exp_name), device=device)
    trainer.train()


def main():
    args = get_args()
    set_seed(args.seed)
    pretty_print_base_args(args)

    assert torch.cuda.is_available(), "CUDA device not available"
    device_str = f"cuda:{args.gpu_id}"
    wp.set_device(device_str)

    if args.use_wandb:
        init_wandb_run(args)
    run(args, torch.device(device_str))


if __name__ == "__main__":
    main()
