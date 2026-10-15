"""
Roll out one episode of a task (with random actions, or a trained policy) and save its animation, frame data and
analytics PDF.

    python scripts/rollout.py [--checkpoint models/.../x.pt] [--out-dir rollouts] [--fps 30] \
        <task> env-config:<model> [env options]

Task arguments are the same as for training (see msk_envs/train/hyperparams.py)
"""
import argparse
import sys

import torch
from tqdm import tqdm

from msk_envs.envs import make_env
from msk_envs.recording import LoggedSim
from msk_envs.train.hyperparams import get_args, pretty_print_base_args
from msk_envs.train.inference import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="", help="Policy checkpoint (random actions if not given)")
    parser.add_argument("--out-dir", default="rollouts")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording rate")
    cli, task_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + task_args  # the task CLI (and its env overrides) reads sys.argv

    args = get_args()
    pretty_print_base_args(args)
    torch.manual_seed(args.seed)

    env_config = args.env_config
    env_config.q_noise = 0.0
    env_config.qv_noise = 0.0
    env_config.apply_swap_lr = False

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    envs = make_env(num_envs=1, env_config=env_config, requires_visuals=True, cuda_graph=has_cuda, device=device)

    if cli.checkpoint:
        policy = load_policy(cli.checkpoint).to(device)
    else:
        policy = lambda obs: torch.randn_like(envs.get_blank_actions())  # noqa: E731

    sim = LoggedSim(envs, device, delta_t_log=1.0 / cli.fps)
    obs = sim.reset()
    for _ in tqdm(range(int(envs.max_episode_duration / envs.delta_t))):
        with torch.no_grad():
            actions = policy(obs)
        finished, obs = sim.step(actions)
        if finished.all():
            break
    print("Mean rewards: ", sim.get_rewards_mean())

    sim.save_animation(cli.out_dir, "animation", use_gzip=True)
    sim.save_frame_data(cli.out_dir, "frame_data", use_gzip=True)
    sim.save_analytics(cli.out_dir, "analytics")


if __name__ == "__main__":
    main()
