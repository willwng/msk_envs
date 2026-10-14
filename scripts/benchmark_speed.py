"""
Measure simulation speed for a task with random actions, and the spread of adaptive integrator steps per env step.

    python scripts/benchmark_speed.py [--num-envs 4096] [--steps 20] [--histogram steps.png] \
        <task> env-config:<model> [env options]

Task arguments are the same as for training (see msk_envs/train/hyperparams.py).
"""
import argparse
import sys
from time import perf_counter

import torch
import warp as wp
from tqdm import tqdm

from msk_envs.envs import make_env
from msk_envs.train.hyperparams import get_args, pretty_print_base_args


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--histogram", default="", help="Save a histogram of integrator steps to this image")
    cli, task_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + task_args  # the task CLI (and its env overrides) reads sys.argv

    args = get_args()
    pretty_print_base_args(args)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    envs = make_env(num_envs=cli.num_envs, env_config=args.env_config, requires_visuals=False, cuda_graph=has_cuda,
                    device=device)
    envs.reset()
    actions = envs.get_blank_actions()
    steps_taken = wp.to_torch(envs.d.steps_attempted)

    all_steps_taken = []
    time_start = perf_counter()
    for _ in tqdm(range(cli.steps)):
        envs.step(torch.randn_like(actions))
        print("Steps taken:", torch.min(steps_taken).item(), "to", torch.max(steps_taken).item())
        all_steps_taken.append(steps_taken.clone())
    elapsed = perf_counter() - time_start

    simulated_time = cli.steps * envs.delta_t
    print("Real time factor (1 world): ", simulated_time / elapsed)
    print("Real time factor (all worlds): ", cli.num_envs * simulated_time / elapsed)

    if cli.histogram:
        import matplotlib.pyplot as plt
        plt.hist(torch.stack(all_steps_taken).cpu().numpy().flatten(), bins=20)
        plt.xlabel("Integrator steps per env step")
        plt.savefig(cli.histogram)
        print("Saved histogram to", cli.histogram)


if __name__ == "__main__":
    main()
