from .hyperparams import get_args
from msk_envs.envs.env_config import EnvConfig
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.nets.networks import load_policy


import torch
from tqdm import tqdm


def main():
    args = get_args()
    env_config = EnvConfig(
        env_variant=args.env_variant,
        reward_lambdas=args.get_reward_lambdas(),
        imitation_weights=args.get_imitation_weights(),
        extra_rewarded_joints=args.extra_rewarded_joints,
        lambda_extra_rewarded_joints=args.lambda_extra_rewarded_joints,
        extra_rewarded_dofs=args.extra_rewarded_dofs,
        lambda_extra_rewarded_dofs=args.lambda_extra_rewarded_dofs,
    )
    # env_config.delta_t = 1.0 / 2000.0

    env_config.q_noise = 0.0
    env_config.qv_noise = 0.0
    env_config.swap_lr = False

    has_cuda_support = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda and has_cuda_support else f"cpu")
    envs = EnvFactory.create_env(num_envs=1,
                                 env_config=env_config,
                                 render=False,
                                 cuda_graph=has_cuda_support,
                                 device=device)

    actions = envs.get_blank_actions()

    # policy = load_policy("/home/marth/Documents/msk_envs/models/imitate_no_damp_2026-01-06_09-40-54/imitate_no_damp_2026-01-06_09-40-54_120000.pt")
    # policy.to(device)


    # Build a SimLogger to give us a whole pdf of stuff
    max_episode_length = int(env_config.max_episode_duration / env_config.delta_t)
    sim = LoggedSim(envs, max_episode_length, device)
    obs = sim.reset()

    for _ in tqdm(range(max_episode_length)):
        # actions = torch.randn_like(actions)
        # actions = envs.get_blank_actions()
        # with torch.no_grad():
        #     actions = policy(obs)
        finished, obs = sim.step(actions)
        if finished:
            break
    print("Mean rewards: ", sim.get_rewards_mean())

    # write to torch
    sim.save_animation("dashboard/trajectories/test", "999")
    sim.save_analytics(".", "deploy_analytics")
    sim.save_frame_data(".", "deploy_frame_data")


if __name__ == "__main__":
    main()
