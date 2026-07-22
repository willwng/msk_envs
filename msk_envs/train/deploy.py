import bolt
import torch
from tqdm import tqdm

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.train.hyperparams import get_args, pretty_print_base_args
from msk_envs.train.dep.dep import DEP
from msk_envs.utils.logged_sim import LoggedSim
# from msk_envs.train.nets.sac_networks import load_policy
# from msk_envs.train.nets.td3_networks import load_policy
from msk_envs.train.nets.ppo_networks import load_policy


def main():
    args = get_args()
    pretty_print_base_args(args)

    # set seed
    torch.manual_seed(args.seed)

    # no noise
    env_config = args.env_config
    env_config.q_noise = 0.0
    env_config.qv_noise = 0.0
    env_config.swap_lr = False
    # env_config.integrator_accuracy = 0.1
    # env_config.armature = 0.0
    # env_config.integrator_use_inf_norm = False

    has_cuda_support = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda_support else f"cpu")
    num_envs = 1
    envs = EnvFactory.create_env(num_envs=num_envs,
                                 env_config=env_config,
                                 requires_visuals=True,
                                 cuda_graph=has_cuda_support,
                                 device=device)

    actions = envs.get_blank_actions()

    # policy = load_policy("/home/marth/Documents/msk_envs/models/sprinter_ppo3_2026-07-01_12-57/sprinter_ppo3_2026-07-01_12-57_10700.pt")
    # policy.to(device)

    dep = DEP(n_motors=envs.num_muscles,
              n_envs=num_envs,
              buffer_size=60,
              bias_rate=0.0066,
              kappa=1000,
              tau=12,
              s4avg=1,
              regularization=32,
              time_dist=2,
              with_learning=True,
              device=device)

    # Build a SimLogger to give us a whole pdf of stuf
    max_episode_length = int(envs.max_episode_duration / envs.delta_t)
    recording_fps = 30.0
    sim = LoggedSim(envs, device, delta_t_log=1.0 / recording_fps)
    obs = sim.reset()

    for i in tqdm(range(max_episode_length)):
        actions = torch.randn_like(actions)
        # actions = torch.ones_like(actions)
        # actions = torch.ones_like(actions) * 0

        # actions[:, :envs.num_muscles] = -1.0
        # muscle_states = envs.muscle_fiber_lengths
        # actions[:, envs.num_muscles:] = torch.randn_like(actions[:, envs.num_muscles:])
        # actions[:, :envs.num_muscles] = dep.step(muscle_states)

        # with torch.no_grad():
        #     actions = policy(obs)

        # try to step the sim, but if it takes too long, break out of the loop
        finished, obs = sim.step(actions)
        if finished.all():
            break

    print("Mean rewards: ", sim.get_rewards_mean())

    # write to torch
    sim.save_animation("dashboard/trajectories/test", "999", use_gzip=True)
    sim.save_analytics(".", "deploy_analytics")
    sim.save_frame_data(".", "deploy_frame_data", use_gzip=True)


if __name__ == "__main__":
    main()
