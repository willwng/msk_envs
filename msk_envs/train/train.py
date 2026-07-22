import os

# Set Warp cache directory before importing warp to ensure it uses the correct location
if 'WARP_CACHE_DIR' not in os.environ:
    import tempfile

    os.environ['WARP_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), f'warp_cache_{os.getuid()}')
    os.makedirs(os.environ['WARP_CACHE_DIR'], exist_ok=True)

import torch
import warp as wp

import msk_envs.train.td3.train as fasttd3
import msk_envs.train.sac.train as fastsac
import msk_envs.train.qflex.train as qflex
import msk_envs.train.ppo.train as ppo
from msk_envs.utils.train_utils import set_seed, init_wandb_run
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.train.hyperparams import get_args, pretty_print_base_args
from msk_envs.train.dep import DEP, DEPConfig, DEPExplorer


def _build_envs(num_envs, num_eval_envs, env_config, build_cuda_graph, device):
    envs = EnvFactory.create_env(
        num_envs=num_envs,
        env_config=env_config,
        requires_visuals=False,
        cuda_graph=build_cuda_graph,
        device=device,
    )
    eval_envs = EnvFactory.create_env(
        num_envs=num_eval_envs,
        env_config=env_config,
        requires_visuals=True,
        cuda_graph=build_cuda_graph,
        device=device,
    )
    return envs, eval_envs


def create_dep_explorer(
        dep_config: DEPConfig,
        num_muscles: int,
        num_worlds: int,
        device: torch.device,
        use_dep: bool
) -> DEPExplorer:
    return DEPExplorer(
        dep_config=dep_config,
        num_muscles=num_muscles,
        num_worlds=num_worlds,
        device=device,
        use_dep=use_dep,
    )


def main():
    # wp.clear_kernel_cache()  # can't risk caching issues

    # Restore original HOME after Warp has initialized (for wandb and other tools)
    if 'ORIG_HOME' in os.environ:
        os.environ['HOME'] = os.environ['ORIG_HOME']

    args = get_args()
    set_seed(args.seed)
    pretty_print_base_args(args)

    assert torch.cuda.is_available(), "CUDA device not available"
    device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)
    wp.set_device(device_str)

    env_config = args.env_config
    td3_config = args.td3_config
    sac_config = args.sac_config
    qflex_config = args.qflex_config
    qflex_ref_config = args.qflex_ref_config
    ppo_config = args.ppo_config
    dep_config = args.dep_config

    if args.use_wandb:
        init_wandb_run(args)

    traj_out_folder, analytics_out_folder = args.traj_out_folder, args.analytics_out_folder
    if args.algo == "td3":
        envs, eval_envs = _build_envs(
            num_envs=td3_config.num_envs,
            num_eval_envs=td3_config.num_eval_envs,
            env_config=env_config,
            build_cuda_graph=True,
            device=device,
        )
        dep_explorer = create_dep_explorer(
            dep_config=dep_config,
            num_muscles=envs.num_muscles,
            num_worlds=envs.num_worlds,
            device=device,
            use_dep=args.use_dep,
        )
        fasttd3.train(
            td3_config=td3_config,
            envs=envs,
            eval_envs=eval_envs,
            dep_explorer=dep_explorer,
            traj_out_folder=traj_out_folder,
            analytics_out_folder=analytics_out_folder,
            exp_name=args.exp_name,
            device=device,
        )
    elif args.algo == "sac":
        envs, eval_envs = _build_envs(
            num_envs=sac_config.num_envs,
            num_eval_envs=sac_config.num_eval_envs,
            env_config=env_config,
            build_cuda_graph=True,
            device=device,
        )
        dep_explorer = create_dep_explorer(
            dep_config=dep_config,
            num_muscles=envs.num_muscles,
            num_worlds=envs.num_worlds,
            device=device,
            use_dep=args.use_dep,
        )
        fastsac.train(
            sac_config=sac_config,
            envs=envs,
            eval_envs=eval_envs,
            dep_explorer=dep_explorer,
            traj_out_folder=traj_out_folder,
            analytics_out_folder=analytics_out_folder,
            exp_name=args.exp_name,
            device=device,
        )
    elif args.algo == "qflex":  # Torch impl of QFlex
        envs, eval_envs = _build_envs(
            num_envs=qflex_config.num_envs,
            num_eval_envs=qflex_config.num_eval_envs,
            env_config=env_config,
            build_cuda_graph=True,
            device=device,
        )
        qflex.train(
            cfg=qflex_config,
            envs=envs,
            eval_envs=eval_envs,
            traj_out_folder=traj_out_folder,
            analytics_out_folder=analytics_out_folder,
            exp_name=args.exp_name,
            device=device,
        )
    elif args.algo == "qflex_jax":  # Reference JAX QFlex
        import msk_envs.train.qflex_ref.train as qflex_jax
        envs, eval_envs = _build_envs(
            num_envs=qflex_ref_config.num_envs,
            num_eval_envs=qflex_ref_config.num_eval_envs,
            env_config=env_config,
            build_cuda_graph=True,
            device=device,
        )
        qflex_jax.train(
            cfg=qflex_ref_config,
            envs=envs,
            eval_envs=eval_envs,
            traj_out_folder=traj_out_folder,
            analytics_out_folder=analytics_out_folder,
            exp_name=args.exp_name,
            device=device,
            seed=args.seed,
        )
    elif args.algo == "qflex_ref":  # Reference JAX QFlex with the OffPolicyTrainer
        import msk_envs.train.qflex_ref.train_ref as qflex_ref
        envs, eval_envs = _build_envs(
            num_envs=qflex_ref_config.num_envs,
            num_eval_envs=qflex_ref_config.num_eval_envs,
            env_config=env_config,
            build_cuda_graph=True,
            device=device,
        )
        qflex_ref.train(
            cfg=qflex_ref_config,
            envs=envs,
            eval_envs=eval_envs,
            traj_out_folder=traj_out_folder,
            analytics_out_folder=analytics_out_folder,
            exp_name=args.exp_name,
            device=device,
            seed=args.seed,
        )
    elif args.algo == "ppo":
        envs, eval_envs = _build_envs(
            num_envs=ppo_config.num_envs,
            num_eval_envs=ppo_config.num_eval_envs,
            env_config=env_config,
            build_cuda_graph=True,
            device=device,
        )
        ppo.train(
            cfg=ppo_config,
            envs=envs,
            eval_envs=eval_envs,
            traj_out_folder=traj_out_folder,
            analytics_out_folder=analytics_out_folder,
            exp_name=args.exp_name,
            device=device,
        )


if __name__ == "__main__":
    main()
