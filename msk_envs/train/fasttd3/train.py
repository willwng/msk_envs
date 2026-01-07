import os
# Set Warp cache directory before importing warp to ensure it uses the correct location
if 'WARP_CACHE_DIR' not in os.environ:
    import tempfile
    os.environ['WARP_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), f'warp_cache_{os.getuid()}')
    os.makedirs(os.environ['WARP_CACHE_DIR'], exist_ok=True)

from msk_envs.envs import EnvFactory, EnvConfig
from msk_envs.utils.logged_sim import LoggedSim

from .buffer import SimpleReplayBuffer
from msk_envs.nets.normalizers import EmpiricalNormalization, RewardNormalizer
from msk_envs.nets.networks import Actor, Critic, load_policy
from msk_envs.nets.simba import SimbaActor, SimbaCritic
from msk_envs.train.fasttd3.hyperparams import get_args, pretty_print_base_args
from msk_envs.utils.train_utils import mark_step, save_params, set_seed

import math
import time
import torch
import tqdm
import wandb
import warp as wp

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.amp import autocast, GradScaler
from tensordict import TensorDict, from_module

torch.set_float32_matmul_precision("high")


def main():
    wp.clear_kernel_cache()  # can't risk caching issues
    
    # Restore original HOME after Warp has initialized (for wandb and other tools)
    if 'ORIG_HOME' in os.environ:
        os.environ['HOME'] = os.environ['ORIG_HOME']

    args = get_args()
    pretty_print_base_args(args)

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=args.exp_name,
            config=vars(args),
            save_code=True,
        )

    set_seed(args.seed)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # TODO: Everything should be configurable from args! Make it more general!
    env_config = EnvConfig(
        env_variant=args.env_variant,
        reward_lambdas=args.get_reward_lambdas(),
        imitation_weights=args.get_imitation_weights(),
        extra_rewarded_joints=args.extra_rewarded_joints,
        lambda_extra_rewarded_joints=args.lambda_extra_rewarded_joints,
        extra_rewarded_dofs=args.extra_rewarded_dofs,
        lambda_extra_rewarded_dofs=args.lambda_extra_rewarded_dofs,
    )

    envs = EnvFactory.create_env(
        num_envs=args.num_envs,
        env_config=env_config,
        render=args.render,
        cuda_graph=args.cuda,
        device=device,
    )
    eval_envs = EnvFactory.create_env(
        num_envs=args.num_eval_envs,
        env_config=env_config,
        render=args.render,
        cuda_graph=args.cuda,
        device=device,
    )

    n_act = envs.num_actions()
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]
    n_critic_obs = n_obs
    action_low, action_high = envs.action_range

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    if args.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=args.gamma,
            device=device,
            g_max=min(abs(args.v_min), abs(args.v_max)),
        )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
        "std_min": args.std_min,
        "std_max": args.std_max,
        "use_gsde": args.use_gsde,
        "gsde_steps": args.gsde_steps,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if args.agent == "simbav2":
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
        actor_cls = SimbaActor
        critic_cls = SimbaCritic
    elif args.agent == "fasttd3":
        actor_cls = Actor
        critic_cls = Critic
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    actor = actor_cls(**actor_kwargs)

    actor_detach = actor_cls(**actor_kwargs)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=args.critic_learning_rate_end,
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=args.actor_learning_rate_end,
    )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(model_path: str):
        policy_eval = load_policy(model_path).to(device=device)

        # Calculate max episode steps from duration and delta_t
        max_episode_steps = int(env_config.max_episode_duration / env_config.delta_t)

        # Build logged sim wrapper
        sim = LoggedSim(eval_envs, max_episode_steps, device=device)
        eval_obs = sim.reset()
        for i in range(max_episode_steps):
            eval_actions = policy_eval(eval_obs)
            finished, eval_obs = sim.step(eval_actions)
            if finished:
                break

        rewards_mean = sim.get_rewards_mean()
        episode_length_mean = sim.get_episode_length_mean()

        # Save analytics
        out_folder = args.traj_out_folder
        os.makedirs(out_folder, exist_ok=True)
        sim.save_animation(out_folder, str(global_step))

        out_folder = args.analytics_out_folder
        os.makedirs(out_folder, exist_ok=True)
        sim.save_frame_data(out_folder, f"frame_data_{global_step}")
        sim.save_analytics(out_folder, f"analytics_{global_step}")

        # Restore back to training device
        actor.to(device=device)
        obs_normalizer.to(device=device)
        return rewards_mean.item(), episode_length_mean.item()

    def update_main(data, logs_dict):
        with autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = observations
            next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = data["observations"]
            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        # Default settings are kept the same, but can now be overridden via args.
        compile_mode = args.compile_mode
        compile_backend = args.compile_backend

        update_main = torch.compile(
            update_main,
            mode=compile_mode,
            backend=compile_backend,
        )
        update_pol = torch.compile(
            update_pol,
            mode=compile_mode,
            backend=compile_backend,
        )
        policy = torch.compile(
            policy,
            mode=None,
            backend=compile_backend,
        )

        # Don't compile normalize_obs to avoid Triton compilation issues
        @torch._dynamo.disable
        def normalize_obs(x):
            return obs_normalizer.forward(x)

        if args.reward_normalization:
            update_stats = torch.compile(
                reward_normalizer.update_stats,
                mode=None,
                backend=compile_backend,
            )
        normalize_reward = torch.compile(
            reward_normalizer.forward,
            mode=None,
            backend=compile_backend,
        )
    else:
        normalize_obs = obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    obs = envs.reset()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        # global_step = torch_checkpoint["global_step"]
        global_step = 0
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    latest_model_path = None

    while global_step < args.total_timesteps:
        mark_step()
        logs_dict = TensorDict()
        if start_time is None and global_step >= args.learning_starts:
            start_time = time.time()

        with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, terminated, truncations, info = envs.step(actions)
        dones = (terminated + truncations).bool()

        if args.reward_normalization:
            update_stats(rewards, dones.float())

        final_obs = info["final_observation"]
        true_next_obs = torch.where(
            dones[:, None] > 0, final_obs, next_obs
        )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_worlds,),
            device=device,
        )
        rb.extend(transition)

        obs = next_obs

        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                raw_rewards = data["next"]["rewards"]
                data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_main(data, logs_dict)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)

                soft_update(qnet, qnet_target, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "rewards/total": rewards.mean(),
                    }

                    # Log raw reward terms before lambda multiplication
                    for reward_name, reward_tensor in info["raw_rewards"].items():
                        logs[f"rewards/{reward_name}_raw"] = reward_tensor.mean()

                    if global_step % args.eval_freq == 0 and latest_model_path is not None:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate(latest_model_path)
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length

                if args.use_wandb:
                    wandb.log(
                        {
                            "frame": global_step * args.num_envs,
                            "critic_lr": q_scheduler.get_last_lr()[0],
                            "actor_lr": actor_scheduler.get_last_lr()[0],
                            **logs,
                        },
                        step=global_step,
                    )

            if global_step > 0 and global_step % args.save_interval == 0:
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    args,
                    f"models/{args.exp_name}/{args.exp_name}_{global_step}.pt",
                )
                latest_model_path = f"models/{args.exp_name}/{args.exp_name}_{global_step}.pt"

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)


if __name__ == "__main__":
    main()
