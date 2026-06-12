import math
import os
import signal
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from tensordict import TensorDict
from torch.amp import autocast, GradScaler
from loguru import logger
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from msk_envs.train.fasttd3.td3_config import TD3Config
from msk_envs.train.nets.buffer import SimpleReplayBuffer
from msk_envs.train.nets.normalizers import EmpiricalNormalization, RewardNormalizer
from msk_envs.train.nets.simba import SimbaActor, SimbaCritic
from msk_envs.train.nets.td3_networks import Actor, Critic, load_policy
from msk_envs.train.nets.optimizer import make_optimizer
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.utils.train_utils import mark_step, TensorAverageMeterDict, LoggingHelper, save_params_td3

torch.set_float32_matmul_precision("high")

save_requested = False


def on_sigusr1(signum, frame):
    global save_requested
    save_requested = True


def train(
        td3_config: TD3Config,
        envs,
        eval_envs,
        dep_explorer,
        traj_out_folder: str,
        analytics_out_folder: str,
        exp_name: str,
        device: torch.device,
):
    global save_requested

    amp_enabled = td3_config.amp
    amp_device_type = "cuda"
    amp_dtype = torch.bfloat16 if td3_config.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    writer = TensorboardSummaryWriter(
        log_dir=f"models/{exp_name}",
        flush_secs=10
    )
    logging_helper = LoggingHelper(
        writer,
        log_dir=f"models/{exp_name}",
        device=device,
        num_envs=td3_config.num_envs,
        num_steps_per_env=td3_config.logging_interval,
        num_learning_iterations=td3_config.num_learning_iterations,
        is_main_process=True,
        num_gpus=1,
    )

    n_act = envs.num_actions()
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]
    if td3_config.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    if td3_config.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=td3_config.gamma,
            device=device,
            g_max=min(abs(td3_config.v_min), abs(td3_config.v_max)),
        )
    else:
        reward_normalizer = nn.Identity()

    action_low, action_high = envs.action_range
    action_scale = torch.ones(n_act, device=device) * (action_high - action_low) / 2.0
    action_bias = torch.zeros(n_act, device=device) + (action_high + action_low) / 2.0

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": td3_config.num_envs,
        "hidden_dim": td3_config.actor_hidden_dim,
        "std_min": td3_config.std_min,
        "std_max": td3_config.std_max,
        "use_tanh": td3_config.use_tanh,
        "use_layer_norm": td3_config.use_layer_norm,
        "device": device,
        "use_gsde": td3_config.use_gsde,
        "gsde_steps": td3_config.gsde_steps,
    }
    critic_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_atoms": td3_config.num_atoms,
        "v_min": td3_config.v_min,
        "v_max": td3_config.v_max,
        "hidden_dim": td3_config.critic_hidden_dim,
        "use_layer_norm": td3_config.use_layer_norm,
        "num_q_networks": td3_config.num_q_networks,
        "device": device,
    }

    if td3_config.agent == "simbav2":
        actor_kwargs.pop("use_tanh")
        actor_kwargs.pop("use_layer_norm")
        critic_kwargs.pop("use_layer_norm")
        critic_kwargs.pop("num_q_networks")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / td3_config.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / td3_config.actor_hidden_dim),
                "alpha_init": 1.0 / (td3_config.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(td3_config.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": td3_config.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / td3_config.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / td3_config.critic_hidden_dim),
                "alpha_init": 1.0 / (td3_config.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(td3_config.critic_hidden_dim),
                "num_blocks": td3_config.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
        actor_cls = SimbaActor
        critic_cls = SimbaCritic
    elif td3_config.agent == "fasttd3":
        actor_cls = Actor
        critic_cls = Critic
    else:
        raise ValueError(f"Agent {td3_config.agent} not supported")

    actor = actor_cls(**actor_kwargs)
    explore = actor.explore
    explore_synergistic = actor.explore_synergistic

    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = make_optimizer(
        model=qnet,
        lr=td3_config.critic_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=td3_config.weight_decay,
        use_muon=td3_config.use_soap,
    )
    actor_optimizer = make_optimizer(
        model=actor,
        lr=td3_config.actor_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=td3_config.weight_decay,
        use_muon=td3_config.use_soap,
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=td3_config.num_learning_iterations,
        eta_min=td3_config.critic_learning_rate_end,
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=td3_config.num_learning_iterations,
        eta_min=td3_config.actor_learning_rate_end,
    )

    rb = SimpleReplayBuffer(
        n_env=td3_config.num_envs,
        buffer_size=td3_config.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_steps=td3_config.num_steps,
        gamma=td3_config.gamma,
        device=device,
    )

    @contextmanager
    def _maybe_amp():
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=td3_config.amp):
            yield

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(model_path: str):
        policy_eval = load_policy(model_path).to(device=device)

        # Build logged sim wrapper
        sim = LoggedSim(eval_envs, device=device)
        eval_obs = sim.reset()
        for _ in range(sim.max_env_steps):
            with torch.no_grad():
                eval_actions = policy_eval(eval_obs)
                finished, eval_obs = sim.step(eval_actions)

            if finished:
                break

        rewards_mean = sim.get_rewards_mean()
        episode_length_mean = sim.get_episode_length_mean()

        # Save analytics
        os.makedirs(traj_out_folder, exist_ok=True)
        sim.save_animation(traj_out_folder, str(global_step), use_gzip=True)

        os.makedirs(analytics_out_folder, exist_ok=True)
        sim.save_frame_data(analytics_out_folder, f"frame_data_{global_step}", use_gzip=True)
        sim.save_analytics(analytics_out_folder, f"analytics_{global_step}")

        # Restore back to training device
        actor.to(device=device)
        obs_normalizer.to(device=device)
        return rewards_mean.item(), episode_length_mean.item()

    def update_main(data):
        with _maybe_amp():
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = observations
            next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                clipped_noise = torch.randn_like(actions)
                clipped_noise = clipped_noise.mul(td3_config.policy_noise).clamp(-td3_config.noise_clip,
                                                                                 td3_config.noise_clip)
                next_state_actions = (actor(next_observations) + clipped_noise).clamp(action_low, action_high)
                discount = td3_config.gamma ** data["next"]["effective_n_steps"]

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
                if td3_config.use_cdq:
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
            qf1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
            qf2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if td3_config.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=td3_config.max_grad_norm if td3_config.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.mean(),
            qf_loss.mean(),
            qf1_next_target_value.mean(),
            qf1_next_target_value.min(),
        )

    def update_pol(data):
        with _maybe_amp():
            critic_observations = data["observations"]
            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if td3_config.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if td3_config.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=td3_config.max_grad_norm if td3_config.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach().item(),
            actor_loss.detach().item(),
        )

    def _sample_and_prepare_batches() -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * td3_config.num_updates
        large_data = rb.sample(large_batch_size)
        samples_per_update = batch_size * envs.num_worlds

        # Normalize all data once
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        raw_rewards = large_data["next"]["rewards"]
        large_data["next"]["rewards"] = normalize_reward(raw_rewards)

        # Split into smaller batches
        prepared_batches = []

        for i in range(td3_config.num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                },
                batch_size=samples_per_update,
            )
            prepared_batches.append(batch_data)
        return prepared_batches

    if td3_config.compile and not td3_config.use_soap:  # SOAP can't handle compile
        # Default settings are kept the same, but can now be overridden via train_config.
        compile_mode = td3_config.compile_mode
        compile_backend = td3_config.compile_backend

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
        explore = torch.compile(
            explore,
            mode=None,
            backend=compile_backend,
        )
        explore_synergistic = torch.compile(
            explore_synergistic,
            mode=None,
            backend=compile_backend,
        )

        # Don't compile normalize_obs to avoid Triton compilation issues
        @torch._dynamo.disable
        def normalize_obs(x):
            return obs_normalizer.forward(x)

        if td3_config.reward_normalization:
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
        if td3_config.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    # Register handler for pre-timeout checkpointing
    signal.signal(signal.SIGUSR1, on_sigusr1)

    global_step = 0
    if td3_config.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{td3_config.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])

        # Extract global_step from checkpoint
        if "global_step" in torch_checkpoint:
            global_step = torch_checkpoint["global_step"]

        # Load optimizer and scheduler states if available
        if "actor_optimizer_state_dict" in torch_checkpoint:
            actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        if "q_optimizer_state_dict" in torch_checkpoint:
            q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        if "actor_scheduler_state_dict" in torch_checkpoint:
            actor_scheduler.load_state_dict(torch_checkpoint["actor_scheduler_state_dict"])
        if "q_scheduler_state_dict" in torch_checkpoint:
            q_scheduler.load_state_dict(torch_checkpoint["q_scheduler_state_dict"])
        if scaler is not None and "grad_scaler_state_dict" in torch_checkpoint:
            scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])

    obs = envs.reset()
    dones = None
    training_metrics = TensorAverageMeterDict()
    latest_model_path = None

    actor_loss = torch.tensor(0.0, device=device)
    actor_grad_norm = torch.tensor(0.0, device=device)
    pbar = tqdm.tqdm(total=td3_config.num_learning_iterations, initial=global_step)

    while global_step < td3_config.num_learning_iterations:
        mark_step()
        with logging_helper.record_collection_time():
            with torch.no_grad(), _maybe_amp():
                norm_obs = normalize_obs(obs)
                if td3_config.use_synergistic_noise:
                    actions = explore_synergistic(
                        obs=norm_obs,
                        dones=dones,
                        max_isometric_force=envs.muscle_max_isometric_forces,
                        active_length_multiplier=envs.muscle_active_length_multiplier,
                        active_velocity_multiplier=envs.muscle_active_velocity_multiplier,
                        moment_arms=envs.muscle_moment_arms,
                    )
                else:
                    actions = explore(obs=norm_obs, dones=dones)

                # DEP EXPLORATION
                actions = dep_explorer.explore(muscle_states=envs.muscle_fiber_lengths, actions=actions)

            next_obs, rewards, terminated, truncations, info = envs.step(actions)
            dones = (terminated + truncations).bool()

            # Update episode stats using logging helper
            logging_helper.update_episode_stats(rewards, dones)

            if td3_config.reward_normalization:
                update_stats(rewards, dones.float())

            # Compute 'true' next_obs for saving
            true_next_obs = torch.where(dones[:, None] > 0, info["final_observation"], next_obs)
            # true_next_obs = torch.where(truncations[:, None] > 0, info["final_observation"], next_obs)

            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                    "next": {
                        "observations": true_next_obs,
                        "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                        "truncations": truncations.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(envs.num_worlds,),
                device=device,
            )
            rb.extend(transition)

            obs = next_obs

        batch_size = max(td3_config.batch_size // td3_config.num_envs, 1)
        # Wait until the replay buffer has collected enough transitions before learning.
        if rb.ptr >= td3_config.learning_starts:
            with logging_helper.record_learn_time():
                # Use batched sampling: sample once, normalize once, split into updates
                prepared_batches = _sample_and_prepare_batches()
                for i, data in enumerate(prepared_batches):
                    buffer_rewards, critic_grad_norm, qf_loss, qf_max, qf_min = update_main(data)
                    if td3_config.num_updates > 1:
                        if i % td3_config.policy_frequency == 1:
                            actor_grad_norm, actor_loss = update_pol(data)
                    elif global_step % td3_config.policy_frequency == 0:
                        actor_grad_norm, actor_loss = update_pol(data)

                    # Accumulate training metrics for smoother logging
                    current_metrics = {
                        "actor_loss": actor_loss,
                        "qf_loss": qf_loss,
                        "qf_max": qf_max,
                        "qf_min": qf_min,
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "buffer_rewards": buffer_rewards,
                    }

                    # Log raw reward terms before lambda multiplication
                    raw_rewards_dict = {}
                    for reward_name, reward_tensor in info["raw_rewards"].items():
                        raw_rewards_dict[f"{reward_name}_raw"] = reward_tensor.mean()

                    training_metrics.add(current_metrics)

                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - td3_config.tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=td3_config.tau)

            if global_step % td3_config.logging_interval == 0:
                with torch.no_grad():
                    # Use accumulated training metrics for smoother logging (reduces noise)
                    accumulated_metrics = training_metrics.mean_and_clear()

                    # Convert tensor values to float for logging
                    loss_dict = {}
                    for key, value in accumulated_metrics.items():
                        if isinstance(value, torch.Tensor):
                            loss_dict[key] = value.item()
                        else:
                            loss_dict[key] = float(value)

                    # Add current env rewards (not part of training loop accumulation)
                    loss_dict["env_rewards"] = rewards.mean().item()

                # Use logging helper
                extra_log_dicts = {
                    "raw_rewards": raw_rewards_dict,
                    "additional_metrics": envs.additional_metrics(),
                }
                logging_helper.post_epoch_logging(it=global_step, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

            if td3_config.save_interval > 0 and global_step > 0 and global_step % td3_config.save_interval == 0:
                logger.info(f"Saving model at global step {global_step}")
                latest_model_path = f"models/{exp_name}/{exp_name}_{global_step}.pt"
                save_params_td3(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    td3_config,
                    latest_model_path,
                    actor_optimizer=actor_optimizer,
                    q_optimizer=q_optimizer,
                    actor_scheduler=actor_scheduler,
                    q_scheduler=q_scheduler,
                    scaler=scaler,
                    include_optim_state=td3_config.save_optimizer_state,
                )

            if global_step % td3_config.eval_freq == 0 and latest_model_path is not None:
                print(f"Evaluating at global step {global_step}")
                eval_avg_return, eval_avg_length = evaluate(latest_model_path)
                # Todo: log eval metrics
                print(f"Eval Average Return: {eval_avg_return}, Eval Average Length: {eval_avg_length}")

        if save_requested:
            latest_model_path = f"models/{exp_name}/{exp_name}_{global_step}.pt"
            logger.info(f"SIGUSR1 received, saving checkpoint at global step {global_step} and exiting with code 42.")
            save_params_td3(
                global_step,
                actor,
                qnet,
                qnet_target,
                obs_normalizer,
                td3_config,
                latest_model_path,
                actor_optimizer=actor_optimizer,
                q_optimizer=q_optimizer,
                actor_scheduler=actor_scheduler,
                q_scheduler=q_scheduler,
                scaler=scaler,
                include_optim_state=True,
            )
            save_requested = False
            sys.exit(42)

        if global_step >= td3_config.num_learning_iterations:
            break
        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)
