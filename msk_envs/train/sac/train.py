import math
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from loguru import logger
from tensordict import TensorDict
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from msk_envs.train.sac.sac_config import SACConfig
from msk_envs.train.sac.sac_utils import save_params
from msk_envs.train.nets.buffer import SimpleReplayBuffer, sample_and_prepare_batches, collect_experience
from msk_envs.train.nets.distributional_critic import DistributionalCritic
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.nets.optimizer import make_optimizer
from msk_envs.train.nets.gaussian_policy import GaussianPolicy, load_policy
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.utils.train_utils import mark_step, TensorAverageMeterDict, LoggingHelper

torch.set_float32_matmul_precision("high")


def train(
        sac_config: SACConfig,
        envs,
        eval_envs,
        dep_explorer,
        traj_out_folder: str,
        analytics_out_folder: str,
        exp_name: str,
        device: torch.device,
):
    amp_enabled = sac_config.amp
    amp_device_type = "cuda"
    amp_dtype = torch.bfloat16 if sac_config.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    writer = TensorboardSummaryWriter(
        log_dir=f"models/{exp_name}",
        flush_secs=10
    )
    logging_helper = LoggingHelper(
        writer,
        log_dir=f"models/{exp_name}",
        device=device,
        num_envs=sac_config.num_envs,
        num_steps_per_env=sac_config.logging_interval,
        num_learning_iterations=sac_config.num_learning_iterations,
        is_main_process=True,
        num_gpus=1,
    )

    n_act = envs.num_actions()
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]
    if sac_config.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    action_low, action_high = envs.action_range
    action_scale = torch.ones(n_act, device=device) * (action_high - action_low) / 2.0
    action_bias = torch.zeros(n_act, device=device) + (action_high + action_low) / 2.0

    actor = GaussianPolicy(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=sac_config.num_envs,
        hidden_dim=sac_config.actor_hidden_dim,
        log_std_max=sac_config.log_std_max,
        log_std_min=sac_config.log_std_min,
        use_tanh=sac_config.use_tanh,
        use_layer_norm=sac_config.use_layer_norm,
        device=device,
        action_scale=action_scale,
        action_bias=action_bias,
    )
    policy = actor.explore

    qnet = DistributionalCritic(
        n_obs=n_obs,
        n_act=n_act,
        num_atoms=sac_config.num_atoms,
        v_min=sac_config.v_min,
        v_max=sac_config.v_max,
        hidden_dim=sac_config.critic_hidden_dim,
        use_layer_norm=sac_config.use_layer_norm,
        num_q_networks=sac_config.num_q_networks,
        device=device,
    )

    qnet_target = DistributionalCritic(
        n_obs=n_obs,
        n_act=n_act,
        num_atoms=sac_config.num_atoms,
        v_min=sac_config.v_min,
        v_max=sac_config.v_max,
        hidden_dim=sac_config.critic_hidden_dim,
        use_layer_norm=sac_config.use_layer_norm,
        num_q_networks=sac_config.num_q_networks,
        device=device,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = make_optimizer(
        model=qnet,
        lr=sac_config.critic_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=sac_config.weight_decay,
        use_soap=False,
    )
    actor_optimizer = make_optimizer(
        model=actor,
        lr=sac_config.actor_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=sac_config.weight_decay,
        use_soap=False,
    )

    target_entropy = -n_act * sac_config.target_entropy_ratio
    log_alpha = torch.tensor([math.log(sac_config.alpha_init)], requires_grad=True, device=device)
    alpha_optimizer = optim.AdamW([log_alpha], lr=sac_config.alpha_learning_rate, fused=True, betas=(0.9, 0.95))

    rb = SimpleReplayBuffer(
        n_env=sac_config.num_envs,
        buffer_size=sac_config.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_steps=sac_config.num_steps,
        gamma=sac_config.gamma,
        device=device,
    )

    @contextmanager
    def _maybe_amp():
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=sac_config.amp):
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

    def _update_main(data):
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
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                discount = sac_config.gamma ** data["next"]["effective_n_steps"]

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards - discount * bootstrap * log_alpha.exp() * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if sac_config.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=sac_config.max_grad_norm if sac_config.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        alpha_loss = torch.tensor(0.0, device=device)
        if sac_config.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with _maybe_amp():
                alpha_loss = (-log_alpha.exp() * (next_state_log_probs.detach() + target_entropy)).mean()
            scaler.scale(alpha_loss).backward()

            scaler.unscale_(alpha_optimizer)
            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
        )

    def _update_pol(data: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with _maybe_amp():
            critic_observations = data["observations"]
            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            # For logging, this is a bit wasteful though, but could be useful
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                # Compute policy entropy (negative log probability)
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (log_alpha.exp().detach() * log_probs - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if sac_config.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=sac_config.max_grad_norm if sac_config.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _checkpoint_metadata(iteration: int | None = None) -> dict:
        metadata = {"iteration": int(iteration)}
        return metadata

    def save(path: str) -> None:  # type: ignore[override]
        save_params(
            global_step,
            actor,
            qnet,
            qnet_target,
            log_alpha,
            obs_normalizer,
            actor_optimizer,
            q_optimizer,
            alpha_optimizer,
            scaler,
            sac_config,
            path,
            save_fn=logging_helper.save_checkpoint_artifact,
            metadata=_checkpoint_metadata(iteration=global_step),
        )

    if sac_config.compile:
        update_main = torch.compile(_update_main)
        update_pol = torch.compile(_update_pol)
        policy = torch.compile(policy)
    else:
        update_main = _update_main
        update_pol = _update_pol
        policy = policy

    @torch._dynamo.disable
    def normalize_obs(x):
        return obs_normalizer.forward(x)

    global_step = 0
    if sac_config.checkpoint_path:
        torch_checkpoint = torch.load(sac_config.checkpoint_path, map_location=device, weights_only=False)
        # Handle DDP-wrapped models
        actor_state_dict = torch_checkpoint["actor_state_dict"]
        qnet_state_dict = torch_checkpoint["qnet_state_dict"]
        actor.load_state_dict(actor_state_dict)
        qnet.load_state_dict(qnet_state_dict)

        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        log_alpha.data.copy_(torch_checkpoint["log_alpha"].to(device))
        actor_optimizer.load_state_dict(torch_checkpoint["actor_optimizer_state_dict"])
        q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        alpha_optimizer.load_state_dict(torch_checkpoint["alpha_optimizer_state_dict"])
        scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])
        global_step = torch_checkpoint["global_step"]

    obs = envs.reset()
    dones = None
    training_metrics = TensorAverageMeterDict()
    latest_model_path = None

    # Initialize metrics that might not be updated every step
    policy_entropy = torch.tensor(0.0, device=device)
    action_std = torch.tensor(0.0, device=device)
    actor_loss = torch.tensor(0.0, device=device)
    actor_grad_norm = torch.tensor(0.0, device=device)
    pbar = tqdm.tqdm(total=sac_config.num_learning_iterations, initial=global_step)

    while global_step <= sac_config.num_learning_iterations:
        mark_step()
        with logging_helper.record_collection_time():
            with torch.no_grad(), _maybe_amp():
                norm_obs = normalize_obs(obs)
                actions = policy(obs=norm_obs, dones=dones)
                # DEP EXPLORATION
                actions = dep_explorer.explore(muscle_states=envs.muscle_norm_fiber_lengths, actions=actions)

            next_obs, rewards, terminated, truncations, info = envs.step(actions.float())
            collect_experience(
                rb=rb, obs=obs, actions=actions, next_obs=next_obs, rewards=rewards,
                terminated=terminated, truncations=truncations, info=info,
            )

            # Update episode stats using logging helper
            dones = (terminated + truncations).bool()
            logging_helper.update_episode_stats(rewards, dones)

            obs = next_obs

        # NOTE: args.batch_size is the global batch size
        batch_size = max(sac_config.batch_size // sac_config.num_envs, 1)
        # Wait until the replay buffer has collected enough transitions before learning.
        if rb.ptr >= sac_config.learning_starts:
            with logging_helper.record_learn_time():
                # Use batched sampling: sample once, normalize once, split into updates
                prepared_batches = sample_and_prepare_batches(
                    rb=rb, obs_normalizer=normalize_obs,
                    num_updates=sac_config.num_updates, target_batch_size=batch_size
                )
                for i, data in enumerate(prepared_batches):
                    # Data is already normalized, just run the updates
                    buffer_rewards, critic_grad_norm, qf_loss, qf_max, qf_min, alpha_loss = update_main(data)
                    if sac_config.num_updates > 1:
                        if i % sac_config.policy_frequency == 1:
                            actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)
                    elif global_step % sac_config.policy_frequency == 0:
                        actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)

                    # Accumulate training metrics for smoother logging
                    current_metrics = {
                        "actor_loss": actor_loss,
                        "qf_loss": qf_loss,
                        "qf_max": qf_max,
                        "qf_min": qf_min,
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "buffer_rewards": buffer_rewards,
                        "alpha_loss": alpha_loss,
                        "alpha_value": log_alpha.exp().detach().mean(),
                        "policy_entropy": policy_entropy,
                        "action_std": action_std,
                    }

                    # Log raw reward terms before lambda multiplication
                    raw_rewards_dict = {}
                    for reward_name, reward_tensor in info["raw_rewards"].items():
                        raw_rewards_dict[f"{reward_name}_raw"] = reward_tensor.mean()

                    training_metrics.add(current_metrics)

                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - sac_config.tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=sac_config.tau)

            if global_step % sac_config.logging_interval == 0:
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

            if sac_config.save_interval > 0 and global_step > 0 and global_step % sac_config.save_interval == 0:
                logger.info(f"Saving model at global step {global_step}")
                latest_model_path = f"models/{exp_name}/{exp_name}_{global_step}.pt"
                save(latest_model_path)
                # self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{global_step:07d}.onnx"))

            if global_step % sac_config.eval_freq == 0 and latest_model_path is not None:
                logger.info(f"Evaluating at global step {global_step}")
                eval_avg_return, eval_avg_length = evaluate(latest_model_path)
                # Todo: log eval metrics
                logger.info(f"Eval Average Return: {eval_avg_return}, Eval Average Length: {eval_avg_length}")

        # Avoid global_step being incremented beyond args.num_learning_iterations, so that the final checkpoint is
        # saved at exactly args.num_learning_iterations. In the `while` condition, we check for self.global_step <=
        # args.num_learning_iterations, so that we have complete logging data at the final step too (assuming
        # `args.num_learning_iterations` is a multiple of `args.logging_interval`).
        if global_step >= sac_config.num_learning_iterations:
            break
        global_step += 1
        pbar.update(1)
