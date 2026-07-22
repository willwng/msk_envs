import os

# The package __init__ sets XLA memory env vars, then jax_compat
# installs the jax.core shim. Both must run BEFORE any `relax` import.
import msk_envs.train.qflex_ref  # noqa: F401
from msk_envs.train.qflex_ref import jax_compat  # noqa: F401

import jax
import numpy as np
import torch
import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from relax.algorithm.flow_exp import FlowExp
from relax.network.flow import create_flow_net
from relax.utils.experience import Experience

from msk_envs.train.nets.buffer import (
    SimpleReplayBuffer,
    collect_experience,
    sample_and_prepare_batches,
)
from msk_envs.train.qflex_ref import bridge
from msk_envs.train.qflex_ref.qflex_ref_config import QFlexRefConfig
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.utils.train_utils import TensorAverageMeterDict, LoggingHelper


def train(
        cfg: QFlexRefConfig,
        envs,
        eval_envs,
        traj_out_folder: str,
        analytics_out_folder: str,
        exp_name: str,
        device: torch.device,
        seed: int = 0,
):
    # ------------------------------------------------------------------ logging
    writer = TensorboardSummaryWriter(log_dir=f"models/{exp_name}", flush_secs=10)
    logging_helper = LoggingHelper(
        writer,
        log_dir=f"models/{exp_name}",
        device=device,
        num_envs=cfg.num_envs,
        num_steps_per_env=cfg.logging_interval,
        num_learning_iterations=cfg.num_learning_iterations,
        is_main_process=True,
        num_gpus=1,
    )
    training_metrics = TensorAverageMeterDict()

    # ------------------------------------------------------------------ envs
    n_obs, n_act = envs.num_obs(), envs.num_actions()

    learning_starts = max(cfg.start_env_steps // cfg.num_envs, 1)
    buffer_capacity = max(cfg.buffer_size // cfg.num_envs, 1)
    batch_size = max(cfg.batch_size // cfg.num_envs, 1)  # per-env; total minibatch = batch_size * num_envs

    # --------------------------------------------------------------- algorithm
    master_key = jax.random.key(seed)
    net_key, master_key = jax.random.split(master_key)
    agent, params = create_flow_net(
        net_key,
        obs_dim=n_obs,
        act_dim=n_act,
        hidden_sizes=tuple(cfg.hidden_sizes),
        num_timesteps=cfg.num_flow_steps,
        use_bn=cfg.use_bn,
        learn_reference_gn=cfg.learn_reference_gn,
    )
    algo = FlowExp(
        agent,
        params,
        gamma=cfg.gamma,
        lr=cfg.learning_rate,
        alpha_lr=cfg.alpha_lr,
        tau=cfg.tau,
        reward_scale=cfg.reward_scale,
        grad_step_size=cfg.grad_step_size,
        grad_step_num=cfg.grad_step_num,
    )
    # Trigger JIT compilation of update / action fns with a representative batch.
    # Match the real per-update minibatch (batch_size * num_envs) so the first
    # real update doesn't retrace under a different shape.
    dummy = Experience.create_example(n_obs, n_act, batch_size=max(batch_size * cfg.num_envs, 2))
    algo.warmup(dummy)

    rb = SimpleReplayBuffer(
        n_env=cfg.num_envs,
        buffer_size=buffer_capacity,
        n_obs=n_obs,
        n_act=n_act,
        n_steps=cfg.num_steps,
        gamma=cfg.gamma,
        device=device,
    )
    obs_normalizer = torch.nn.Identity()  # reference nets use internal BatchNorm

    # --------------------------------------------------------- action helpers
    def explore(obs_torch: torch.Tensor, key: jax.Array) -> torch.Tensor:
        a_np = algo.get_action(key, bridge.torch_to_jax(obs_torch))
        return torch.as_tensor(np.asarray(a_np), device=device, dtype=torch.float32)

    def act_deterministic(obs_torch: torch.Tensor) -> torch.Tensor:
        # The public get_deterministic_action wrapper is broken upstream
        # (jnp.random.key), so call the working jitted internal directly.
        a = algo._get_deterministic_action(
            algo.get_policy_params(), algo.get_policy_state(), bridge.torch_to_jax(obs_torch)
        )
        return torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)

    # ----------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate() -> tuple[float, float]:
        sim = LoggedSim(eval_envs, device=device)
        eval_obs = sim.reset()
        for _ in range(sim.max_env_steps):
            eval_actions = act_deterministic(eval_obs)
            finished, eval_obs = sim.step(eval_actions)
            if finished:
                break
        rewards_mean = sim.get_rewards_mean()
        episode_length_mean = sim.get_episode_length_mean()
        os.makedirs(traj_out_folder, exist_ok=True)
        os.makedirs(analytics_out_folder, exist_ok=True)
        sim.save_animation(traj_out_folder, str(global_step), use_gzip=True)
        sim.save_frame_data(analytics_out_folder, f"frame_data_{global_step}", use_gzip=True)
        sim.save_analytics(analytics_out_folder, f"analytics_{global_step}")
        return rewards_mean.item(), episode_length_mean.item()

    # -------------------------------------------------------------- training
    obs = envs.reset()
    global_step = 0
    pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=global_step)
    while global_step < cfg.num_learning_iterations:
        with logging_helper.record_collection_time():
            with torch.no_grad():
                if global_step < learning_starts:
                    actions = torch.rand((cfg.num_envs, n_act), device=device) * 2.0 - 1.0
                else:
                    sample_key = jax.random.fold_in(master_key, global_step)
                    actions = explore(obs, sample_key)

            next_obs, rewards, terminated, truncations, info = envs.step(actions)
            collect_experience(
                rb=rb, obs=obs, actions=actions, next_obs=next_obs, rewards=rewards,
                terminated=terminated, truncations=truncations, info=info,
            )
            dones = (terminated + truncations).bool()
            logging_helper.update_episode_stats(rewards, dones)
            obs = next_obs

        if rb.ptr >= learning_starts:
            with logging_helper.record_learn_time():
                prepared_batches = sample_and_prepare_batches(
                    rb=rb, obs_normalizer=obs_normalizer,
                    num_updates=cfg.num_updates, target_batch_size=batch_size,
                )
                for i, data in enumerate(prepared_batches):
                    exp = bridge.build_experience(data, device=device)
                    update_key = jax.random.fold_in(master_key, global_step * cfg.num_updates + i)
                    info_metrics, _ = algo.update(update_key, exp)
                    training_metrics.add(info_metrics)

            if global_step % cfg.logging_interval == 0:
                raw_rewards_dict = {
                    f"{name}_raw": t.mean() for name, t in info["raw_rewards"].items()
                }
                with torch.no_grad():
                    loss_metrics = training_metrics.get_metrics_and_clear()
                    loss_metrics["env_rewards"] = rewards.mean().item()
                    extra_log_dicts = {
                        "raw_rewards": raw_rewards_dict,
                        "additional_metrics": envs.additional_metrics(),
                    }
                    logging_helper.post_epoch_logging(
                        it=global_step, loss_dict=loss_metrics, extra_log_dicts=extra_log_dicts
                    )

            if cfg.save_interval > 0 and global_step > 0 and global_step % cfg.save_interval == 0:
                logger.info(f"Saving policy at global step {global_step}")
                save_path = f"models/{exp_name}/policy_{global_step}.pkl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                algo.save_policy(save_path)

            if global_step % cfg.eval_freq == 0:
                logger.info(f"Evaluating at global step {global_step}")
                eval_avg_return, eval_avg_length = evaluate()
                logger.info(
                    f"Eval Average Return: {eval_avg_return}, Eval Average Length: {eval_avg_length}"
                )

        global_step += 1
        pbar.update(1)
