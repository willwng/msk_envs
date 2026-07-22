"""QFlex (FlowExp) using the *reference* ``OffPolicyTrainer``. """

import os
from pathlib import Path

# Ordering: package __init__ sets XLA env vars, then jax_compat installs the shim,
# both before any `relax` import.
import msk_envs.train.qflex_ref  # noqa: F401
from msk_envs.train.qflex_ref import jax_compat  # noqa: F401

import jax
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from tensorboardX import SummaryWriter

from relax.algorithm.flow_exp import FlowExp
from relax.network.flow import create_flow_net
from relax.buffer import TreeBuffer
from relax.utils.experience import Experience
from relax.trainer.off_policy import OffPolicyTrainer

from msk_envs.train.qflex_ref import bridge
from msk_envs.train.qflex_ref.qflex_ref_config import QFlexRefConfig
from msk_envs.train.qflex_ref.vec_env import MSKVectorEnv
from msk_envs.utils.logged_sim import LoggedSim


class _InProcessEvaluator:
    """Stand-in for the reference subprocess evaluator. """

    def __init__(self, eval_fn):
        self._eval_fn = eval_fn
        self.stdin = self

    def write(self, data):
        try:
            sample_step = int(bytes(data).decode().split(",")[0])
        except Exception:
            return
        self._eval_fn(sample_step)

    def close(self):
        pass

    def wait(self):
        pass


class MSKOffPolicyTrainer(OffPolicyTrainer):
    """Reference OffPolicyTrainer with in-process (msk LoggedSim) evaluation."""

    def __init__(self, *args, eval_envs, device, traj_out_folder, analytics_out_folder, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_envs = eval_envs
        self._device = device
        self._traj_out = traj_out_folder
        self._analytics_out = analytics_out_folder

    def setup(self, dummy_data: Experience):
        # Reference setup, minus the subprocess evaluator and policy-structure dump.
        self.algorithm.warmup(dummy_data)
        self.logger = SummaryWriter(str(self.log_path))
        self.progress = tqdm(total=self.total_step, desc="Sample Step", dynamic_ncols=True)
        self.evaluator = _InProcessEvaluator(self._run_eval)

    @torch.no_grad()
    def _run_eval(self, sample_step: int):
        algo = self.algorithm
        sim = LoggedSim(self._eval_envs, device=self._device)
        obs = sim.reset()
        for _ in range(sim.max_env_steps):
            # Public get_deterministic_action is broken upstream (jnp.random.key);
            # call the working jitted internal directly.
            a = algo._get_deterministic_action(
                algo.get_policy_params(), algo.get_policy_state(), bridge.torch_to_jax(obs)
            )
            actions = torch.as_tensor(np.asarray(a), device=self._device, dtype=torch.float32)
            finished, obs = sim.step(actions)
            if finished:
                break
        ret = sim.get_rewards_mean().item()
        length = sim.get_episode_length_mean().item()
        self.add_scalar("evaluate/return", ret, sample_step)
        self.add_scalar("evaluate/length", length, sample_step)
        os.makedirs(self._traj_out, exist_ok=True)
        os.makedirs(self._analytics_out, exist_ok=True)
        sim.save_animation(self._traj_out, str(sample_step), use_gzip=True)
        sim.save_frame_data(self._analytics_out, f"frame_data_{sample_step}", use_gzip=True)
        sim.save_analytics(self._analytics_out, f"analytics_{sample_step}")
        logger.info(f"[eval @ {sample_step}] return={ret:.3f} length={length:.1f}")


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
    vec_env = MSKVectorEnv(envs, device)
    obs_dim, act_dim = vec_env.obs_dim, vec_env.act_dim

    # --- reference network + algorithm construction ---
    key = jax.random.key(seed)
    net_key, run_key = jax.random.split(key)
    agent, params = create_flow_net(
        net_key, obs_dim, act_dim, tuple(cfg.hidden_sizes),
        num_timesteps=cfg.num_flow_steps, use_bn=cfg.use_bn, learn_reference_gn=cfg.learn_reference_gn,
    )
    algorithm = FlowExp(
        agent, params,
        gamma=cfg.gamma, lr=cfg.learning_rate, alpha_lr=cfg.alpha_lr, tau=cfg.tau,
        reward_scale=cfg.reward_scale, grad_step_size=cfg.grad_step_size, grad_step_num=cfg.grad_step_num,
    )
    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=cfg.buffer_size, seed=seed)

    trainer = MSKOffPolicyTrainer(
        env=vec_env,
        algorithm=algorithm,
        buffer=buffer,
        log_path=Path(f"models/{exp_name}"),
        batch_size=cfg.batch_size,
        start_step=cfg.start_env_steps,
        total_step=cfg.num_learning_iterations,
        sample_per_iteration=1,
        update_per_iteration=cfg.num_updates,
        save_policy_every=cfg.save_interval,
        warmup_with="random",
        eval_envs=eval_envs,
        device=device,
        traj_out_folder=traj_out_folder,
        analytics_out_folder=analytics_out_folder,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, cfg.batch_size))
    logger.info(
        f"Running reference OffPolicyTrainer: num_envs={cfg.num_envs} "
        f"total_step={cfg.num_learning_iterations} transitions "
        f"batch_size={cfg.batch_size} updates/iter={cfg.num_updates}"
    )
    trainer.run(run_key)
