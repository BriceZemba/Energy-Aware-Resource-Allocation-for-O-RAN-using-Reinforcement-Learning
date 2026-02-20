"""
train_ppo.py  –  PPO Agent Training for O-RAN Energy-Aware Resource Allocation
===============================================================================
Uses Stable-Baselines3's PPO implementation with a custom feature extractor.

Why PPO?
--------
• The action space is continuous (RB fraction, power scale, active probability),
  which suits actor-critic methods over value-based discrete methods (DQN).
• PPO's clipped surrogate objective prevents destructively large policy updates,
  which is critical in non-stationary wireless environments.
• PPO naturally handles the multi-objective trade-off via the scalarised reward
  without requiring explicit Pareto front computation.
• Sample efficiency is acceptable for simulation-based training; real-world
  deployment would move to model-based RL or offline RL pre-training.

Hyperparameter Rationale
-------------------------
learning_rate : 3e-4   – Adam LR; standard for PPO; tuned via grid search.
n_steps       : 2048   – Rollout buffer length before each gradient update;
                          large enough to capture ≥8 full episodes (max_steps=256).
batch_size    : 128    – Mini-batch size; balances gradient noise vs. compute.
n_epochs      : 10     – Number of PPO epochs per rollout; typical range [4,20].
gamma         : 0.99   – Discount factor; wireless decisions have medium horizon.
gae_lambda    : 0.95   – GAE λ; reduces variance with slight bias – standard.
clip_range    : 0.2    – ε in clipped surrogate; prevents over-optimisation.
ent_coef      : 0.01   – Entropy bonus; encourages exploration of power/RB space.
vf_coef       : 0.5    – Value function loss coefficient; standard.
max_grad_norm : 0.5    – Gradient clipping; stabilises training with large obs.
net_arch      : [256, 256] – Moderate depth sufficient for 5N-dimensional state.
"""

import os
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.oran_env import ORANEnv


# ─────────────────────────────────────────────────────────────────────────────
# Custom Callback: tracks per-episode QoS and energy statistics
# ─────────────────────────────────────────────────────────────────────────────

class ORANMetricsCallback(BaseCallback):
    """
    Records episode-level metrics (QoS score, energy, violations) during
    training for post-hoc analysis and convergence monitoring.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ep_qos_scores:   list[float] = []
        self.ep_energy_norms: list[float] = []
        self.ep_violations:   list[int]   = []
        self.ep_rewards:      list[float] = []
        self._ep_qos_buf:  list[float] = []
        self._ep_enrg_buf: list[float] = []
        self._ep_viol_buf: list[int]   = []
        self._ep_rew_buf:  list[float] = []

    def _on_step(self) -> bool:
        infos  = self.locals.get("infos", [{}])
        dones  = self.locals.get("dones", [False])
        rewards = self.locals.get("rewards", [0.0])

        for info, done, rew in zip(infos, dones, rewards):
            self._ep_qos_buf.append(info.get("qos_score", 0.0))
            self._ep_enrg_buf.append(info.get("energy_norm", 0.0))
            self._ep_viol_buf.append(info.get("n_violations", 0))
            self._ep_rew_buf.append(float(rew))

            if done:
                self.ep_qos_scores.append(float(np.mean(self._ep_qos_buf)))
                self.ep_energy_norms.append(float(np.mean(self._ep_enrg_buf)))
                self.ep_violations.append(int(np.sum(self._ep_viol_buf)))
                self.ep_rewards.append(float(np.sum(self._ep_rew_buf)))
                self._ep_qos_buf, self._ep_enrg_buf = [], []
                self._ep_viol_buf, self._ep_rew_buf = [], []

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(seed: int = 0, **env_kwargs):
    def _init():
        env = ORANEnv(seed=seed, **env_kwargs)
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    total_timesteps: int = 300_000,
    save_dir: str = "outputs/models",
    log_dir: str  = "outputs/logs",
    env_kwargs: dict | None = None,
    seed: int = 42,
) -> tuple[PPO, ORANMetricsCallback, VecNormalize]:
    """
    Train a PPO agent on the O-RAN environment.

    Parameters
    ----------
    total_timesteps : int   – Total environment interactions for training.
    save_dir        : str   – Directory to save model checkpoints.
    log_dir         : str   – TensorBoard log directory.
    env_kwargs      : dict  – Keyword arguments forwarded to ORANEnv.
    seed            : int   – Global RNG seed.

    Returns
    -------
    model    : trained PPO model
    callback : ORANMetricsCallback with collected metrics
    vec_env  : VecNormalize wrapper (needed for inference)
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    env_kwargs = env_kwargs or {}

    # ── Training environment (vectorised + observation normalisation) ────────
    train_env = DummyVecEnv([make_env(seed=seed + i, **env_kwargs) for i in range(4)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # ── Evaluation environment (unnormalised for fair metric comparison) ─────
    eval_env_raw = DummyVecEnv([make_env(seed=seed + 100, **env_kwargs)])
    eval_env     = VecNormalize(eval_env_raw, training=False, norm_reward=False)

    # ── Policy network architecture ──────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
        ortho_init=True,                   # orthogonal weight init (Huang et al.)
    )

    # ── PPO model ────────────────────────────────────────────────────────────
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 128,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = policy_kwargs,
        tensorboard_log = log_dir,
        seed            = seed,
        verbose         = 1,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    metrics_cb = ORANMetricsCallback()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = save_dir,
        log_path             = log_dir,
        eval_freq            = 10_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 0,
    )

    ckpt_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = save_dir,
        name_prefix = "ppo_oran",
    )

    # ── Training ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  PPO Training  –  O-RAN Energy-Aware Resource Allocation")
    print("=" * 65)
    t0 = time.time()

    model.learn(
        total_timesteps = total_timesteps,
        callback        = [metrics_cb, eval_cb, ckpt_cb],
        tb_log_name     = "PPO_ORAN",
        reset_num_timesteps = True,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s  ({total_timesteps/elapsed:.0f} steps/s)")

    # Save final model + normalisation stats
    model.save(os.path.join(save_dir, "ppo_oran_final"))
    train_env.save(os.path.join(save_dir, "vec_normalize.pkl"))

    return model, metrics_cb, train_env
