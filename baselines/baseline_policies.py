"""
baseline_policies.py  –  Reference Baselines for Comparison
=============================================================
Implements two deterministic baselines against which the PPO agent is compared:

1. StaticEqualAllocation
   ----------------------
   All BSs always active, equal RB share, fixed power level.
   This represents a conventional "always-on" network with uniform resource
   assignment – a common industrial default.
   Action: rb = 1/N per BS, power = 0.5 (half maximum), all BSs on.

2. GreedyLoadProportional
   -----------------------
   Resource blocks allocated in proportion to observed traffic load.
   Power scaled proportionally to load. BSs with load < threshold are turned off.
   This is a lightweight heuristic approximating operator rule-based management.
   Action at time t:
       rb_i   = λ_i / Σ_j λ_j          (proportional RB share)
       s_i    = min(λ_i * power_cap, 1)  (load-proportional power)
       a_i    = 1[λ_i > load_threshold]  (threshold-based sleep)

Both baselines operate on the same ORANEnv without any learned parameters,
providing a lower bound and a competitive heuristic for comparison.
"""

import numpy as np
from typing import Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environment.oran_env import ORANEnv


# ─────────────────────────────────────────────────────────────────────────────

class StaticEqualAllocation:
    """
    Baseline 1: Static equal resource allocation across all BSs.

    Every BS remains permanently active with identical RB fraction and
    a fixed mid-range power level.  Represents a naive 'always-on' policy.
    """

    def __init__(self, N_bs: int, power_level: float = 0.5):
        self.N = N_bs
        self.power_level = power_level

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Return a fixed action regardless of the observation."""
        rb     = np.full(self.N, 1.0 / self.N, dtype=np.float32)
        power  = np.full(self.N, self.power_level, dtype=np.float32)
        active = np.ones(self.N, dtype=np.float32)
        return np.concatenate([rb, power, active])


class GreedyLoadProportional:
    """
    Baseline 2: Greedy load-proportional resource allocation.

    Allocates RBs proportional to current traffic load inferred from the
    first N elements of the observation vector (traffic load per BS).
    BSs with load below a threshold are put to sleep.
    """

    def __init__(
        self,
        N_bs: int,
        load_obs_slice: slice | None = None,
        sleep_threshold: float = 0.15,
        power_cap: float = 0.9,
    ):
        self.N               = N_bs
        self.obs_slice       = load_obs_slice or slice(0, N_bs)  # traffic in obs
        self.sleep_threshold = sleep_threshold
        self.power_cap       = power_cap

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        load = np.array(obs[self.obs_slice], dtype=np.float64)

        # Active mask: sleep lightly loaded BSs
        active = (load >= self.sleep_threshold).astype(np.float32)
        if active.sum() == 0:                          # keep at least one alive
            active[np.argmax(load)] = 1.0

        # Proportional RB allocation among active BSs
        active_load = load * active
        total_load  = active_load.sum() + 1e-9
        rb          = (active_load / total_load).astype(np.float32)

        # Load-proportional power scaling
        power = np.clip(active_load * self.power_cap / (load.max() + 1e-9), 0, 1).astype(np.float32)

        return np.concatenate([rb, power, active])


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(policy, env: ORANEnv, n_episodes: int = 30) -> dict:
    """
    Roll out *policy* on *env* for *n_episodes* and collect metrics.

    Parameters
    ----------
    policy    : object with .select_action(obs) -> np.ndarray
    env       : ORANEnv instance
    n_episodes: int

    Returns
    -------
    dict with keys:
        mean_reward        – mean total episode reward
        std_reward         – standard deviation of episode rewards
        mean_qos           – mean per-step QoS satisfaction ratio
        mean_energy_norm   – mean normalised energy consumption
        mean_violations    – mean violations per episode
        qos_satisfaction   – fraction of steps with zero violations
        mean_active_bs     – mean number of active BSs
        all_rewards        – list of per-episode total rewards
        all_qos            – list of per-episode mean QoS
        all_energy         – list of per-episode mean normalised energy
    """
    all_rewards, all_qos, all_energy, all_violations, all_active = [], [], [], [], []

    for ep in range(n_episodes):
        obs    = env.reset()
        done   = False
        ep_rew, ep_qos, ep_enrg, ep_viol, ep_act = 0.0, [], [], [], []

        while not done:
            action              = policy.select_action(obs)
            obs, rew, done, info = env.step(action)
            ep_rew  += rew
            ep_qos.append(info["qos_score"])
            ep_enrg.append(info["energy_norm"])
            ep_viol.append(info["n_violations"])
            ep_act.append(info["active_bs"])

        all_rewards.append(ep_rew)
        all_qos.append(float(np.mean(ep_qos)))
        all_energy.append(float(np.mean(ep_enrg)))
        all_violations.append(int(np.sum(ep_viol)))
        all_active.append(float(np.mean(ep_act)))

    qos_satisfaction = float(
        np.mean([v == 0 for v in all_violations])
    )

    return {
        "mean_reward":      float(np.mean(all_rewards)),
        "std_reward":       float(np.std(all_rewards)),
        "mean_qos":         float(np.mean(all_qos)),
        "mean_energy_norm": float(np.mean(all_energy)),
        "mean_violations":  float(np.mean(all_violations)),
        "qos_satisfaction": qos_satisfaction,
        "mean_active_bs":   float(np.mean(all_active)),
        "all_rewards":      all_rewards,
        "all_qos":          all_qos,
        "all_energy":       all_energy,
    }
