"""
ppo_policy_wrapper.py  –  PPO Inference Wrapper
================================================
Wraps the trained SB3 PPO model with the same .select_action(obs) interface
as the baseline policies, enabling drop-in comparison in evaluate_policy().
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


class PPOPolicyWrapper:
    """
    Thin wrapper around a trained SB3 PPO model for evaluation.

    Parameters
    ----------
    model      : stable_baselines3.PPO  – trained model
    vec_norm   : VecNormalize | None    – normalisation stats (must be provided
                                          if obs were normalised during training)
    deterministic : bool  – use deterministic (greedy) action selection
    """

    def __init__(self, model: PPO, vec_norm=None, deterministic: bool = True):
        self.model         = model
        self.vec_norm      = vec_norm
        self.deterministic = deterministic

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_2d = obs.reshape(1, -1)

        if self.vec_norm is not None:
            obs_2d = self.vec_norm.normalize_obs(obs_2d)

        action, _ = self.model.predict(obs_2d, deterministic=self.deterministic)
        return action.flatten().astype(np.float32)
