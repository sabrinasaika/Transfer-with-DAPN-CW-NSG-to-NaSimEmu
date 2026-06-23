"""Append last action / last reward to kill-chain observations."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.kill_chain import KC_DIM as _DEFAULT_KC_DIM, CTX_DIM


class AddContextWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, kc_dim: int | None = None):
        super().__init__(env)
        kc_dim = kc_dim if kc_dim is not None else _DEFAULT_KC_DIM
        self._kc_dim = kc_dim
        assert env.observation_space.shape == (kc_dim,)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(kc_dim + CTX_DIM,), dtype=np.float32)
        self._last_action = 0.0
        self._last_reward = 0.0

    def reset(self, **kwargs):
        self._last_action = 0.0
        self._last_reward = 0.0
        obs, info = self.env.reset(**kwargs)
        return self._append(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = float(action) / max(1.0, float(self.env.action_space.n - 1))
        self._last_reward = float(np.clip(reward / 100.0, -1.0, 1.0))
        return self._append(obs), reward, terminated, truncated, info

    def _append(self, obs: np.ndarray) -> np.ndarray:
        ctx = np.array([self._last_action, self._last_reward], dtype=np.float32)
        return np.concatenate([obs, ctx]).astype(np.float32)
