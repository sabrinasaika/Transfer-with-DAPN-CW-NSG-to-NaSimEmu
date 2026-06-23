"""Apply trained DAPN translator G to NaSim KC+ctx observations."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from envs.kill_chain import KC_DIM, CTX_DIM
from models.dapn_nets import load_translator


class DAPNTranslatorWrapper(gym.ObservationWrapper):
    """Map NaSim 26-D KC+ctx → CW-like 26-D (or 36-D) via translator G."""

    def __init__(self, env: gym.Env, translator_path: str, device: str = "cpu"):
        super().__init__(env)
        self.device = device
        self.G = load_translator(translator_path, device=device)
        out_dim = self.G.output_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(out_dim + (0 if out_dim > KC_DIM + CTX_DIM else 0),),
            dtype=np.float32,
        )
        # If output is 36-D raw CW, no ctx append; if 26-D, keep ctx from translated KC part
        if out_dim == 36:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(36,), dtype=np.float32)
            self._raw_cw = True
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(KC_DIM + CTX_DIM,), dtype=np.float32)
            self._raw_cw = False

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            t = self.G(torch.from_numpy(obs).unsqueeze(0).to(self.device)).cpu().numpy()[0]
        if self._raw_cw:
            return np.clip(t, 0.0, 1.0).astype(np.float32)
        # 26-D: translator maps full vector; clip KC portion
        return np.clip(t, 0.0, 1.0).astype(np.float32)
