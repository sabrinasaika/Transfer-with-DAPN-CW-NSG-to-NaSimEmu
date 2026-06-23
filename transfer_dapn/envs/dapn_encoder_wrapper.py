"""Apply trained DAPN encoder: KC+ctx → policy input (dims from checkpoint)."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.kill_chain import POLICY_DIM as _DEFAULT_POLICY_DIM
from models.encoder import load_encoder, encode_obs


class DAPNEncoderWrapper(gym.ObservationWrapper):
    """Wrap KC+ctx obs → encoded policy input (dims from encoder checkpoint)."""

    def __init__(self, env: gym.Env, encoder_path: str, device: str = "cpu",
                 mask_is_target: bool = True):
        super().__init__(env)
        self.device = device
        self.mask_is_target = mask_is_target
        self.encoder = load_encoder(encoder_path, device=device)
        policy_dim = getattr(self.encoder, "policy_dim", _DEFAULT_POLICY_DIM)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(policy_dim,), dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return encode_obs(
            self.encoder, obs, device=self.device,
            mask_is_target=self.mask_is_target,
        )
