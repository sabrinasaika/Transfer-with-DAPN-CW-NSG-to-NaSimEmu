"""DAPN shared encoder: game-state features → domain-invariant latent.

Input : GAME_STATE_DIM = 35  (7 slots × 5 features, is_target excluded)
Output: LATENT_DIM     = 64

After encoding, the policy receives:
  [64-D latent | 7-D is_target | 2-D ctx] = 73-D  (POLICY_DIM)

is_target bypasses the encoder in the raw pipeline. By default it is zeroed
(mask_is_target=True) so the policy cannot trivially learn "always pick slot 1".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from envs.kill_chain import (
    GAME_STATE_DIM, LATENT_DIM, POLICY_DIM, KC_DIM, CTX_DIM,
    MAX_SLOTS, extract_game_state, extract_is_target,
)


class Encoder(nn.Module):
    def __init__(self, obs_dim: int = GAME_STATE_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.obs_dim    = obs_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_encoder(path: str, device: str = "cpu") -> Encoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    obs_dim    = ckpt.get("obs_dim",    GAME_STATE_DIM)
    latent_dim = ckpt.get("latent_dim", LATENT_DIM)
    enc = Encoder(obs_dim, latent_dim)
    enc.load_state_dict(ckpt["encoder_state_dict"])
    enc.eval()
    enc.kc_dim     = ckpt.get("kc_dim",     KC_DIM)
    enc.max_slots  = ckpt.get("max_slots",  MAX_SLOTS)
    enc.ctx_dim    = ckpt.get("ctx_dim",    CTX_DIM)
    enc.policy_dim = ckpt.get(
        "policy_dim", latent_dim + enc.max_slots + enc.ctx_dim)
    return enc


def encode_obs(encoder: Encoder, kc_ctx_obs, device: str = "cpu",
               mask_is_target: bool = True) -> np.ndarray:
    """KC+ctx → [latent | is_target | ctx]; dims read from encoder checkpoint."""
    kc_dim    = getattr(encoder, "kc_dim",    KC_DIM)
    max_slots = getattr(encoder, "max_slots", MAX_SLOTS)
    ctx_dim   = getattr(encoder, "ctx_dim",   CTX_DIM)

    x = np.asarray(kc_ctx_obs, dtype=np.float32)
    kc  = x[:kc_dim]
    ctx = x[kc_dim:kc_dim + ctx_dim]
    gs        = kc.reshape(max_slots, 6)[:, :5].reshape(-1)
    is_target = kc.reshape(max_slots, 6)[:, 5]
    if mask_is_target:
        is_target = np.zeros_like(is_target)
    with torch.no_grad():
        latent = encoder(
            torch.from_numpy(gs).unsqueeze(0).to(device)
        ).cpu().numpy()[0]
    return np.concatenate([latent, is_target, ctx]).astype(np.float32)


def encode_batch(encoder: Encoder, kc_ctx_batch: np.ndarray,
                 device: str = "cpu", mask_is_target: bool = True) -> np.ndarray:
    """(N, kc+ctx) batch → (N, policy_dim) policy-input batch."""
    kc_dim    = getattr(encoder, "kc_dim",    KC_DIM)
    max_slots = getattr(encoder, "max_slots", MAX_SLOTS)
    ctx_dim   = getattr(encoder, "ctx_dim",   CTX_DIM)

    x   = np.asarray(kc_ctx_batch, dtype=np.float32)
    kc  = x[:, :kc_dim]
    ctx = x[:, kc_dim:kc_dim + ctx_dim]

    gs        = kc.reshape(-1, max_slots, 6)[:, :, :5].reshape(len(x), -1)
    is_target = kc.reshape(-1, max_slots, 6)[:, :, 5]
    if mask_is_target:
        is_target = np.zeros_like(is_target)

    with torch.no_grad():
        latent = encoder(
            torch.from_numpy(gs).to(device)
        ).cpu().numpy()
    return np.concatenate([latent, is_target, ctx], axis=1).astype(np.float32)
