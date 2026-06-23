"""DAPN adversarial encoder with Gradient Reversal Layer (Ganin et al., 2016).

Architecture
------------
  Encoder E   : GAME_STATE_DIM (35) → LATENT_DIM (64)
  Decoder D   : LATENT_DIM (64) → GAME_STATE_DIM (35)  reconstruction head
  Classifier C : LATENT_DIM (64) → 2 (source=CW / target=NaSim domain label)

During training:
  - E + D minimise reconstruction loss (keeps latents task-informative)
  - E + C minimise domain classification loss (GRL flips gradients for E)
  - E learns domain-invariant features that still encode game state
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

from envs.kill_chain import GAME_STATE_DIM, LATENT_DIM, MAX_SLOTS, CTX_DIM
from models.encoder import Encoder


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class DomainClassifier(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, latent: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        rev = grad_reverse(latent, alpha)
        return self.net(rev)


class Decoder(nn.Module):
    """Reconstruct game-state from latent (task preservation)."""

    def __init__(self, latent_dim: int = LATENT_DIM, obs_dim: int = GAME_STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class DAPNAdversarial(nn.Module):
    """Encoder + decoder + domain classifier for adversarial training."""

    def __init__(self,
                 obs_dim:    int = GAME_STATE_DIM,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder    = Encoder(obs_dim, latent_dim)
        self.decoder    = Decoder(latent_dim, obs_dim)
        self.classifier = DomainClassifier(latent_dim)

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        latent = self.encoder(x)
        domain_logits = self.classifier(latent, alpha)
        recon = self.decoder(latent)
        return latent, domain_logits, recon


def save_encoder(model: DAPNAdversarial, path: str,
                 obs_dim: int = GAME_STATE_DIM,
                 latent_dim: int = LATENT_DIM,
                 max_slots: int = MAX_SLOTS,
                 kc_dim: int | None = None,
                 ctx_dim: int = CTX_DIM) -> None:
    if kc_dim is None:
        kc_dim = max_slots * 6
    torch.save({
        "obs_dim":             obs_dim,
        "latent_dim":          latent_dim,
        "max_slots":           max_slots,
        "kc_dim":              kc_dim,
        "ctx_dim":             ctx_dim,
        "policy_dim":          latent_dim + max_slots + ctx_dim,
        "encoder_state_dict":  model.encoder.state_dict(),
    }, path)
