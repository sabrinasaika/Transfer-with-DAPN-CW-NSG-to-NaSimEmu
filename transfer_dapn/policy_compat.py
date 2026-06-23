"""Load SB3 PPO policy weights across conda envs (numpy 1.x ↔ 2.x).

Training in the `nsg` env pickles metadata with numpy 2.x, which breaks
PPO.load() in the `cyberwheel` env.  policy.pth is portable — load it here.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import torch
from stable_baselines3 import PPO


def load_ppo_policy_weights(
    zip_path: str | Path,
    env,
    *,
    net_arch: list[int] | None = None,
    learning_rate: float = 3e-4,
) -> PPO:
    """Build a fresh PPO on ``env`` and load ``policy.pth`` from an SB3 zip."""
    arch = net_arch or [128, 128]
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=arch),
    )
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("policy.pth") as f:
            state = torch.load(f, map_location="cpu", weights_only=False)
    model.policy.load_state_dict(state)
    model.policy.eval()
    return model
