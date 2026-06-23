"""NaSim KC+ctx → encoder wrapper for pe (NaSim-native invariant policy).

Without encoder: 44-D KC+ctx obs, Discrete(8) actions.
With encoder:    73-D encoded obs, Discrete(8) actions — matches p1's input space.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from envs.nasim_wrapper import NaSimKillChainWrapper
from envs.context_wrapper import AddContextWrapper
from envs.dapn_encoder_wrapper import DAPNEncoderWrapper


class NaSimNativeWrapper(gym.Wrapper):
    """NaSim env with KC+ctx obs (44-D) or DAPN-encoded obs (73-D)."""

    def __init__(self, encoder_path: Optional[str] = None, scenario: str = "two_subnet"):
        base = AddContextWrapper(NaSimKillChainWrapper(scenario=scenario))
        if encoder_path:
            base = DAPNEncoderWrapper(base, encoder_path)
        super().__init__(base)
