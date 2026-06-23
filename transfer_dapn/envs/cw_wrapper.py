"""CyberWheel kill-chain wrapper: 24-D KC obs, Discrete(5) slot actions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyberwheel"))

from envs.cw_native_wrapper import CWNativeWrapper, _make_cw_env
from envs.kill_chain import (
    KC_DIM, MAX_SLOTS, ENTRY_SLOT, build_kc_obs,
    cw_kc_compromised, cw_kc_to_access, inactive_slot_feats, slot_feats_from_state,
)
from envs.scenario_cfg import get_scenario, padded_host_names

_STEP_COST = -1.0
_GOAL_REWARD = 100.0
_NOOP = MAX_SLOTS  # 7 = noop


def _cw_raw_to_kc(raw_obs: np.ndarray, obs_index: dict, host_names) -> np.ndarray:
    """Build 7-slot KC obs from CW red obs_vec."""
    slots = []
    for slot, hname in enumerate(host_names):
        if not hname or hname not in obs_index:
            slots.append(inactive_slot_feats())
            continue
        idx = obs_index[hname]
        feats = raw_obs[idx:idx + 7]
        sweeped, scanned, discovered, on_host, escalated, impacted = feats[1:7]
        access = cw_kc_to_access(on_host, escalated, impacted)
        compromised = cw_kc_compromised(
            sweeped, scanned, discovered, on_host, escalated, impacted)
        slots.append(slot_feats_from_state(
            slot,
            access,
            bool(discovered >= 0.5),
            compromised,
            on_host=float(on_host),
        ))
    return build_kc_obs(slots)


class CWKillChainWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env | None = None, scenario: str = "two_subnet"):
        super().__init__()
        self._cfg = get_scenario(scenario)
        self._host_names = padded_host_names(self._cfg)
        self._target_slot = self._cfg.target_slot
        self._env = env if env is not None else _make_cw_env(scenario)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(KC_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_SLOTS + 1)  # 7 slots + noop

    def reset(self, seed=None, options=None):
        raw, info = self._env.reset(seed=seed, options=options)
        vec = self._denorm(raw)
        return _cw_raw_to_kc(vec, self._obs_index(), self._host_names), info or {}

    def step(self, action: int):
        cw_action = self._translate(int(action))
        raw, _r, _t, _tr, info = self._env.step(
            {"red": cw_action, "blue": 0})
        vec = self._denorm(raw)

        goal = float(info.get("red_reward", 0) if isinstance(info, dict) else 0) >= 100.0
        reward = _STEP_COST + (_GOAL_REWARD if goal else 0.0)
        terminated = bool(goal)
        truncated = bool(_tr) and not terminated

        obs = _cw_raw_to_kc(vec, self._obs_index(), self._host_names)
        return obs, reward, terminated, truncated, {"win": goal, **(info or {})}

    def _obs_index(self) -> dict:
        return getattr(self._env.red_agent.observation, "obs_index", {})

    def _denorm(self, raw) -> np.ndarray:
        vec = raw.get("red") if isinstance(raw, dict) else raw
        return np.asarray(vec, dtype=np.float32).ravel()

    def _translate(self, slot: int) -> int:
        if slot == _NOOP or not self._host_names[slot]:
            slot = ENTRY_SLOT

        aspace = self._env.red_agent.action_space
        hname = self._host_names[slot]
        if hname not in aspace.host_index_map:
            return 0

        base = aspace.host_index_map[hname]
        if base >= aspace._action_space_size:
            return 0

        obs = self._env.red_agent.observation.obs
        if hname not in obs:
            return 0

        h = obs[hname]
        if not h.get("sweeped", 0):
            kc = 0
        elif not h.get("scanned", 0):
            kc = 1
        elif not h.get("discovered", 0):
            kc = 2
        elif not h.get("on_host", 0):
            kc = 3
        elif not h.get("escalated", 0) and slot != self._target_slot:
            kc = 4
        else:
            kc = 5

        action = base + kc
        if action >= aspace._action_space_size:
            return 0
        return action

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
