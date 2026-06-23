"""NaSim kill-chain wrapper for NSG-aligned nasim_two_subnet scenario."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nasimemu.nasim.envs import NASimEnv
from nasimemu.nasim.envs.utils import AccessLevel

from envs.kill_chain import inactive_slot_feats, slot_feats_from_state
from envs.kill_chain_nsg import (
    KC_DIM,
    MAX_SLOTS,
    NASIM_NSG_ENTRY,
    NASIM_NSG_SLOT_ORDER,
    ENTRY_SLOT,
    TARGET_SLOT,
    build_kc_obs,
)
from envs.host_map_nsg import (
    discover_host_layout, flat_action, EXPLOIT_LOCAL, EXPLOIT_SERVICE,
    PRIVESC_LOCAL, entry_on_host,
)
from envs.scenario_load import load_nsg_nasim_scenario

_STEP_COST = -1.0
_GOAL_REWARD = 100.0
_STEP_LIMIT = 200
_NOOP = MAX_SLOTS

_SCAN_SERVICE = 0
_SCAN_SUBNET = 2
_ENTRY_ADDR = NASIM_NSG_ENTRY


def _nasim_state_to_kc(state) -> np.ndarray:
    slots = []
    host_map = {addr: host for addr, host in state.hosts}
    for slot, addr in enumerate(NASIM_NSG_SLOT_ORDER):
        host = host_map.get(addr)
        if host is None:
            slots.append(inactive_slot_feats())
            continue
        access = int(host.access)
        slots.append(slot_feats_from_state(
            slot, access, bool(host.discovered), bool(host.compromised),
            on_host=float(access >= AccessLevel.USER),
        ))
    return build_kc_obs(slots)


def _host_row(state, addr):
    for addr_i, host in state.hosts:
        if addr_i == addr:
            return host
    return None


class NaSimNSGKillChainWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env=None):
        super().__init__()
        self._scenario = load_nsg_nasim_scenario()
        if env is None:
            self._env = NASimEnv(
                self._scenario, fully_obs=False, flat_actions=True, flat_obs=True)
        else:
            self._env = env
        self._layout = discover_host_layout(self._scenario.address_space)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(KC_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_SLOTS + 1)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._env.reset()
        # Eager entry: exploit http on client entry, then subnet scan.
        entry_idx = self._layout[ENTRY_SLOT]
        for local in (EXPLOIT_LOCAL[_ENTRY_ADDR], _SCAN_SUBNET):
            self._env.step(flat_action(entry_idx, local))
        return _nasim_state_to_kc(self._env.current_state), {}

    def step(self, action: int):
        self._env.step(self._translate(int(action)))
        self._step_count += 1
        goal = self._env.goal_reached(self._env.current_state)
        reward = _STEP_COST + (_GOAL_REWARD if goal else 0.0)
        terminated = bool(goal)
        truncated = (not terminated) and (self._step_count >= _STEP_LIMIT)
        obs = _nasim_state_to_kc(self._env.current_state)
        return obs, reward, terminated, truncated, {"win": goal}

    def _translate(self, slot: int) -> int:
        if slot == _NOOP:
            return flat_action(self._layout[ENTRY_SLOT], _SCAN_SUBNET)
        kc = _nasim_state_to_kc(self._env.current_state)
        if slot > ENTRY_SLOT and not entry_on_host(kc):
            slot = ENTRY_SLOT
        return self._advance_slot(slot)

    def _advance_slot(self, slot: int) -> int:
        addr = NASIM_NSG_SLOT_ORDER[slot]
        host_idx = self._layout[slot]
        entry_idx = self._layout[ENTRY_SLOT]
        row = _host_row(self._env.current_state, addr)
        if row is None:
            return flat_action(entry_idx, _SCAN_SUBNET)

        access = int(row.access)
        entry = _host_row(self._env.current_state, _ENTRY_ADDR)
        entry_access = int(entry.access) if entry is not None else 0

        if access >= AccessLevel.ROOT:
            return flat_action(host_idx, _SCAN_SUBNET)

        if access >= AccessLevel.USER:
            if slot == TARGET_SLOT:
                return flat_action(host_idx, PRIVESC_LOCAL)
            return flat_action(host_idx, _SCAN_SUBNET)

        if not row.discovered:
            if slot == ENTRY_SLOT:
                return flat_action(host_idx, _SCAN_SUBNET)
            if entry_access >= AccessLevel.USER:
                return flat_action(entry_idx, _SCAN_SUBNET)
            return flat_action(host_idx, 0)

        svc = EXPLOIT_SERVICE.get(addr)
        if svc:
            hv = self._env.current_state.get_host(addr)
            if not hv.is_running_service(svc):
                return flat_action(host_idx, _SCAN_SERVICE)

        if addr in EXPLOIT_LOCAL:
            return flat_action(host_idx, EXPLOIT_LOCAL[addr])
        return flat_action(host_idx, _SCAN_SERVICE)

    def close(self):
        pass
