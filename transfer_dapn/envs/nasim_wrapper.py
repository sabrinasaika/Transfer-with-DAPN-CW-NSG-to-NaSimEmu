"""NaSim kill-chain wrapper: 24-D KC obs, Discrete(5) slot actions, NaSim flat actions inside."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nasimemu.nasim.envs import NASimEnv
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from envs.kill_chain import (
    KC_DIM, MAX_SLOTS, SLOT_ORDER, ENTRY_SLOT, ENTRY_ADDR,
    build_kc_obs, inactive_slot_feats, slot_feats_from_state,
)
from envs.host_map import (
    discover_host_layout, flat_action, entry_on_host, ACTIONS_PER_HOST,
)
from envs.scenario_cfg import (
    ScenarioCfg, TWO_SUBNET, get_scenario,
    exploit_local_dict, exploit_service_dict, padded_slot_order,
)
from envs.scenario_load import load_scenario_cfg

_STEP_COST = -1.0
_GOAL_REWARD = 100.0
_STEP_LIMIT = 200
_NOOP = MAX_SLOTS  # slot index = MAX_SLOTS means noop

# NaSim per-host local action indices (ServiceScan, OSScan, SubnetScan, ProcessScan, exploits…)
_SCAN_SERVICE = 0
_SCAN_OS = 1
_SCAN_SUBNET = 2
_SCAN_PROCESS = 3


def _nasim_state_to_kc(state, slot_order) -> np.ndarray:
    """Map NaSim state → KC obs aligned with CWKillChainWrapper semantics."""
    slots = []
    host_map = {addr: host for addr, host in state.hosts}

    for slot, addr in enumerate(slot_order):
        if addr is None:
            slots.append(inactive_slot_feats())
            continue
        host = host_map.get(addr)
        if host is None:
            slots.append(inactive_slot_feats())
            continue

        access = int(host.access)
        discovered = bool(host.discovered)
        on_host = access >= AccessLevel.USER
        slots.append(slot_feats_from_state(
            slot,
            access,
            discovered,
            bool(host.compromised),
            on_host=float(on_host),
        ))
    return build_kc_obs(slots)


def _host_row(state, addr):
    for addr_i, host in state.hosts:
        if addr_i == addr:
            return host
    return None


def _init_host_vector(env, entry_addr) -> None:
    h0 = env.network.hosts[entry_addr]
    HostVector.vectorize(h0, env.scenario.address_space_bounds)


def _merge_emu_matrix(env, matrix, entry_addr) -> None:
    """Apply emulator host-vector rows onto env.current_state (monotonic merge)."""
    _init_host_vector(env, entry_addr)
    mat = np.asarray(matrix)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    for row in mat:
        row = np.asarray(row, dtype=np.float32).ravel()
        if row.size < HostVector.state_size:
            continue
        new_hv = HostVector(row[: HostVector.state_size])
        addr = new_hv.address
        if addr not in env.current_state.host_num_map:
            continue
        old_hv = env.current_state.get_host(addr)
        merged = old_hv.vector.copy()
        ni, oi = new_hv.vector, old_hv.vector
        # Keep best-known access and discovery flags across partial MSF updates.
        merged[HostVector._compromised_idx] = max(oi[HostVector._compromised_idx], ni[HostVector._compromised_idx])
        merged[HostVector._reachable_idx] = max(oi[HostVector._reachable_idx], ni[HostVector._reachable_idx])
        merged[HostVector._discovered_idx] = max(oi[HostVector._discovered_idx], ni[HostVector._discovered_idx])
        merged[HostVector._access_idx] = max(oi[HostVector._access_idx], ni[HostVector._access_idx])
        merged[HostVector._value_idx] = max(oi[HostVector._value_idx], ni[HostVector._value_idx])
        # OR-merge service / OS / process knowledge bits.
        for srv in env.scenario.services:
            if new_hv.is_running_service(srv):
                idx = HostVector._get_service_idx(HostVector.service_idx_map[srv])
                merged[idx] = 1.0
        env.current_state.update_host(addr, HostVector(merged))


class NaSimKillChainWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env=None, scenario: str | ScenarioCfg = "two_subnet"):
        super().__init__()
        self._cfg = get_scenario(scenario) if isinstance(scenario, str) else scenario
        self._slot_order = padded_slot_order(self._cfg)
        self._exploit_local = exploit_local_dict(self._cfg)
        self._exploit_service = exploit_service_dict(self._cfg)
        self._privesc_local = self._cfg.privesc_local
        self._entry_addr = self._cfg.entry_addr
        self._target_slot = self._cfg.target_slot
        self._scenario = load_scenario_cfg(self._cfg)
        self._emulated = env is not None
        if env is None:
            self._env = NASimEnv(
                self._scenario, fully_obs=False, flat_actions=True, flat_obs=True)
        else:
            self._env = env
        self._layout = discover_host_layout(
            self._scenario.address_space, self._slot_order)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(KC_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_SLOTS + 1)  # 7 slots + noop
        self._step_count = 0
        self._last_step_info: dict = {}

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._last_step_info = {}
        if self._emulated:
            matrix = self._env.reset()
            _merge_emu_matrix(self._env, matrix, self._entry_addr)
        else:
            self._env.reset()

        aph = self._cfg.actions_per_host
        if self._emulated:
            eager = (_SCAN_SERVICE, self._exploit_local[self._entry_addr], _SCAN_SUBNET)
        else:
            eager = (self._exploit_local[self._entry_addr], _SCAN_SUBNET)
        entry_idx = self._layout[ENTRY_SLOT]
        for local in eager:
            self._step_flat(flat_action(entry_idx, local, aph))

        obs = _nasim_state_to_kc(self._env.current_state, self._slot_order)
        return obs, {}

    def step(self, action: int):
        action = int(action)
        self._step_flat(self._translate(action))
        self._step_count += 1

        goal = self._env.goal_reached(self._env.current_state)
        reward = _STEP_COST + (_GOAL_REWARD if goal else 0.0)
        terminated = bool(goal)
        truncated = (not terminated) and (self._step_count >= _STEP_LIMIT)

        obs = _nasim_state_to_kc(self._env.current_state, self._slot_order)
        return obs, reward, terminated, truncated, {"win": goal}

    def _step_flat(self, flat: int) -> None:
        if self._emulated:
            action = self._env.action_space.get_action(flat)
            matrix, _, _, info = self._env.step(action)
            _merge_emu_matrix(self._env, matrix, self._entry_addr)
            self._last_step_info = info or {}
        else:
            self._env.step(flat)
            self._last_step_info = {}

    def _needs_service_scan(self, addr) -> bool:
        if not self._emulated:
            return False
        svc = self._exploit_service.get(addr)
        if not svc:
            return False
        hv = self._env.current_state.get_host(addr)
        return not hv.is_running_service(svc)

    def _translate(self, slot: int) -> int:
        aph = self._cfg.actions_per_host
        if slot == _NOOP:
            local = _SCAN_SERVICE if self._emulated else _SCAN_SUBNET
            return flat_action(self._layout[ENTRY_SLOT], local, aph)

        if slot >= len(self._slot_order) or self._slot_order[slot] is None:
            slot = ENTRY_SLOT

        kc = _nasim_state_to_kc(self._env.current_state, self._slot_order)
        if slot > ENTRY_SLOT and not entry_on_host(kc):
            slot = ENTRY_SLOT

        return self._advance_slot(slot)

    def _emu_pre_user_action(self, addr, host_idx) -> int:
        aph = self._cfg.actions_per_host
        if self._needs_service_scan(addr):
            return flat_action(host_idx, _SCAN_SERVICE, aph)
        if addr in self._exploit_local:
            return flat_action(host_idx, self._exploit_local[addr], aph)
        return flat_action(host_idx, _SCAN_SERVICE, aph)

    def _advance_slot(self, slot: int) -> int:
        aph = self._cfg.actions_per_host
        addr = self._slot_order[slot]
        entry_idx = self._layout[ENTRY_SLOT]
        if addr is None or slot not in self._layout:
            local = _SCAN_SERVICE if self._emulated else _SCAN_SUBNET
            return flat_action(entry_idx, local, aph)

        host_idx = self._layout[slot]
        row = _host_row(self._env.current_state, addr)
        if row is None:
            local = _SCAN_SERVICE if self._emulated else _SCAN_SUBNET
            return flat_action(entry_idx, local, aph)

        access = int(row.access)
        entry = _host_row(self._env.current_state, self._entry_addr)
        entry_access = int(entry.access) if entry is not None else 0

        if self._emulated and access < AccessLevel.USER:
            if slot != ENTRY_SLOT and entry_access < AccessLevel.USER:
                return self._emu_pre_user_action(self._entry_addr, entry_idx)
            if not row.discovered:
                return flat_action(host_idx, _SCAN_SERVICE, aph)
            return self._emu_pre_user_action(addr, host_idx)

        if access >= AccessLevel.ROOT:
            return flat_action(host_idx, _SCAN_SUBNET, aph)

        if access >= AccessLevel.USER:
            if slot == self._target_slot:
                return flat_action(host_idx, self._privesc_local, aph)
            if slot != 3:
                return flat_action(host_idx, self._privesc_local, aph)
            return flat_action(host_idx, _SCAN_SUBNET, aph)

        if not row.discovered:
            if slot == ENTRY_SLOT:
                return flat_action(host_idx, _SCAN_SUBNET, aph)
            if entry is not None and entry_access >= AccessLevel.USER:
                return flat_action(entry_idx, _SCAN_SUBNET, aph)
            return flat_action(host_idx, _SCAN_SERVICE, aph)

        if self._needs_service_scan(addr):
            return flat_action(host_idx, _SCAN_SERVICE, aph)

        if addr in self._exploit_local:
            return flat_action(host_idx, self._exploit_local[addr], aph)
        return flat_action(host_idx, _SCAN_SERVICE, aph)

    def close(self):
        pass
