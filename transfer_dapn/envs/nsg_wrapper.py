"""NetSecGame kill-chain wrapper: 60-D KC obs, Discrete(11) slot actions."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from netsecgame import Action, ActionType, IP, Network, Service, generate_valid_actions

from envs.kill_chain import inactive_slot_feats, slot_feats_from_state
from envs.kill_chain_nsg import (
    KC_DIM,
    MAX_SLOTS,
    ENTRY_SLOT,
    TARGET_SLOT,
    NSG_HOST_IPS,
    NSG_SERVER_NET,
    NSG_TARGET_IP,
    NSG_TARGET_PRIV_SERVICE,
    NSG_TARGET_USER_SERVICE,
    build_kc_obs,
    nsg_slot_network,
)
from envs.nsg_client import NSGClient, nasim_goal_reached
from envs.nsg_scenario_patch import apply_nsg_scenario_patches

apply_nsg_scenario_patches()

_STEP_COST = -1.0
_GOAL_REWARD = 100.0
_STEP_LIMIT = 200
_NOOP = MAX_SLOTS


def _ip_known(state, ip_str: str) -> bool:
    return IP(ip_str) in state.known_hosts


def _ip_controlled(state, ip_str: str) -> bool:
    return IP(ip_str) in state.controlled_hosts


def _services_known(state, ip_str: str) -> bool:
    ip = IP(ip_str)
    return ip in state.known_services and bool(state.known_services[ip])


def _nsg_state_to_kc(state, target_stage: int):
    from envs.kill_chain import ACCESS_NONE, ACCESS_ROOT, ACCESS_USER

    slots = []
    for slot, ip_str in enumerate(NSG_HOST_IPS):
        if not ip_str:
            slots.append(inactive_slot_feats())
            continue
        discovered = _ip_known(state, ip_str)
        controlled = _ip_controlled(state, ip_str)
        access = ACCESS_NONE
        if slot == TARGET_SLOT:
            if target_stage >= 2:
                access = ACCESS_ROOT
            elif target_stage >= 1 or controlled:
                access = ACCESS_USER
        elif controlled:
            access = ACCESS_USER
        slots.append(
            slot_feats_from_state(
                slot,
                access,
                discovered,
                controlled or access >= ACCESS_USER,
                on_host=float(access >= ACCESS_USER),
            )
        )
    return build_kc_obs(slots)


class NSGKillChainWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, client: NSGClient | None = None, seed: int = 0):
        super().__init__()
        self._client = client if client is not None else NSGClient(seed=seed)
        self._owns_client = client is None
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(KC_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_SLOTS + 1)
        self._last_obs = None
        self._target_stage = 0
        self._step_count = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._client.seed = int(seed)
        self._target_stage = 0
        self._step_count = 0
        obs = self._client.reset()
        # Eager pivot from entry — mirrors NaSim reset subnet scan after entry access.
        pivot = self._entry_pivot_scan(obs.state)
        if pivot is not None:
            obs = self._client.step(pivot)
        self._last_obs = obs
        return _nsg_state_to_kc(obs.state, self._target_stage), {}

    def step(self, action: int):
        assert self._last_obs is not None
        prev = self._last_obs.state
        nsg_action = self._translate(int(action), prev)
        obs = self._client.step(nsg_action)
        self._last_obs = obs
        self._step_count += 1
        self._update_target_stage(prev, obs.state, nsg_action)

        win = nasim_goal_reached(self._target_stage)
        reward = _STEP_COST + (_GOAL_REWARD if win else 0.0)
        terminated = bool(win)
        truncated = (not terminated) and (
            self._step_count >= _STEP_LIMIT or (bool(obs.end) and not win)
        )

        return (
            _nsg_state_to_kc(obs.state, self._target_stage),
            reward,
            terminated,
            truncated,
            {"win": win, "target_stage": self._target_stage, "nsg_reward": obs.reward},
        )

    def _update_target_stage(self, prev, state, action: Action) -> None:
        target = IP(NSG_TARGET_IP)
        if action.action_type == ActionType.ExploitService:
            svc = action.parameters.get("target_service")
            host = action.parameters.get("target_host")
            if host == target and svc is not None:
                if svc.name == NSG_TARGET_USER_SERVICE and target in state.controlled_hosts:
                    self._target_stage = max(self._target_stage, 1)
                elif (
                    svc.name == NSG_TARGET_PRIV_SERVICE
                    and self._target_stage >= 1
                    and target in state.controlled_hosts
                ):
                    self._target_stage = 2
        elif (
            self._target_stage >= 1
            and action.action_type == ActionType.FindData
            and action.parameters.get("target_host") == target
        ):
            prev_data = prev.known_data.get(target, set())
            new_data = state.known_data.get(target, set())
            if len(new_data) > len(prev_data):
                self._target_stage = 2

    def _entry_pivot_scan(self, state) -> Action | None:
        source = self._default_source(state)
        net = Network(NSG_SERVER_NET, 24)
        act = Action(ActionType.ScanNetwork, parameters={
            "source_host": source, "target_network": net})
        if act in set(generate_valid_actions(state)):
            return act
        return None

    def _service_on_host(self, state, host_ip: IP, name: str) -> Service | None:
        for svc in state.known_services.get(host_ip, set()):
            if svc.name == name:
                return svc
        return None

    def _user_exploit(self, state, source, target_ip: IP) -> Action:
        svc = self._service_on_host(state, target_ip, NSG_TARGET_USER_SERVICE)
        if svc is None:
            svc = Service(NSG_TARGET_USER_SERVICE)
        return Action(ActionType.ExploitService, parameters={
            "source_host": source,
            "target_host": target_ip,
            "target_service": svc,
        })

    def _privesc_candidates(self, state, target_ip: IP) -> list[Action]:
        priv = self._service_on_host(state, target_ip, NSG_TARGET_PRIV_SERVICE)
        cands = [
            Action(ActionType.FindData, parameters={
                "source_host": target_ip, "target_host": target_ip}),
        ]
        if priv is not None:
            cands.append(Action(ActionType.ExploitService, parameters={
                "source_host": target_ip,
                "target_host": target_ip,
                "target_service": priv,
            }))
        return cands

    def _default_source(self, state):
        entry = IP(NSG_HOST_IPS[ENTRY_SLOT])
        if entry in state.controlled_hosts:
            return entry
        controlled = sorted(state.controlled_hosts, key=lambda ip: ip.ip)
        return controlled[0] if controlled else entry

    def _pick_action(self, state, candidates: list[Action]) -> Action:
        valid = set(generate_valid_actions(state))
        for act in candidates:
            if act in valid:
                return act
        for act in candidates:
            if act.action_type == ActionType.ExploitService:
                th = act.parameters.get("target_host")
                ts = act.parameters.get("target_service")
                if ts is None:
                    continue
                for v in valid:
                    if (v.action_type == ActionType.ExploitService
                            and v.parameters.get("target_host") == th
                            and v.parameters.get("target_service") is not None
                            and v.parameters.get("target_service").name == ts.name):
                        return v
        # Never fall back to unrelated exploits — idle scan/discover instead.
        source = self._default_source(state)
        for act in valid:
            if act.action_type == ActionType.FindServices:
                return act
        for act in valid:
            if act.action_type == ActionType.ScanNetwork:
                return act
        return Action(ActionType.FindServices, parameters={
            "source_host": source,
            "target_host": IP(NSG_HOST_IPS[ENTRY_SLOT]),
        })

    def _translate(self, slot: int, state) -> Action:
        if slot == _NOOP:
            slot = ENTRY_SLOT
        if slot >= len(NSG_HOST_IPS) or not NSG_HOST_IPS[slot]:
            slot = ENTRY_SLOT

        ip_str = NSG_HOST_IPS[slot]
        target_ip = IP(ip_str)
        source = self._default_source(state)

        if not _ip_known(state, ip_str):
            net_cidr, prefix = nsg_slot_network(slot)
            return self._pick_action(state, [
                Action(ActionType.ScanNetwork, parameters={
                    "source_host": source, "target_network": Network(net_cidr, prefix)}),
            ])

        if not _services_known(state, ip_str):
            return self._pick_action(state, [
                Action(ActionType.FindServices, parameters={
                    "source_host": source, "target_host": target_ip}),
            ])

        # Slots 2–9: scan/discover only (no exploits), matching NaSim host_map_nsg.
        if slot not in (ENTRY_SLOT, TARGET_SLOT):
            if _ip_controlled(state, ip_str):
                return self._pick_action(state, [
                    Action(ActionType.FindServices, parameters={
                        "source_host": source, "target_host": target_ip}),
                ])
            return self._pick_action(state, [
                Action(ActionType.FindServices, parameters={
                    "source_host": source, "target_host": target_ip}),
            ])

        if slot == TARGET_SLOT:
            if not _ip_controlled(state, NSG_TARGET_IP):
                return self._pick_action(state, [self._user_exploit(state, source, target_ip)])
            if self._target_stage < 1:
                return self._pick_action(state, [self._user_exploit(state, source, target_ip)])
            if self._target_stage < 2:
                return self._pick_action(state, self._privesc_candidates(state, target_ip))

        # Entry slot with access: pivot subnet scan toward servers (NaSim SubnetScan).
        if slot == ENTRY_SLOT and _ip_controlled(state, ip_str):
            pivot = self._entry_pivot_scan(state)
            if pivot is not None:
                return pivot

        return self._pick_action(state, [
            Action(ActionType.FindServices, parameters={
                "source_host": source, "target_host": target_ip}),
        ])

    def close(self):
        if self._owns_client:
            self._client.close()
