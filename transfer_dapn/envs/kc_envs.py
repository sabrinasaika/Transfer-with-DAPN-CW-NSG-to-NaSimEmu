"""Shared kill-chain env builders (CW / NSG / NaSim transfer pipeline)."""

from __future__ import annotations

import gymnasium as gym


def make_cw_kc_env(scenario: str = "two_subnet") -> gym.Env:
    from envs.cw_wrapper import CWKillChainWrapper
    from envs.context_wrapper import AddContextWrapper
    return AddContextWrapper(CWKillChainWrapper(scenario=scenario))


def make_nasim_kc_env(scenario: str = "two_subnet") -> gym.Env:
    from envs.nasim_wrapper import NaSimKillChainWrapper
    from envs.context_wrapper import AddContextWrapper
    return AddContextWrapper(NaSimKillChainWrapper(scenario=scenario))


def make_nsg_kc_env() -> gym.Env:
    from envs.nsg_wrapper import NSGKillChainWrapper
    from envs.context_wrapper import AddContextWrapper
    from envs.kill_chain_nsg import KC_DIM
    return AddContextWrapper(NSGKillChainWrapper(), kc_dim=KC_DIM)


def make_nasim_nsg_kc_env() -> gym.Env:
    from envs.nasim_nsg_wrapper import NaSimNSGKillChainWrapper
    from envs.context_wrapper import AddContextWrapper
    from envs.kill_chain_nsg import KC_DIM
    return AddContextWrapper(NaSimNSGKillChainWrapper(), kc_dim=KC_DIM)


def make_emu_kc_env() -> gym.Env:
    from nasimemu.env_emu import EmulatedNASimEnv
    from envs.nasim_wrapper import NaSimKillChainWrapper
    from envs.context_wrapper import AddContextWrapper
    from envs.scenario_load import load_fixed_dmz_scenario

    scenario = load_fixed_dmz_scenario()
    emu = EmulatedNASimEnv(
        scenario=scenario, fully_obs=False, flat_actions=True, flat_obs=True)
    return AddContextWrapper(NaSimKillChainWrapper(env=emu))
