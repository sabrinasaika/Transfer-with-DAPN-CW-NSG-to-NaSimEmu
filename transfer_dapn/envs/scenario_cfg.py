"""Scenario registry for DAPN transfer experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
_DAPN = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ScenarioCfg:
    name: str
    nasim_yaml: Path
    cw_env_yaml: str
    slot_order: tuple
    host_names: tuple
    entry_addr: tuple
    target_addr: tuple
    exploit_local: tuple
    exploit_service: tuple
    entry_slot: int = 0
    target_slot: int = 1
    max_slots: int = 7
    actions_per_host: int = 10
    privesc_local: int = 9
    eager_atypes: tuple = (2, 0)  # ExploitService, ScanNetwork (hidden reset)
    cw_max_hosts: int = 7
    data_kc_obs: str = "data/kc_obs.npz"
    encoder_out: str = "artifacts/models/dapn_encoder_kc7.pt"
    p1_policy_dir: str = "artifacts/policies/cw_dapn_policy"
    p1_policy_final: str = "artifacts/policies/cw_dapn_policy_final.zip"
    pe_policy_dir: str = "artifacts/policies/nasim_kc_invariant"
    pe_policy_final: str = "artifacts/policies/nasim_kc_invariant_final.zip"
    traj_out: str = "artifacts/results/traj_similarity.png"


def _exploit_map(pairs):
    return dict(pairs)


TWO_SUBNET = ScenarioCfg(
    name="two_subnet",
    nasim_yaml=_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml",
    cw_env_yaml="fixed_dmz_two_subnet_transfer.yaml",
    slot_order=(
        (1, 0), (3, 0), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2),
    ),
    host_names=(
        "dmz_entry", "service_target", "user_server", "user_worker",
        "user_extra", "service_server_1", "service_server_2",
    ),
    entry_addr=(1, 0),
    target_addr=(3, 0),
    target_slot=1,
    exploit_local=(
        ((1, 0), 4), ((2, 0), 6), ((2, 1), 8), ((2, 2), 5),
        ((3, 0), 5), ((3, 1), 6), ((3, 2), 7),
    ),
    exploit_service=(
        ((1, 0), "21_linux_proftpd"),
        ((2, 0), "80_linux_phpwiki"),
        ((2, 1), "80_windows_wp_ninja"),
        ((2, 2), "80_linux_drupal"),
        ((3, 0), "80_linux_drupal"),
        ((3, 1), "80_linux_phpwiki"),
        ((3, 2), "9200_windows_elasticsearch"),
    ),
    cw_max_hosts=7,
)

ONE_SUBNET = ScenarioCfg(
    name="one_subnet",
    nasim_yaml=_REPO / "scenarios" / "fixed_dmz_one_subnet_4host.v2.yaml",
    cw_env_yaml="fixed_dmz_one_subnet_transfer.yaml",
    slot_order=((1, 0), (2, 0), (2, 1), (2, 2)),
    host_names=("dmz_entry", "target_drupal", "phpwiki_server", "windows_server"),
    entry_addr=(1, 0),
    target_addr=(2, 0),
    target_slot=1,
    exploit_local=(
        ((1, 0), 4), ((2, 0), 5), ((2, 1), 6), ((2, 2), 8),
    ),
    exploit_service=(
        ((1, 0), "21_linux_proftpd"),
        ((2, 0), "80_linux_drupal"),
        ((2, 1), "80_linux_phpwiki"),
        ((2, 2), "80_windows_wp_ninja"),
    ),
    cw_max_hosts=4,
    data_kc_obs="data/kc_obs_one_subnet.npz",
    encoder_out="artifacts/models/dapn_encoder_one_subnet.pt",
    p1_policy_dir="artifacts/policies/cw_dapn_one_subnet",
    p1_policy_final="artifacts/policies/cw_dapn_one_subnet_final.zip",
    pe_policy_dir="artifacts/policies/nasim_kc_one_subnet",
    pe_policy_final="artifacts/policies/nasim_kc_one_subnet_final.zip",
    traj_out="artifacts/results/traj_similarity_one_subnet.png",
)

SCENARIOS = {
    "two_subnet": TWO_SUBNET,
    "one_subnet": ONE_SUBNET,
}


def get_scenario(name: str = "two_subnet") -> ScenarioCfg:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}; choose from {list(SCENARIOS)}")
    return SCENARIOS[name]


def padded_host_names(cfg: ScenarioCfg) -> tuple:
    names = list(cfg.host_names) + [""] * cfg.max_slots
    return tuple(names[: cfg.max_slots])


def padded_slot_order(cfg: ScenarioCfg) -> tuple:
    addrs = list(cfg.slot_order) + [None] * cfg.max_slots
    return tuple(addrs[: cfg.max_slots])


def exploit_local_dict(cfg: ScenarioCfg) -> dict:
    return _exploit_map(cfg.exploit_local)


def exploit_service_dict(cfg: ScenarioCfg) -> dict:
    return _exploit_map(cfg.exploit_service)
