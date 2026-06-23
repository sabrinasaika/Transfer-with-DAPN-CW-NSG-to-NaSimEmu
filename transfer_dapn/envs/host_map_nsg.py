"""Host layout and flat-action indices for NSG-aligned NaSim scenario.

Scenario: transfer_dapn/scenarios/nasim_two_subnet.yaml
  subnets: [1, 5, 5]  — internet:1, clients:5, servers:5

Exploit ordering in YAML:
  index 0: e_http   -> local 4
  index 1: e_ssh    -> local 5
  privesc pe_tomcat -> local 6

ACTIONS_PER_HOST = 4 scans + 2 exploits + 1 privesc = 7
"""

from __future__ import annotations

from envs.kill_chain import KC_FEATS
from envs.kill_chain_nsg import ENTRY_SLOT, NASIM_NSG_ENTRY, NASIM_NSG_SLOT_ORDER

ACTIONS_PER_HOST = 7

EXPLOIT_LOCAL = {
    (1, 0): 4,   # e_http   (entry client)
    (2, 0): 5,   # e_ssh    (smb_server TARGET)
}

EXPLOIT_SERVICE = {
    (1, 0): "http",
    (2, 0): "ssh",
}

PRIVESC_LOCAL = 6

_EXPLOIT_NAME = {
    ((1, 0), 4): "e_http",
    ((2, 0), 5): "e_ssh",
}


def discover_host_layout(address_space):
    layout = {}
    for slot, addr in enumerate(NASIM_NSG_SLOT_ORDER):
        if addr in address_space:
            layout[slot] = address_space.index(addr)
    return layout


def flat_action(slot_host_idx: int, local_offset: int) -> int:
    return slot_host_idx * ACTIONS_PER_HOST + local_offset


def flat_to_atype(flat: int) -> int:
    """Map NSG NaSim flat action → canonical action type (5 classes).

    Layout per host (7 locals): 0=svc scan, 1=os scan, 2=subnet scan,
    3=proc scan, 4=e_http, 5=e_ssh, 6=privesc.
    """
    local = flat % ACTIONS_PER_HOST
    if local == 2:
        return 0  # ScanNetwork
    if local == 0:
        return 1  # FindServices
    if local in (4, 5):
        return 2  # ExploitService
    if local in (1, 3):
        return 3  # FindData
    if local == 6:
        return 4  # ExfiltrateData (privesc completes goal)
    return 1


def entry_on_host(kc_obs) -> bool:
    return float(kc_obs[ENTRY_SLOT * KC_FEATS + 2]) > 0.5
