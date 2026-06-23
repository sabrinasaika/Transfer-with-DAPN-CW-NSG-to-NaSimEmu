"""10-slot kill-chain layout for NSG two_networks ↔ NaSim nasim_two_subnet.

Follows NaSim topology exactly (transfer_dapn/scenarios/nasim_two_subnet.yaml):
  subnet 1 (clients): 5 hosts — entry at (1,0)
  subnet 2 (servers): 5 hosts — target at (2,0)

Slot assignment:
  slot 0  → (1,0) / client_1       ENTRY
  slot 1  → (2,0) / smb_server      TARGET
  slots 2-5 → (2,1)..(2,4) servers
  slots 6-9 → (1,1)..(1,4) clients

Observation dimensions:
  KC_DIM         = 10 × 6 = 60
  KC+ctx         = 62
  GAME_STATE_DIM = 10 × 5 = 50
  POLICY_DIM     = 64 + 10 + 2 = 76
"""

from __future__ import annotations

import numpy as np

from envs.kill_chain import (
    ACCESS_NONE,
    ACCESS_ROOT,
    ACCESS_USER,
    CTX_DIM,
    KC_FEATS,
    LATENT_DIM,
    inactive_slot_feats,
    slot_feats_from_state,
)

__all__ = [
    "MAX_SLOTS", "KC_DIM", "GS_FEATS", "GAME_STATE_DIM", "POLICY_DIM",
    "ENTRY_SLOT", "TARGET_SLOT", "KC_FEATS", "CTX_DIM", "LATENT_DIM",
    "NASIM_NSG_SLOT_ORDER", "NSG_HOST_IPS", "NSG_ENTRY_IP", "NSG_TARGET_IP",
    "NSG_CC_IP", "NSG_CLIENT_NET", "NSG_SERVER_NET",
    "NSG_TARGET_USER_SERVICE", "NSG_TARGET_PRIV_SERVICE", "NSG_TARGET_SERVICE",
    "NSG_GOAL_DATA", "NASIM_NSG_ENTRY", "NASIM_NSG_TARGET",
    "build_kc_obs", "extract_game_state", "extract_is_target",
    "nsg_slot_network", "inactive_slot_feats", "slot_feats_from_state",
]

MAX_SLOTS = 10
KC_DIM = MAX_SLOTS * KC_FEATS          # 60
GS_FEATS = 5
GAME_STATE_DIM = MAX_SLOTS * GS_FEATS  # 50
POLICY_DIM = LATENT_DIM + MAX_SLOTS + CTX_DIM  # 76

ENTRY_SLOT = 0
TARGET_SLOT = 1

NASIM_NSG_SLOT_ORDER = [
    (1, 0),  # 0  entry client
    (2, 0),  # 1  smb_server TARGET
    (2, 1), (2, 2), (2, 3), (2, 4),  # 2-5  servers
    (1, 1), (1, 2), (1, 3), (1, 4),  # 6-9  clients
]

# NSG two_networks fixed IPs (use_dynamic_addresses=False)
NSG_HOST_IPS = [
    "192.168.2.2",   # slot 0  client_1
    "192.168.1.2",   # slot 1  smb_server
    "192.168.1.3",   # slot 2  db_server
    "192.168.1.4",   # slot 3  web_server
    "192.168.1.5",   # slot 4  other_server_1
    "192.168.1.6",   # slot 5  other_server_2
    "192.168.2.3",   # slot 6  client_2
    "192.168.2.4",   # slot 7  client_3
    "192.168.2.5",   # slot 8  client_4
    "192.168.2.6",   # slot 9  client_5
]

NSG_ENTRY_IP = NSG_HOST_IPS[ENTRY_SLOT]
NSG_TARGET_IP = NSG_HOST_IPS[TARGET_SLOT]
NSG_CC_IP = "213.47.23.195"
NSG_CLIENT_NET = "192.168.2.0"
NSG_SERVER_NET = "192.168.1.0"

NSG_TARGET_USER_SERVICE = "microsoft-ds"   # ~ e_ssh  → USER on (2,0)
NSG_TARGET_PRIV_SERVICE = "ms-wbt-server"  # ~ pe_tomcat → ROOT on (2,0)
NSG_TARGET_SERVICE = NSG_TARGET_USER_SERVICE
NSG_GOAL_DATA = ("User1", "DataFromServer1")

NASIM_NSG_ENTRY = NASIM_NSG_SLOT_ORDER[ENTRY_SLOT]
NASIM_NSG_TARGET = NASIM_NSG_SLOT_ORDER[TARGET_SLOT]


def build_kc_obs(slots) -> np.ndarray:
    out = np.zeros(KC_DIM, dtype=np.float32)
    for i, feats in enumerate(slots):
        out[i * KC_FEATS:(i + 1) * KC_FEATS] = feats
    return out


def extract_game_state(kc_obs: np.ndarray) -> np.ndarray:
    kc = np.asarray(kc_obs, dtype=np.float32).reshape(MAX_SLOTS, KC_FEATS)
    return kc[:, :5].reshape(-1)


def extract_is_target(kc_obs: np.ndarray) -> np.ndarray:
    kc = np.asarray(kc_obs, dtype=np.float32).reshape(MAX_SLOTS, KC_FEATS)
    return kc[:, 5].astype(np.float32)


def nsg_slot_network(slot: int) -> tuple[str, int]:
    """Return (network_cidr, prefix) for ScanNetwork on this slot."""
    if slot == ENTRY_SLOT or slot >= 6:
        return NSG_CLIENT_NET, 24
    return NSG_SERVER_NET, 24
