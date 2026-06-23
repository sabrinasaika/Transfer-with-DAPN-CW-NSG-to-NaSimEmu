"""7-slot kill-chain observation layout.

Slots cover all hosts in fixed_dmz_two_subnet (NaSim) / fixed-dmz-two-subnet (CW).
Both domains expose the same 7 kill-chain slots.

Per-slot features (6-D):
  [phase_norm, reachable, on_host, is_entry, active, is_target]

Game-state features (5-D per slot, is_target excluded):
  [phase_norm, reachable, on_host, is_entry, active]

Observation dimensions:
  KC_DIM        = 7 × 6  = 42   (raw kill-chain)
  KC+ctx        = 42 + 2 = 44   (what wrappers produce)
  GAME_STATE_DIM= 7 × 5  = 35   (encoder input — is_target stripped)
  LATENT_DIM    =           64   (encoder output)
  POLICY_DIM    = 64+7+2 = 73   (latent + is_target + ctx — policy input)
"""

from __future__ import annotations

import numpy as np

MAX_SLOTS = 7
KC_FEATS  = 6
KC_DIM    = MAX_SLOTS * KC_FEATS         # 42
CTX_DIM   = 2

GS_FEATS       = 5                       # per-slot game-state (no is_target)
GAME_STATE_DIM = MAX_SLOTS * GS_FEATS    # 35
LATENT_DIM     = 64
POLICY_DIM     = LATENT_DIM + MAX_SLOTS + CTX_DIM  # 73

# NaSim slot assignment (fixed_dmz_two_subnet.v2.yaml):
#   slot 0 → (1,0) dmz_entry        ENTRY  (DMZ subnet)
#   slot 1 → (3,0) service_target   TARGET (service subnet, sensitive)
#   slot 2 → (2,0) user_server      PIVOT  (user subnet)
#   slot 3 → (2,1) user_worker      BRANCH (user subnet, Windows)
#   slot 4 → (2,2) user_extra       (user subnet, drupal)
#   slot 5 → (3,1) service_extra_1  (service subnet, phpwiki)
#   slot 6 → (3,2) service_extra_2  (service subnet, Windows elasticsearch)
SLOT_ORDER  = [(1, 0), (3, 0), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2)]
ENTRY_SLOT  = 0
TARGET_SLOT = 1
ENTRY_ADDR  = (1, 0)
TARGET_ADDR = (3, 0)

# CW host names for each slot (empty string = no CW host → always zero-padded)
HOST_NAMES = [
    "dmz_entry",       # slot 0  (1,0)
    "service_target",  # slot 1  (3,0)  TARGET
    "user_server",     # slot 2  (2,0)
    "user_worker",     # slot 3  (2,1)
    "user_extra",      # slot 4  (2,2)
    "service_server_1",# slot 5  (3,1)
    "service_server_2",# slot 6  (3,2)
]

# Match nasimemu.nasim.envs.utils.AccessLevel
ACCESS_NONE = 0
ACCESS_USER = 1
ACCESS_ROOT = 2


def phase_norm(access: int, discovered: bool, compromised: bool) -> float:
    """Canonical kill-chain phase in [0, 1] — shared by CW and NaSim wrappers."""
    if not discovered and access < ACCESS_USER:
        return 0.0
    if access >= ACCESS_ROOT:
        return 1.0
    if access >= ACCESS_USER:
        return 0.6
    if compromised:
        return 0.4
    return 0.2


def cw_kc_to_access(on_host: float, escalated: float, impacted: float) -> int:
    """Map CyberWheel KC bits to NaSim-style access level."""
    if impacted >= 0.5:
        return ACCESS_ROOT
    if escalated >= 0.5 or on_host >= 0.5:
        return ACCESS_USER
    return ACCESS_NONE


def cw_kc_compromised(
    sweeped: float, scanned: float, discovered: float,
    on_host: float, escalated: float, impacted: float,
) -> bool:
    """Match NaSim host.compromised — true only after access is gained."""
    return on_host >= 0.5 or escalated >= 0.5 or impacted >= 0.5


def slot_feats_from_state(
    slot: int,
    access: int,
    discovered: bool,
    compromised: bool,
    on_host: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Build one 6-D KC slot vector from canonical state fields."""
    if on_host is None:
        on_host = 1.0 if access >= ACCESS_USER else 0.0
    phase = phase_norm(access, discovered, compromised)
    reachable = 1.0 if (discovered or slot == ENTRY_SLOT) else 0.0
    active = 1.0 if (discovered or slot == ENTRY_SLOT) else 0.0
    return (
        phase,
        reachable,
        float(on_host),
        float(slot == ENTRY_SLOT),
        active,
        float(slot == TARGET_SLOT),
    )


def inactive_slot_feats() -> tuple[float, float, float, float, float, float]:
    """Zero vector for CW padding slots 4–6 and other inactive hosts."""
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def build_kc_obs(slots) -> np.ndarray:
    """slots: iterable of 6-tuples → 42-D KC obs."""
    out = np.zeros(KC_DIM, dtype=np.float32)
    for i, feats in enumerate(slots):
        out[i * KC_FEATS:(i + 1) * KC_FEATS] = feats
    return out


def extract_game_state(kc_obs: np.ndarray) -> np.ndarray:
    """42-D KC obs → 35-D game-state (drop is_target column)."""
    kc = np.asarray(kc_obs, dtype=np.float32).reshape(MAX_SLOTS, KC_FEATS)
    return kc[:, :5].reshape(-1)


def extract_is_target(kc_obs: np.ndarray) -> np.ndarray:
    """42-D KC obs → 7-D is_target vector."""
    kc = np.asarray(kc_obs, dtype=np.float32).reshape(MAX_SLOTS, KC_FEATS)
    return kc[:, 5].astype(np.float32)
