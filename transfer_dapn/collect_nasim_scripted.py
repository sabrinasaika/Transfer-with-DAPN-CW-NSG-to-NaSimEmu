"""Collect diverse NaSim KC+ctx observations via a scripted kill-chain policy.

Sequences the correct NASimEmuEnv actions in order:
  ServiceScan(1,0) → Exploit e_proftpd(1,0) → SubnetScan(1,0)
  → [optional lateral scans] → ServiceScan(3,0) → Exploit e_drupal(3,0)
  → PrivEsc(3,0)  [WIN]

Collects one KC+ctx obs after each action, giving all 5 kill-chain phases
(0.0 / 0.2 / 0.6 / 1.0) across all 7 KC slots.

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \\
      transfer_dapn/collect_nasim_scripted.py --samples 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO  = Path(__file__).resolve().parents[1]
_DAGTS = Path("/home/ssaika@cs.utep.edu/NASimEmu-agents")

sys.path.insert(0, str(_DAGTS))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "transfer_dapn"))

from config import config as pe_config
pe_config.device       = "cpu"
pe_config.opt_lr       = 1e-3
pe_config.opt_l2       = 1e-4
pe_config.opt_max_norm = 3.0
pe_config.alpha_h      = 0.3
pe_config.emb_dim      = 64
pe_config.node_dim     = 31
pe_config.action_dim   = 10
pe_config.pos_enc_dim  = 8

from envs.scenario_load import patch_nasim_load_scenario
patch_nasim_load_scenario()

from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from envs.kill_chain import (
    KC_DIM, KC_FEATS, MAX_SLOTS, SLOT_ORDER,
    ENTRY_SLOT, TARGET_SLOT, CTX_DIM,
    inactive_slot_feats, slot_feats_from_state, build_kc_obs,
)

_SCENARIO    = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_DEFAULT_OUT = str(_REPO / "transfer_dapn/data/nasim_kc_obs_scripted.npy")

ENTRY_ADDR  = (1, 0)
TARGET_ADDR = (3, 0)
LATERAL     = [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)]

# Fixed action IDs for this scenario (verified from action_list printout)
_SVC_SCAN  = 0   # ServiceScan
_SUB_SCAN  = 2   # SubnetScan
_E_PROFTPD = 7   # Exploit e_proftpd  — used on ENTRY (1,0)
_E_DRUPAL  = 4   # Exploit e_drupal   — used on TARGET (3,0)
_PRIVESC   = 9   # PrivilegeEscalation pe_kernel


def matrix_to_kc(s: np.ndarray,
                 last_action: float = 0.0,
                 last_reward: float = 0.0) -> np.ndarray:
    """NASimEmuEnv matrix obs → 44-D KC+ctx."""
    host_map: dict = {}
    for row in s[:-1]:
        hv   = HostVector(row)
        addr = tuple(int(x) for x in hv.address)
        if addr == (0, 0):
            continue
        host_map[addr] = hv

    slots = []
    for slot, addr in enumerate(SLOT_ORDER):
        hv = host_map.get(addr)
        if hv is None:
            slots.append(inactive_slot_feats())
        else:
            access      = int(hv.access)
            discovered  = float(hv.discovered) > 0
            on_host     = access >= AccessLevel.USER
            compromised = float(hv.compromised) > 0
            slots.append(slot_feats_from_state(
                slot, access, discovered, compromised,
                on_host=float(on_host),
            ))

    kc  = build_kc_obs(slots)
    ctx = np.array([
        last_action / max(1.0, float(MAX_SLOTS)),
        float(np.clip(last_reward / 100.0, -1.0, 1.0)),
    ], dtype=np.float32)
    return np.concatenate([kc, ctx]).astype(np.float32)


def scripted_episode(env: NASimEmuEnv, rng: np.random.Generator) -> list[np.ndarray]:
    """Run one scripted kill-chain episode; return KC+ctx obs list."""
    env._generate_env()
    s = env.reset()

    obs  = []
    last_act, last_rew = 0.0, 0.0

    def record():
        obs.append(matrix_to_kc(s, last_act, last_rew))

    def step(addr, act_id) -> bool:
        nonlocal s, last_act, last_rew
        record()
        s, r, done, _ = env.step((addr, act_id))
        last_act = float(act_id) / max(1.0, float(MAX_SLOTS))
        last_rew = float(r)
        return bool(done)

    # ── Phase 1: compromise entry ──────────────────────────────────────────
    step(ENTRY_ADDR, _SVC_SCAN)       # discover proftpd on (1,0)
    step(ENTRY_ADDR, _E_PROFTPD)      # exploit → USER on entry

    # ── Phase 2: discover the network from entry ───────────────────────────
    step(ENTRY_ADDR, _SUB_SCAN)       # subnet scan → reveals all adjacent hosts

    # ── Phase 3: random lateral scans (adds slot diversity) ───────────────
    scan_order = rng.permutation(len(LATERAL))
    n_scans    = int(rng.integers(1, len(LATERAL) + 1))
    for i in scan_order[:n_scans]:
        done = step(LATERAL[i], _SVC_SCAN)
        if done:
            record()
            return obs

    # ── Phase 4: exploit target ────────────────────────────────────────────
    step(TARGET_ADDR, _SVC_SCAN)      # service scan on target
    done = step(TARGET_ADDR, _E_DRUPAL)   # exploit → USER on target
    if done:
        record()
        return obs

    # ── Phase 5: privilege escalation → ROOT (WIN) ────────────────────────
    step(TARGET_ADDR, _PRIVESC)
    record()                           # winning state
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--out",     default=_DEFAULT_OUT)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = NASimEmuEnv(scenario_name=_SCENARIO, fully_obs=False)

    all_obs: list[np.ndarray] = []
    ep = 0
    while len(all_obs) < args.samples:
        all_obs.extend(scripted_episode(env, rng))
        ep += 1
        if ep % 1000 == 0:
            print(f"  episodes={ep}  obs={len(all_obs)}/{args.samples}")

    arr = np.array(all_obs[:args.samples], dtype=np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, arr)
    print(f"\nSaved {arr.shape} → {args.out}")

    # ── Diversity check ────────────────────────────────────────────────────
    kc = arr[:, :KC_DIM].reshape(-1, MAX_SLOTS, KC_FEATS)
    print("\nDiversity — phase values per KC slot:")
    for sl in range(MAX_SLOTS):
        phases  = kc[:, sl, 0]
        uniq    = sorted(set(round(float(x), 3) for x in phases))
        nonzero = float((phases > 0).mean())
        print(f"  slot {sl} {SLOT_ORDER[sl]}: phases={uniq}  nonzero={nonzero:.3f}")


if __name__ == "__main__":
    main()
