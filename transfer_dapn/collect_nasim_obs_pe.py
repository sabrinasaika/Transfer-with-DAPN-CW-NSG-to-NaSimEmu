"""Collect diverse NaSim KC+ctx observations using the trained pe policy.

pe actually progresses the kill chain, producing varied states across all
7 slots — unlike a random policy which stays at the initial state forever.

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \
      transfer_dapn/collect_nasim_obs_pe.py --samples 50000
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

# ── pe config ─────────────────────────────────────────────────────────────────
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

import torch
from nasim_problem.nasim_net_inv_mact import NASimNetInvMAct
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from envs.kill_chain import (
    KC_DIM, KC_FEATS, MAX_SLOTS, SLOT_ORDER,
    ENTRY_SLOT, TARGET_SLOT, CTX_DIM, build_kc_obs,
)
from envs.scenario_load import patch_nasim_load_scenario
patch_nasim_load_scenario()

_SCENARIO = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_PE_MODEL = str(_DAGTS / "wandb/run-20260612_093816-biw5us4u/files/model.pt")
_DEFAULT_OUT = str(_REPO / "transfer_dapn/data/nasim_kc_obs_pe.npy")


def _phase_norm(access: float, discovered: bool, compromised: bool) -> float:
    if not discovered and access < AccessLevel.USER:
        return 0.0
    if access >= AccessLevel.ROOT:
        return 1.0
    if access >= AccessLevel.USER:
        return 0.6
    if compromised:
        return 0.4
    return 0.2


def matrix_to_kc_ctx(s: np.ndarray,
                     last_action: float = 0.0,
                     last_reward: float = 0.0) -> np.ndarray:
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
            slots.append((0.0, 0.0, 0.0,
                          float(slot == ENTRY_SLOT), 0.0,
                          float(slot == TARGET_SLOT)))
        else:
            access      = float(hv.access)
            discovered  = float(hv.discovered) > 0.5
            on_host     = access >= AccessLevel.USER
            compromised = float(hv.compromised) > 0.5
            phase       = _phase_norm(access, discovered, compromised)
            reachable   = 1.0 if (discovered or slot == ENTRY_SLOT) else 0.0
            slots.append((phase, reachable, float(on_host),
                          float(slot == ENTRY_SLOT), reachable,
                          float(slot == TARGET_SLOT)))

    kc  = build_kc_obs(slots)
    ctx = np.array([
        float(last_action) / max(1.0, float(MAX_SLOTS)),
        float(np.clip(last_reward / 100.0, -1.0, 1.0)),
    ], dtype=np.float32)
    return np.concatenate([kc, ctx]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples",  type=int, default=50000)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--out",      default=_DEFAULT_OUT)
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("Loading pe …")
    net = NASimNetInvMAct()
    ckpt = torch.load(_PE_MODEL, map_location="cpu")
    net.load_state_dict(ckpt)
    net.eval()
    net.set_force_continue(True)

    env = NASimEmuEnv(scenario_name=_SCENARIO, fully_obs=False)
    SLOT_SET = set(SLOT_ORDER)

    obs_list: list[np.ndarray] = []
    ep = 0

    # look up e_proftpd id (alphabetical sort: drupal=4, elasticsearch=5, phpwiki=6, proftpd=7)
    env._generate_env()
    env.reset()
    exploit_id = next(
        i for i, (cls, params) in enumerate(env.action_list)
        if params.get("name") == "e_proftpd"
    )
    _SCAN_SUBNET = 2
    ENTRY_ADDR = SLOT_ORDER[ENTRY_SLOT]

    while len(obs_list) < args.samples:
        env._generate_env()
        s = env.reset()
        # Eager entry: exploit then subnet scan (matches CW start state)
        for act_id in (exploit_id, _SCAN_SUBNET):
            s, _, _, _ = env.step((ENTRY_ADDR, act_id))
        last_action = 0.0
        last_reward = 0.0

        for _ in range(args.max_steps):
            # record KC+ctx obs BEFORE the action
            kc_ctx = matrix_to_kc_ctx(s, last_action, last_reward)
            obs_list.append(kc_ctx)

            with torch.no_grad():
                pe_acts, _, _, _ = net.forward([s])
            pe_target, pe_act_id = pe_acts[0]
            pe_addr = tuple(int(x) for x in pe_target)
            pe_slot = (SLOT_ORDER.index(pe_addr)
                       if pe_addr in SLOT_SET else MAX_SLOTS)
            last_action = float(pe_slot)

            s, r, done, info = env.step(pe_acts[0])
            last_reward = float(r * 10.0)   # undo NASimEmuEnv /10 scaling

            if done:
                break

        ep += 1
        if ep % 100 == 0:
            print(f"  episodes={ep}  obs collected={len(obs_list)}/{args.samples}")

    arr = np.array(obs_list[:args.samples], dtype=np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, arr)
    print(f"\nSaved {arr.shape} → {args.out}")

    # quick diversity check
    kc = arr[:, :KC_DIM].reshape(-1, MAX_SLOTS, KC_FEATS)
    print("\nDiversity check — unique phase values per slot:")
    for sl in range(MAX_SLOTS):
        phases = kc[:, sl, 0]
        uniq   = sorted(set(round(float(x), 3) for x in phases))
        print(f"  slot {sl}: {uniq}")


if __name__ == "__main__":
    main()
