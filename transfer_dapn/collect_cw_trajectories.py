"""Collect p1 winning trajectories in CW (native training environment).

Maps CW kill-chain phase → 5 CW action types so results are comparable
with the NaSim trajectory_similarity.py plots.

Run with cyberwheel conda env:
  cd /home/ssaika@cs.utep.edu/NASimEmu
  conda run -n cyberwheel python transfer_dapn/collect_cw_trajectories.py --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DAPN = _REPO / "transfer_dapn"

sys.path.insert(0, str(_REPO / "cyberwheel"))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_DAPN))

import torch
from stable_baselines3 import PPO
from models.encoder import load_encoder, encode_obs
from envs.cw_wrapper import CWKillChainWrapper
from envs.kill_chain import MAX_SLOTS

_DEFAULT_ENCODER = str(_DAPN / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
_DEFAULT_P1      = str(_DAPN / "artifacts/policies/cw_dapn_policy/best_model.zip")
_DEFAULT_OUT     = str(_DAPN / "artifacts/results/cw_trajectories.json")

# CW kill-chain phase (kc offset within a host block) → action type index
# 0=pingsweep→ScanNetwork, 1=portscan→FindServices, 2=exploit→ExploitService,
# 3=on_host(lateral)→ExploitService, 4=escalate→ExfiltrateData, 5=impact→ExfiltrateData
_KC_TO_ATYPE = {0: 0, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4}

ATYPE_NAMES = [
    "ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData"
]


def _slot_kc_phase(env: CWKillChainWrapper, slot: int) -> int:
    """Return the CW KC phase (0-5) that _translate would pick for this slot."""
    if slot == MAX_SLOTS:
        return 0   # noop → pingsweep
    hname = env._env.red_agent.action_space.host_names[slot] if hasattr(
        env._env.red_agent.action_space, "host_names") else None

    # Re-derive phase from observation (same logic as cw_wrapper._translate)
    try:
        obs = env._env.red_agent.observation.obs
        from envs.kill_chain import HOST_NAMES
        hname = HOST_NAMES[slot]
        if not hname or hname not in obs:
            return 0
        h = obs[hname]
        if not h.get("sweeped", 0):    return 0
        if not h.get("scanned", 0):    return 1
        if not h.get("discovered", 0): return 2
        if not h.get("on_host", 0):    return 3
        if not h.get("escalated", 0) and slot != 3: return 4
        return 5
    except Exception:
        return 0


def collect_cw_trajectories(p1_model, encoder, n: int, max_steps: int = 200):
    """Collect n winning CW trajectories; record action type at each step."""
    trajs = []
    ep = 0
    while len(trajs) < n:
        env = CWKillChainWrapper()
        try:
            obs, _ = env.reset()
            traj = []
            won = False
            for _ in range(max_steps):
                obs44 = np.concatenate([obs, np.zeros(2, dtype=np.float32)])
                enc = encode_obs(encoder, obs44, mask_is_target=True)
                raw_slot, _ = p1_model.predict(enc, deterministic=True)
                raw_slot = min(int(raw_slot), MAX_SLOTS)

                # Get KC phase before stepping → map to action type
                kc_phase = _slot_kc_phase(env, raw_slot)
                atype = _KC_TO_ATYPE.get(kc_phase, 0)
                traj.append(atype)

                obs, r, term, trunc, info = env.step(raw_slot)
                if term or trunc:
                    if info.get("win"):
                        won = True
                    break
            if won:
                trajs.append(traj)
        finally:
            env.close()

        ep += 1
        if ep % 20 == 0:
            print(f"  ep={ep}  wins={len(trajs)}/{n}")

    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",         type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--encoder",   default=_DEFAULT_ENCODER)
    ap.add_argument("--policy-p1", default=_DEFAULT_P1)
    ap.add_argument("--out",       default=_DEFAULT_OUT)
    args = ap.parse_args()

    print("Loading encoder + p1 …")
    encoder  = load_encoder(args.encoder, device="cpu")
    p1_model = PPO.load(args.policy_p1)

    print(f"\nCollecting {args.n} CW winning trajectories …")
    trajs = collect_cw_trajectories(p1_model, encoder, args.n, args.max_steps)

    lens = [len(t) for t in trajs]
    print(f"\nDone: {len(trajs)} trajectories, mean length = {np.mean(lens):.1f}")

    counts = np.zeros(5)
    for t in trajs:
        for a in t:
            counts[a] += 1
    counts /= counts.sum()
    print("\nAction-type distribution:")
    for i, name in enumerate(ATYPE_NAMES):
        print(f"  {name:<20} {counts[i]:.3f}")

    print(f"\nSample: {' → '.join(ATYPE_NAMES[a] for a in trajs[0])}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"trajectories": trajs, "mean_steps": float(np.mean(lens))}, f)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
