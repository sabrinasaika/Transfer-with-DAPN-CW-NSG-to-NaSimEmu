"""Collect p1 action-distribution data from CyberWheel (its training domain).

Runs the trained DAPN policy in CW, records which KC slot it selects at each
timestep, and saves per-step counts as JSON.

Usage (must run in cyberwheel conda env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  conda run -n cyberwheel python transfer_dapn/collect_p1_cw_actions.py \
      --episodes 300 --out transfer_dapn/artifacts/results/p1_cw_actions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "transfer_dapn"))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "cyberwheel"))

from stable_baselines3 import PPO
from envs.kc_envs import make_cw_kc_env
from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
from envs.kill_chain import MAX_SLOTS

_ROOT      = Path(__file__).resolve().parent
_P1_POLICY = str(_ROOT / "artifacts/policies/cw_dapn_policy/best_model.zip")
_ENCODER   = str(_ROOT / "artifacts/models/dapn_encoder_kc7.pt.best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",  type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions from policy distribution (default: argmax)")
    ap.add_argument("--out", default="transfer_dapn/artifacts/results/p1_cw_actions.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build env: CW KC (already has AddContext) → DAPN encode → 73-D
    base_env  = make_cw_kc_env()     # returns AddContextWrapper(CWKillChainWrapper())
    enc_env   = DAPNEncoderWrapper(base_env, _ENCODER, device="cpu", mask_is_target=True)

    model = PPO.load(_P1_POLICY, device="cpu")

    # per-step slot counts: records[step] = {slot: count}
    records = []
    wins = 0

    for ep in range(args.episodes):
        obs, _ = enc_env.reset(seed=int(rng.integers(0, 2**31)))
        ep_traj = []
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            slot = int(action)
            ep_traj.append(slot)
            obs, reward, terminated, truncated, info = enc_env.step(slot)
            if terminated:
                wins += 1
                break
            if truncated:
                break

        records.append(ep_traj)
        if (ep + 1) % 50 == 0:
            print(f"  ep {ep+1}/{args.episodes}  wins={wins}")

    print(f"\np1 CW wins: {wins}/{args.episodes} ({100*wins/args.episodes:.1f}%)")

    # Build per-step distribution
    max_t = max(len(t) for t in records)
    per_step = []
    for t in range(max_t):
        counts = {str(s): 0 for s in range(MAX_SLOTS + 1)}
        n_active = 0
        for traj in records:
            if t < len(traj):
                slot = min(traj[t], MAX_SLOTS)
                counts[str(slot)] += 1
                n_active += 1
        per_step.append({"step": t, "n_active": n_active, "counts": counts})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"episodes": args.episodes, "max_steps": args.max_steps,
                   "wins": wins, "per_step": per_step}, f)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
