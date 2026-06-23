"""Action-type distribution over winning trajectories — three policies.

  CW (DAPN)             — CW native winning trajectories (cw_trajectories.json)
  NSG→NaSim             — p1 DAPN transfer winning trajectories (collected live)
  NaSim Emulator Policy — pe winning trajectories (pe_trajectories.json)

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \\
      transfer_dapn/plot_action_distribution.py \\
      --out transfer_dapn/artifacts/results/action_distribution.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches

_REPO  = Path(__file__).resolve().parents[1]
_DAPN  = _REPO / "transfer_dapn"

sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_DAPN))

from stable_baselines3 import PPO
from envs.kill_chain import MAX_SLOTS
from envs.nasim_wrapper import NaSimKillChainWrapper
from envs.scenario_load import patch_nasim_load_scenario
from models.encoder import load_encoder, encode_obs

_DEFAULT_ENCODER = str(_DAPN / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
_DEFAULT_P1      = str(_DAPN / "artifacts/policies/cw_dapn_policy/best_model.zip")
_DEFAULT_CW_JSON = str(_DAPN / "artifacts/results/cw_trajectories.json")
_DEFAULT_PE_JSON = str(_DAPN / "artifacts/results/pe_trajectories.json")
_DEFAULT_OUT     = str(_DAPN / "artifacts/results/action_distribution.png")

ATYPE_NAMES  = ["ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData"]
ATYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
N_ATYPES = 5

_EAGER_ATYPES = [2, 0]   # ExploitService, ScanNetwork (hidden reset actions in p1)


def flat_to_atype(flat: int) -> int:
    local = flat % 10
    if local == 2:       return 0
    if local == 0:       return 1
    if 4 <= local <= 8:  return 2
    if local in (1, 3):  return 3
    if local == 9:       return 4
    return 0


def collect_p1_trajectories(p1_model, encoder, n: int, max_steps: int = 200):
    env = NaSimKillChainWrapper()
    trajs = []
    ep = 0
    while len(trajs) < n:
        obs, _ = env.reset()
        traj = list(_EAGER_ATYPES)
        for _ in range(max_steps):
            obs44 = np.concatenate([obs, np.zeros(2, dtype=np.float32)])
            enc = encode_obs(encoder, obs44, mask_is_target=True)
            raw_slot, _ = p1_model.predict(enc, deterministic=True)
            raw_slot = min(int(raw_slot), MAX_SLOTS)
            flat = env._translate(raw_slot)
            traj.append(flat_to_atype(flat))
            obs, _r, term, trunc, info = env.step(raw_slot)
            if term or trunc:
                if info.get("win"):
                    trajs.append(traj)
                break
        ep += 1
    env.close()
    return trajs


def atype_distribution(trajs):
    counts = np.zeros(N_ATYPES)
    for t in trajs:
        for a in t:
            counts[a] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def plot_distribution(cw_dist, p1_dist, pe_dist, out: str):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(N_ATYPES)
    w = 0.25

    ax.bar(x - w, cw_dist, w, color=ATYPE_COLORS, alpha=0.70, edgecolor="white")
    ax.bar(x,     p1_dist, w, color=ATYPE_COLORS, alpha=0.85, edgecolor="white", hatch="//")
    ax.bar(x + w, pe_dist, w, color=ATYPE_COLORS, alpha=0.45, edgecolor="white", hatch="..")

    ax.set_xticks(x)
    ax.set_xticklabels(ATYPE_NAMES, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Fraction of actions", fontsize=11)
    ax.set_title("Action-type distribution over winning trajectories", fontsize=12)

    lh = [
        matplotlib.patches.Patch(facecolor="grey", alpha=0.70, edgecolor="white",
                                  label="CW (DAPN)"),
        matplotlib.patches.Patch(facecolor="grey", alpha=0.85, edgecolor="white",
                                  hatch="//", label="NSG→NaSim"),
        matplotlib.patches.Patch(facecolor="grey", alpha=0.45, edgecolor="white",
                                  hatch="..", label="NaSim Emulator Policy"),
    ]
    ax.legend(handles=lh, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")

    print("\nAction-type distributions:")
    print(f"  {'Type':<20} {'CW (DAPN)':>12} {'NSG→NaSim':>12} {'NaSim Emu Policy':>18}")
    for i, name in enumerate(ATYPE_NAMES):
        print(f"  {name:<20} {cw_dist[i]:>12.3f} {p1_dist[i]:>12.3f} {pe_dist[i]:>18.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=50)
    ap.add_argument("--max-steps",  type=int, default=200)
    ap.add_argument("--encoder",    default=_DEFAULT_ENCODER)
    ap.add_argument("--policy-p1",  default=_DEFAULT_P1)
    ap.add_argument("--cw-json",    default=_DEFAULT_CW_JSON)
    ap.add_argument("--pe-json",    default=_DEFAULT_PE_JSON)
    ap.add_argument("--out",        default=_DEFAULT_OUT)
    args = ap.parse_args()

    patch_nasim_load_scenario()

    print("Loading CW trajectories …")
    with open(args.cw_json) as f:
        cw_data = json.load(f)
    cw_trajs = cw_data["trajectories"][:args.n]
    print(f"  {len(cw_trajs)} trajectories, mean steps = {cw_data['mean_steps']:.1f}")

    print("\nLoading pe trajectories …")
    with open(args.pe_json) as f:
        pe_data = json.load(f)
    pe_trajs = pe_data["trajectories"][:args.n]
    print(f"  {len(pe_trajs)} trajectories, mean steps = {pe_data['mean_steps']:.1f}")

    print("\nCollecting NSG→NaSim (p1) winning trajectories …")
    encoder  = load_encoder(args.encoder, device="cpu")
    p1_model = PPO.load(args.policy_p1)
    p1_trajs = collect_p1_trajectories(p1_model, encoder, args.n, args.max_steps)
    print(f"  {len(p1_trajs)} trajectories, mean steps = {np.mean([len(t) for t in p1_trajs]):.1f}")

    cw_dist = atype_distribution(cw_trajs)
    p1_dist = atype_distribution(p1_trajs)
    pe_dist = atype_distribution(pe_trajs)

    plot_distribution(cw_dist, p1_dist, pe_dist, args.out)


if __name__ == "__main__":
    main()
