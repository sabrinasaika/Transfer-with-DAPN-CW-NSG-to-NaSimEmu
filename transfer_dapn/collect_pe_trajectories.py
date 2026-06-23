"""Collect pe (NaSim invariant) winning trajectories once and save to JSON.

Run this once; both traj_similarity.py and plot_p2_vs_pe.py load from the file.

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \\
      transfer_dapn/collect_pe_trajectories.py --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO  = Path(__file__).resolve().parents[1]
_DAPN  = _REPO / "transfer_dapn"
_DAGTS = Path("/home/ssaika@cs.utep.edu/NASimEmu-agents")

sys.path.insert(0, str(_DAGTS))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_DAPN))

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
from envs.scenario_load import patch_nasim_load_scenario

_SCENARIO_TWO = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_SCENARIO_ONE = str(_REPO / "scenarios" / "fixed_dmz_one_subnet_4host.v2.yaml")
_PE_MODEL = str(_DAGTS / "wandb/run-20260612_093816-biw5us4u/files/model.pt")

ATYPE_NAMES = ["ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData"]


def emu_act_to_atype(action_list, act_id: int) -> int:
    if act_id < 0:
        return 4
    cls, _ = action_list[act_id]
    name = cls.__name__
    if name == "SubnetScan":              return 0
    if name == "ServiceScan":             return 1
    if name == "Exploit":                 return 2
    if name in ("OSScan", "ProcessScan"): return 3
    if name == "PrivilegeEscalation":     return 4
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=50)
    ap.add_argument("--max-steps",  type=int, default=200)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--scenario",   default="two_subnet",
                    choices=["two_subnet", "one_subnet"])
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    scenario_yaml = _SCENARIO_ONE if args.scenario == "one_subnet" else _SCENARIO_TWO
    default_out = str(_DAPN / "artifacts/results/" /
                      f"pe_trajectories{'_one_subnet' if args.scenario == 'one_subnet' else ''}.json")
    out_path = args.out or default_out

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    patch_nasim_load_scenario()

    print(f"Loading pe … (scenario={args.scenario})")
    pe_net = NASimNetInvMAct()
    pe_net.load_state_dict(torch.load(_PE_MODEL, map_location="cpu"))
    pe_net.eval()
    # Do NOT set force_continue — let pe send terminal action (-1) naturally.
    # We stop when act_id < 0 and mark won if goal_reached at that point.

    env = NASimEmuEnv(scenario_name=scenario_yaml, fully_obs=False)
    trajs = []
    ep = 0

    print(f"Collecting {args.n} pe winning trajectories (seed={args.seed}) …")
    while len(trajs) < args.n:
        np.random.seed(args.seed + ep)
        env._generate_env()
        s = env.reset()
        traj = []
        won = False
        for _ in range(args.max_steps):
            with torch.no_grad():
                acts, _, _, _ = pe_net.forward([s])
            _, act_id = acts[0]
            atype = emu_act_to_atype(env.action_list, int(act_id))
            traj.append(atype)
            s, _, _, _ = env.step(acts[0])
            if int(act_id) < 0:
                # pe sent terminal action; check if goal actually reached
                if env.env.goal_reached(env.env.current_state):
                    won = True
                break
            if env.env.goal_reached(env.env.current_state):
                won = True
                break
        if won:
            trajs.append(traj)
        ep += 1
        if ep % 20 == 0:
            print(f"  ep={ep}  collected={len(trajs)}/{args.n}")

    env.close()
    lens = [len(t) for t in trajs]
    print(f"\nDone: {len(trajs)} trajectories, mean steps = {np.mean(lens):.1f}")

    counts = np.zeros(5)
    for t in trajs:
        for a in t:
            counts[a] += 1
    counts /= counts.sum()
    print("\nAction-type distribution:")
    for i, name in enumerate(ATYPE_NAMES):
        print(f"  {name:<20} {counts[i]:.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"trajectories": trajs, "mean_steps": float(np.mean(lens)),
                   "seed": args.seed, "scenario": args.scenario}, f)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
