"""Evaluate p2 (raw-CW policy) in NaSim using a kill-chain obs adapter.

Adapter: NaSim partially-observable matrix → 50-D CW obs_vec
  For each host in SLOT_ORDER, extract 7 kill-chain status flags:
    [1.0, sweeped, scanned, discovered, on_host, escalated, impacted]
  Then normalise: (raw + 1) / 5  (matches CWNativeWrapper._process_obs)

Action mapping: p2 CW action (0-41) → NaSim (addr, action_id)
  host_idx = action // 6
  kc_phase = action  % 6
  kc_phase→NaSim: 0→SubnetScan(2), 1→ServiceScan(0), 2-3→Exploit, 4-5→PrivEsc(9)

Collects winning trajectories and saves action-type sequences as JSON.

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \\
      transfer_dapn/eval_p2_nasim.py --n 50 \\
      --out transfer_dapn/artifacts/results/p2_trajectories.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO  = Path(__file__).resolve().parents[1]
_DAPN  = _REPO / "transfer_dapn"

sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_DAPN))

from stable_baselines3 import PPO
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel
from envs.kill_chain import SLOT_ORDER, MAX_SLOTS
from envs.host_map import EXPLOIT_LOCAL
from envs.scenario_load import patch_nasim_load_scenario

_SCENARIO  = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_DEFAULT_P2 = str(_DAPN / "artifacts/policies/cw_raw_policy/best_model.zip")
_DEFAULT_OUT = str(_DAPN / "artifacts/results/p2_trajectories.json")

# 50-D CW obs constants
_MAX_HOSTS  = 7
_HOST_FEAT  = 7
_OBS_SIZE   = _MAX_HOSTS * _HOST_FEAT + 1   # 50
_MAX_ATTR   = 4

# CW kc_phase → NaSim local action offset
_PHASE_TO_LOCAL = {
    0: 2,   # pingsweep  → SubnetScan
    1: 0,   # portscan   → ServiceScan
    2: None,  # exploit  → host-specific exploit (EXPLOIT_LOCAL)
    3: None,  # on_host  → same exploit (lateral)
    4: 9,   # escalate   → PrivEsc
    5: 9,   # impact     → PrivEsc
}

ATYPE_NAMES  = ["ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData"]
ATYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _local_to_atype(local: int) -> int:
    if local == 2:          return 0   # SubnetScan  → ScanNetwork
    if local == 0:          return 1   # ServiceScan → FindServices
    if 4 <= local <= 8:     return 2   # Exploit     → ExploitService
    if local in (1, 3):     return 3   # OSScan/ProcessScan → FindData
    if local == 9:          return 4   # PrivEsc     → ExfiltrateData
    return 0


def nasim_matrix_to_cw_obs(s: np.ndarray) -> np.ndarray:
    """NaSim partially-observable matrix → 50-D CW obs_vec.

    Reads host status from NaSim host vectors and maps to the 7 CW
    kill-chain flags per host, in SLOT_ORDER.
    """
    host_map = {}
    for row in s[:-1]:
        hv = HostVector(row)
        addr = tuple(int(x) for x in hv.address)
        if addr == (0, 0):
            continue
        host_map[addr] = hv

    raw = np.zeros(_OBS_SIZE, dtype=np.float32)
    for slot, addr in enumerate(SLOT_ORDER):
        base = slot * _HOST_FEAT
        hv = host_map.get(addr)
        if hv is None:
            # host not yet observed — all flags zero except address_flag
            raw[base + 0] = 0.0
            continue

        access     = int(hv.access)
        discovered = float(hv.discovered) > 0
        compromised = float(hv.compromised) > 0
        on_host    = access >= AccessLevel.USER
        escalated  = access >= AccessLevel.ROOT
        # impacted = target host with root access (sensitive host)
        is_sensitive = addr in {(3, 0)}   # fixed scenario target
        impacted = is_sensitive and escalated

        raw[base + 0] = 1.0          # address_flag: host exists in scenario
        raw[base + 1] = float(discovered)    # sweeped
        raw[base + 2] = float(discovered)    # scanned (approximated by discovered)
        raw[base + 3] = float(compromised)   # discovered (initial foothold)
        raw[base + 4] = float(on_host)       # on_host
        raw[base + 5] = float(escalated)     # escalated
        raw[base + 6] = float(impacted)      # impacted

    raw[_OBS_SIZE - 1] = 0.0   # standalone: unused

    # Apply same normalisation as CWNativeWrapper._process_obs
    return ((raw + 1.0) / (_MAX_ATTR + 1.0)).astype(np.float32)


def cw_action_to_nasim(action: int, action_list) -> tuple:
    """p2 CW action (0-41) → NaSim (addr, action_id)."""
    host_idx = action // 6
    kc_phase = action % 6

    if host_idx >= len(SLOT_ORDER):
        host_idx = 0
    addr = SLOT_ORDER[host_idx]

    local = _PHASE_TO_LOCAL[kc_phase]
    if local is None:
        # Exploit: use host-specific exploit from EXPLOIT_LOCAL
        local_kc = EXPLOIT_LOCAL.get(addr, 4)
        # Convert KC flat local offset to NASimEmuEnv action_id
        from envs.host_map import _EXPLOIT_NAME
        name = _EXPLOIT_NAME.get((addr, local_kc))
        if name:
            try:
                act_id = next(i for i, (_, p) in enumerate(action_list)
                              if p.get("name") == name)
            except StopIteration:
                act_id = 4
        else:
            act_id = 4
        return (addr, act_id)

    if local == 9:
        try:
            act_id = next(i for i, (_, p) in enumerate(action_list)
                          if p.get("name") == "pe_kernel")
        except StopIteration:
            act_id = 9
        return (addr, act_id)

    return (addr, local)


def collect_p2_trajectories(p2_model, n: int, max_steps: int = 200):
    """Evaluate p2 in NaSim via obs adapter; collect winning trajectories."""
    env = NASimEmuEnv(scenario_name=_SCENARIO, fully_obs=False)
    trajs = []
    ep = 0
    wins = 0

    while len(trajs) < n and ep < n * 10:
        env._generate_env()
        s = env.reset()
        traj = []
        won = False

        for _ in range(max_steps):
            cw_obs = nasim_matrix_to_cw_obs(s)
            raw_action, _ = p2_model.predict(cw_obs, deterministic=True)
            nasim_action = cw_action_to_nasim(int(raw_action), env.action_list)

            # Map to action type for recording
            _, act_id = nasim_action
            local = act_id % 10
            atype = _local_to_atype(local)
            traj.append(atype)

            s, _, _, _ = env.step(nasim_action)
            if env.env.goal_reached(env.env.current_state):
                won = True
                break

        if won:
            trajs.append(traj)
            wins += 1
        ep += 1
        if ep % 20 == 0:
            print(f"  ep={ep}  wins={wins}  collected={len(trajs)}/{n}")

    env.close()
    print(f"\nTotal: {ep} episodes, {wins} wins ({100*wins/max(ep,1):.1f}% win rate)")
    return trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=50)
    ap.add_argument("--max-steps",  type=int, default=200)
    ap.add_argument("--policy-p2",  default=_DEFAULT_P2)
    ap.add_argument("--out",        default=_DEFAULT_OUT)
    args = ap.parse_args()

    patch_nasim_load_scenario()

    print(f"Loading p2 from {args.policy_p2} …")
    p2_model = PPO.load(args.policy_p2)

    print(f"\nEvaluating p2 in NaSim (obs adapter: NaSim→50-D CW) …")
    trajs = collect_p2_trajectories(p2_model, args.n, args.max_steps)

    if not trajs:
        print("No winning trajectories collected — p2 fails to transfer.")
        result = {"trajectories": [], "mean_steps": 0.0, "win_rate": 0.0}
    else:
        lens = [len(t) for t in trajs]
        print(f"Collected {len(trajs)} trajectories, mean steps = {np.mean(lens):.1f}")
        counts = np.zeros(5)
        for t in trajs:
            for a in t:
                counts[a] += 1
        counts /= counts.sum()
        print("\nAction-type distribution:")
        for i, name in enumerate(ATYPE_NAMES):
            print(f"  {name:<20} {counts[i]:.3f}")
        result = {"trajectories": trajs, "mean_steps": float(np.mean(lens)),
                  "win_rate": len(trajs) / max(len(trajs) * 10, 1)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
