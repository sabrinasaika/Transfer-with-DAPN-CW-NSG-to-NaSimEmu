"""Shadow rollout: compare p1 (CW DAPN) and pe (NaSim Invariant GNN) on the SAME states.

At every step in a NaSim episode (pe drives), both policies are queried on the
identical matrix observation. We record KC-slot choices and plot their
time-varying distributions plus step-level agreement.

Usage (nasimemu-env — has torch_geometric + SB3):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python transfer_dapn/shadow_rollout.py \\
      --episodes 300 --max-steps 50 \\
      --out transfer_dapn/artifacts/results/action_dist.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from stable_baselines3 import PPO
from nasim_problem.nasim_net_inv_mact import NASimNetInvMAct
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from envs.kill_chain import (
    KC_DIM, CTX_DIM, MAX_SLOTS, SLOT_ORDER,
    inactive_slot_feats, slot_feats_from_state, build_kc_obs,
)
from envs.host_map import (
    discover_host_layout, flat_action, flat_to_slot, flat_to_emu_action,
    EXPLOIT_LOCAL,
)
from envs.kill_chain import ENTRY_SLOT, ENTRY_ADDR
from envs.nasim_wrapper import NaSimKillChainWrapper
from envs.scenario_load import patch_nasim_load_scenario
from models.encoder import load_encoder, encode_obs

_SCENARIO  = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_PE_MODEL  = str(_DAGTS / "wandb/run-20260612_093816-biw5us4u/files/model.pt")
_DEFAULT_ENCODER = str(_DAPN / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
_DEFAULT_P1      = str(_DAPN / "artifacts/policies/cw_dapn_policy/best_model.zip")

SLOT_LABELS = [
    "Entry (1,0)", "Target (3,0)", "User-0 (2,0)", "User-1 (2,1)",
    "User-2 (2,2)", "Svc-1  (3,1)", "Svc-2  (3,2)", "Noop / Terminal",
]
SLOT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
SLOT_SET = set(SLOT_ORDER)

# CW-style action-type categories matching Figure 12/13 legend
# NaSim mapping: ScanNetwork=SubnetScan(2), FindServices=ServiceScan(0),
#   ExploitService=Exploit(4-8), FindData=OSScan(1)+ProcessScan(3),
#   ExfiltrateData=PrivEsc(9)
ATYPE_LABELS = [
    "ActionType.ScanNetwork",
    "ActionType.FindServices",
    "ActionType.ExploitService",
    "ActionType.FindData",
    "ActionType.ExfiltrateData",
]
ATYPE_COLORS = [
    "#1f77b4",   # blue  – ScanNetwork
    "#ff7f0e",   # orange – FindServices
    "#2ca02c",   # green  – ExploitService
    "#d62728",   # red    – FindData
    "#9467bd",   # purple – ExfiltrateData
]
N_ATYPES = len(ATYPE_LABELS)


def local_to_atype(local: int) -> int:
    """NaSim local action offset → CW action-type index."""
    if local == 2:   return 0  # SubnetScan  → ScanNetwork
    if local == 0:   return 1  # ServiceScan → FindServices
    if 4 <= local <= 8: return 2  # Exploit  → ExploitService
    if local in (1, 3): return 3  # OSScan/ProcessScan → FindData
    if local == 9:   return 4  # PrivEsc     → ExfiltrateData
    return 0  # fallback


def emu_action_to_atype(action_list, act_id: int) -> int:
    """NASimEmuEnv action_id → CW action-type index."""
    if act_id < 0:
        return 0
    cls, _ = action_list[act_id]
    name = cls.__name__
    if name == "SubnetScan":
        return 0
    if name == "ServiceScan":
        return 1
    if name == "Exploit":
        return 2
    if name in ("OSScan", "ProcessScan"):
        return 3
    if name == "PrivilegeEscalation":
        return 4
    return 0


def pe_id_to_atype(act_id: int) -> int:
    """NASimEmuEnv action_id → CW action-type index (same offsets as NaSim local)."""
    if act_id == -1:
        return 0   # Terminal treated as ScanNetwork (noop-like)
    return local_to_atype(act_id)

# Match NaSimKillChainWrapper hidden entry setup (non-emulated path).
_SCAN_SUBNET = 2


def _apply_eager_entry(env, kc: NaSimKillChainWrapper):
    """Entry exploit then subnet scan (SubnetScan needs host_compromised=True)."""
    exploit_id = next(
        i for i, (_, params) in enumerate(env.action_list)
        if params.get("name") == "e_proftpd"
    )
    s = None
    for action_id in (exploit_id, _SCAN_SUBNET):
        s, _, _, _ = env.step((ENTRY_ADDR, action_id))
    return s


def matrix_to_kc_ctx(s, last_action: float = 0.0, last_reward: float = 0.0) -> np.ndarray:
    """NASimEmuEnv matrix obs → 44-D KC+ctx (matches NaSimKillChainWrapper)."""
    host_map = {}
    for row in s[:-1]:
        hv = HostVector(row)
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
            access = int(hv.access)
            discovered = bool(float(hv.discovered) > 0)
            on_host = access >= AccessLevel.USER
            slots.append(slot_feats_from_state(
                slot, access, discovered, bool(float(hv.compromised) > 0),
                on_host=float(on_host),
            ))

    kc = build_kc_obs(slots)
    ctx = np.array([
        last_action / max(1.0, float(MAX_SLOTS)),
        float(np.clip(last_reward / 100.0, -1.0, 1.0)),
    ], dtype=np.float32)
    return np.concatenate([kc, ctx]).astype(np.float32)


def _attach_kc_translator(nasim_env) -> NaSimKillChainWrapper:
    """Reuse NaSimKillChainWrapper._translate on the live NASimEmu inner env."""
    kc = NaSimKillChainWrapper.__new__(NaSimKillChainWrapper)
    kc._env = nasim_env
    kc._layout = discover_host_layout(nasim_env.scenario.address_space)
    kc._emulated = False
    return kc


def p1_effective_slot(kc: NaSimKillChainWrapper, raw_slot: int) -> int:
    """Policy KC slot → host actually targeted after NaSimKillChainWrapper logic."""
    flat = kc._translate(raw_slot if raw_slot < MAX_SLOTS else MAX_SLOTS)
    return flat_to_slot(flat, kc._layout)


def pe_action_to_slot(pe_target, pe_act_id: int) -> int:
    if int(pe_act_id) == -1:
        return MAX_SLOTS
    addr = tuple(int(x) for x in pe_target)
    return SLOT_ORDER.index(addr) if addr in SLOT_SET else MAX_SLOTS


def _query_policies(s, kc, pe_net, p1_model, encoder, action_list,
                    last_action, last_reward):
    """Return (p1_atype, pe_atype, p1_slot, pe_slot, p1_raw) on shared matrix obs."""
    with torch.no_grad():
        pe_acts, _, _, _ = pe_net.forward([s])
    pe_target, pe_act_id = pe_acts[0]
    pe_slot  = pe_action_to_slot(pe_target, int(pe_act_id))
    pe_atype = emu_action_to_atype(action_list, int(pe_act_id))

    kc_ctx = matrix_to_kc_ctx(s, last_action, last_reward)
    enc = encode_obs(encoder, kc_ctx, mask_is_target=True)
    p1_action, _ = p1_model.predict(enc, deterministic=True)
    p1_raw   = min(int(p1_action), MAX_SLOTS)
    p1_slot  = p1_effective_slot(kc, p1_raw)
    p1_flat  = kc._translate(p1_raw if p1_raw < MAX_SLOTS else MAX_SLOTS)
    _, p1_act_id = flat_to_emu_action(p1_flat, kc._layout, action_list)
    p1_atype = emu_action_to_atype(action_list, p1_act_id)
    return p1_atype, pe_atype, p1_slot, pe_slot, p1_raw


def run_shadow_rollout(pe_net, p1_model, encoder, n_episodes: int,
                       max_steps: int, seed: int, driver: str = "p1"):
    """Shadow rollout on identical NaSim states. driver='p1' (default) or 'pe'."""
    env = NASimEmuEnv(scenario_name=_SCENARIO, fully_obs=False)

    p1_trajs, pe_trajs = [], []
    p1_atype_trajs, pe_atype_trajs = [], []
    agreements = []
    wins = 0

    for ep in range(n_episodes):
        env._generate_env()
        s = env.reset()
        kc = _attach_kc_translator(env.env)
        last_action, last_reward = 0.0, 0.0
        p1_traj, pe_traj = [], []
        p1_atype_traj, pe_atype_traj = [], []
        ep_agree = []

        for _ in range(max_steps):
            p1_atype, pe_atype, p1_slot, pe_slot, p1_raw = _query_policies(
                s, kc, pe_net, p1_model, encoder, env.action_list,
                last_action, last_reward)

            p1_traj.append(p1_slot)
            pe_traj.append(pe_slot)
            p1_atype_traj.append(p1_atype)
            pe_atype_traj.append(pe_atype)
            ep_agree.append(int(p1_atype == pe_atype))

            if driver == "p1":
                flat = kc._translate(p1_raw)
                step_action = flat_to_emu_action(flat, kc._layout, env.action_list)
            else:
                with torch.no_grad():
                    pe_acts, _, _, _ = pe_net.forward([s])
                step_action = pe_acts[0]

            s, r, done, info = env.step(step_action)
            last_action = float(p1_raw) / max(1.0, float(MAX_SLOTS))
            last_reward = float(r)
            if done:
                wins += 1
                break

        p1_trajs.append(p1_traj)
        pe_trajs.append(pe_traj)
        p1_atype_trajs.append(p1_atype_traj)
        pe_atype_trajs.append(pe_atype_traj)
        if ep_agree:
            agreements.extend(ep_agree)

        if (ep + 1) % 50 == 0:
            agree_pct = 100 * np.mean(agreements) if agreements else 0.0
            print(f"  ep {ep+1}/{n_episodes}  wins={wins}  agreement={agree_pct:.1f}%")

    agree_rate = float(np.mean(agreements)) if agreements else 0.0
    print(f"\n{driver} drives: wins={wins}/{n_episodes} ({100*wins/n_episodes:.1f}%)")
    print(f"Step-level action-type agreement (same state): "
          f"{100*agree_rate:.1f}%  ({len(agreements)} steps)")
    return p1_atype_trajs, pe_atype_trajs, agree_rate


def _build_dist(trajs: list[list[int]], max_steps: int):
    """Count CW action-type occurrences per time step across all trajectories."""
    counts = [np.zeros(N_ATYPES, dtype=np.float32) for _ in range(max_steps)]
    n_active = [0] * max_steps
    for traj in trajs:
        for step, atype in enumerate(traj):
            if step >= max_steps:
                break
            counts[step][min(atype, N_ATYPES - 1)] += 1
            n_active[step] += 1
    return counts, n_active


def plot_action_dist(records: dict, max_steps: int, out: str,
                     agreement: float, driver: str = "p1") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title in zip(axes, ["p1", "pe"],
                               ["CW→NaSim Transfer (p1)",
                                "NaSim Invariant (pe)"]):
        counts, n_active = _build_dist(records[key], max_steps)
        xs_valid = [t for t in range(max_steps) if n_active[t] >= 10]
        if not xs_valid:
            continue

        bottoms = np.zeros(len(xs_valid))
        for atype_idx in range(N_ATYPES):
            ratios = np.array([
                counts[t][atype_idx] / max(1, n_active[t]) for t in xs_valid
            ])
            ax.bar(xs_valid, ratios, bottom=bottoms,
                   color=ATYPE_COLORS[atype_idx],
                   label=ATYPE_LABELS[atype_idx],
                   edgecolor="none", width=0.9)
            bottoms += ratios

        ax2 = ax.twinx()
        ax2.plot(xs_valid, [n_active[t] for t in xs_valid],
                 "k--", linewidth=1.2, label="# trajectories")
        ax2.set_ylabel("# trajectories", fontsize=9)
        ax2.spines["top"].set_visible(False)

        ax.set_title(f"Action Distribution — {title}", fontsize=11)
        ax.set_xlabel("Time step", fontsize=9)
        ax.set_ylabel("Fraction of actions", fontsize=9)
        ax.set_xlim(-0.5, max(xs_valid) + 0.5)
        ax.set_ylim(0.0, 1.05)
        ax.spines["top"].set_visible(False)

        handles_bar, labels_bar = ax.get_legend_handles_labels()
        handles_line, labels_line = ax2.get_legend_handles_labels()
        ax.legend(handles_line + handles_bar,
                  labels_line + labels_bar,
                  fontsize=7, loc="lower center",
                  bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)

    fig.suptitle(
        f"Action type distributions",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=_DEFAULT_ENCODER)
    ap.add_argument("--policy-p1", default=_DEFAULT_P1)
    ap.add_argument("--driver", choices=["p1", "pe"], default="pe",
                    help="Who steps the env (default pe: compare on pe's exploration states)")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=50,
                    help="Plot window (steps recorded per episode)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(_DAPN / "artifacts/results/action_dist.png"))
    args = ap.parse_args()

    print("Loading pe (NASimNetInvMAct) …")
    pe_net = NASimNetInvMAct()
    pe_net.load_state_dict(torch.load(_PE_MODEL, map_location="cpu"))
    pe_net.eval()
    pe_net.set_force_continue(True)

    print("Loading p1 (SB3) + DAPN encoder …")
    encoder = load_encoder(args.encoder, device="cpu")
    p1_model = PPO.load(args.policy_p1)

    print(f"\nShadow rollout: {args.episodes} episodes, max {args.max_steps} steps")
    print(f"  {args.driver} drives env; both policies queried each step\n")

    patch_nasim_load_scenario()

    p1_atype_trajs, pe_atype_trajs, agree = run_shadow_rollout(
        pe_net, p1_model, encoder, args.episodes, args.max_steps,
        args.seed, driver=args.driver)

    plot_action_dist({"p1": p1_atype_trajs, "pe": pe_atype_trajs},
                     args.max_steps, args.out, agree, driver=args.driver)


if __name__ == "__main__":
    main()
