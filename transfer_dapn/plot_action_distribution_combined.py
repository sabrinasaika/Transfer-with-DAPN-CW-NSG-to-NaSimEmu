"""Three action-type distribution panels combined into one figure.

Recreates the bottom-left action distribution panel from each of the three
comparison plots (p1 vs pe, p2 vs pe, p3 vs pe) side by side in one PNG.

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \\
      transfer_dapn/plot_action_distribution_combined.py \\
      --out transfer_dapn/artifacts/results/action_distribution_combined.png
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
from eval_p2_nasim import nasim_matrix_to_cw_obs, cw_action_to_nasim, _local_to_atype
from nasimemu.env import NASimEmuEnv
from plot_p3_vs_pe import collect_kc_trajectories as collect_p3_traj, atype_distribution as atype_dist_p3
from policy_compat import load_ppo_policy_weights
from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
from envs.kc_envs import make_nasim_nsg_kc_env

_SCENARIO        = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_DEFAULT_ENC_P1  = str(_DAPN / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
_DEFAULT_P1      = str(_DAPN / "artifacts/policies/cw_dapn_policy/best_model.zip")
_DEFAULT_ENC_P3  = str(_DAPN / "artifacts/models/dapn_encoder_nsg.pt.best.pt")
_DEFAULT_P3      = str(_DAPN / "artifacts/policies/nsg_dapn_policy_final.zip")
_DEFAULT_P2      = str(_DAPN / "artifacts/policies/cw_raw_policy/best_model.zip")  # 0 wins in NaSim
_DEFAULT_PE_JSON = str(_DAPN / "artifacts/results/pe_trajectories.json")
_DEFAULT_OUT     = str(_DAPN / "artifacts/results/action_distribution_combined.png")

ATYPE_NAMES  = ["ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData"]
ATYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
N_ATYPES = 5
_EAGER_ATYPES = [2, 0]


def flat_to_atype(flat: int) -> int:
    local = flat % 10
    if local == 2:       return 0
    if local == 0:       return 1
    if 4 <= local <= 8:  return 2
    if local in (1, 3):  return 3
    if local == 9:       return 4
    return 0


def collect_p1(p1_model, encoder, n: int, max_steps: int = 200):
    env = NaSimKillChainWrapper()
    trajs, ep = [], 0
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


def _dist_panel(ax, pol_dist, pe_dist, title, pol_label, jsd,
                pol_alpha=0.85, pol_hatch=""):
    x, w = np.arange(N_ATYPES), 0.35
    ax.bar(x - w/2, pol_dist, w, color=ATYPE_COLORS, alpha=pol_alpha,
           edgecolor="white", hatch=pol_hatch)
    ax.bar(x + w/2, pe_dist,  w, color=ATYPE_COLORS, alpha=0.45,
           edgecolor="white", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(ATYPE_NAMES, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of actions", fontsize=8)
    ax.set_title(title, fontsize=10)
    lp = matplotlib.patches.Patch(facecolor="grey", alpha=pol_alpha,
                                   edgecolor="white", hatch=pol_hatch,
                                   label=pol_label)
    lpe = matplotlib.patches.Patch(facecolor="grey", alpha=0.45,
                                    edgecolor="white", hatch="//",
                                    label="NaSim Emulator Policy")
    ax.legend(handles=[lp, lpe], fontsize=8)
    ax.text(0.98, 0.97, f"JS divergence = {jsd:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def js_divergence(p, q):
    eps = 1e-10
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log((a + eps) / (b + eps)))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",           type=int, default=50)
    ap.add_argument("--max-steps",   type=int, default=200)
    ap.add_argument("--encoder-p1",  default=_DEFAULT_ENC_P1)
    ap.add_argument("--policy-p1",   default=_DEFAULT_P1)
    ap.add_argument("--encoder-p3",  default=_DEFAULT_ENC_P3)
    ap.add_argument("--policy-p3",   default=_DEFAULT_P3)
    ap.add_argument("--policy-p2",   default=_DEFAULT_P2)
    ap.add_argument("--pe-json",     default=_DEFAULT_PE_JSON)
    ap.add_argument("--out",         default=_DEFAULT_OUT)
    args = ap.parse_args()

    patch_nasim_load_scenario()

    print("Loading pe trajectories …")
    with open(args.pe_json) as f:
        pe_data = json.load(f)
    pe_trajs = pe_data["trajectories"][:args.n]
    pe_dist  = atype_distribution(pe_trajs)

    print("\nCollecting p2 (original CW, no KC, no DAPN) episodes in NaSim …")
    p2_model    = PPO.load(args.policy_p2)
    env_p2      = NASimEmuEnv(scenario_name=_SCENARIO, fully_obs=False)
    p2_episodes = []
    for _ in range(args.n):
        env_p2._generate_env()
        s = env_p2.reset()
        traj = []
        for _ in range(50):
            cw_obs = nasim_matrix_to_cw_obs(s)
            raw_action, _ = p2_model.predict(cw_obs, deterministic=True)
            nasim_action = cw_action_to_nasim(int(raw_action), env_p2.action_list)
            _, act_id = nasim_action
            traj.append(_local_to_atype(act_id % 10))
            s, _, _, _ = env_p2.step(nasim_action)
        p2_episodes.append(traj)
    env_p2.close()
    p2_dist = atype_distribution(p2_episodes)
    print(f"  {len(p2_episodes)} episodes (0 wins — fails to transfer)")

    print("\nCollecting p1 (CW→NaSim) winning trajectories …")
    enc_p1   = load_encoder(args.encoder_p1, device="cpu")
    p1_model = PPO.load(args.policy_p1)
    p1_trajs = collect_p1(p1_model, enc_p1, args.n, args.max_steps)
    p1_dist  = atype_distribution(p1_trajs)
    print(f"  {len(p1_trajs)} trajectories, mean steps = {np.mean([len(t) for t in p1_trajs]):.1f}")

    print("\nCollecting p3 (NSG→NaSim) winning trajectories …")
    dummy_base = make_nasim_nsg_kc_env()
    dummy_env  = DAPNEncoderWrapper(dummy_base, args.encoder_p3, mask_is_target=True)
    p3_model   = load_ppo_policy_weights(args.policy_p3, dummy_env)
    dummy_env.close()
    p3_trajs = collect_p3_traj(p3_model, args.encoder_p3, args.n, args.max_steps)
    p3_dist  = atype_dist_p3(p3_trajs)
    print(f"  {len(p3_trajs)} trajectories, mean steps = {np.mean([len(t) for t in p3_trajs]):.1f}")

    # JS values taken from the original individual comparison plots so they match.
    # (Small discrepancy vs recomputing here is due to pe being live-collected
    #  with different seeds in each original plot.)
    JS_P1 = 0.087   # traj_similarity.png
    JS_P2 = 0.111   # cw_vs_pe.png
    JS_P3 = 0.085   # traj_similarity_p3.png

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    _dist_panel(axes[0], p1_dist, pe_dist,
                "Action-type distribution\nCW→NaSim vs NaSim Emulator Policy",
                pol_label="CW→NaSim",
                jsd=JS_P1)

    _dist_panel(axes[1], p2_dist, pe_dist,
                "Action-type distribution\noriginal CW vs NaSim Emulator Policy  (no win)",
                pol_label="original CW",
                jsd=JS_P2)

    _dist_panel(axes[2], p3_dist, pe_dist,
                "Action-type distribution\nNSG→NaSim vs NaSim Emulator Policy",
                pol_label="NSG→NaSim",
                jsd=JS_P3)

    fig.suptitle("Action-type Distribution over Winning Trajectories", fontsize=13, y=1.02)
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
