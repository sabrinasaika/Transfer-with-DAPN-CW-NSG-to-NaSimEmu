"""Compare winning trajectories: p1 (CW→NaSim transfer) vs pe (NaSim invariant).

Collects N winning trajectories from each policy on the fixed DMZ scenario,
maps every action to one of 5 CW action types, then computes:
  - Mean trajectory length
  - Action-type distribution per policy
  - Normalised edit distance (Levenshtein) between trajectory pairs
  - Jensen-Shannon divergence of the action-type distributions
  - A side-by-side trajectory heatmap

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \\
      transfer_dapn/trajectory_similarity.py --n 50 \\
      --out transfer_dapn/artifacts/results/traj_similarity.png
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

from envs.kill_chain import MAX_SLOTS, SLOT_ORDER
from envs.host_map import discover_host_layout, flat_to_emu_action, EXPLOIT_LOCAL
from envs.nasim_wrapper import NaSimKillChainWrapper
from envs.scenario_load import patch_nasim_load_scenario, load_fixed_dmz_scenario
from models.encoder import load_encoder, encode_obs

_SCENARIO        = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_PE_MODEL        = str(_DAGTS / "wandb/run-20260612_093816-biw5us4u/files/model.pt")
_DEFAULT_ENCODER = str(_DAPN / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
_DEFAULT_P1      = str(_DAPN / "artifacts/policies/cw_dapn_policy/best_model.zip")
_DEFAULT_PE_JSON = str(_DAPN / "artifacts/results/pe_trajectories.json")

# ── Action-type scheme ────────────────────────────────────────────────────────
ATYPE_NAMES = [
    "ScanNetwork",
    "FindServices",
    "ExploitService",
    "FindData",
    "ExfiltrateData",
]
ATYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
N_ATYPES = len(ATYPE_NAMES)

def emu_act_to_atype(action_list, act_id: int) -> int:
    if act_id < 0:
        return 4          # terminal / exfiltrate
    cls, _ = action_list[act_id]
    name = cls.__name__
    if name == "SubnetScan":      return 0
    if name == "ServiceScan":     return 1
    if name == "Exploit":         return 2
    if name in ("OSScan", "ProcessScan"): return 3
    if name == "PrivilegeEscalation":     return 4
    return 0

def flat_to_atype(flat: int) -> int:
    local = flat % 10
    if local == 2:           return 0   # SubnetScan
    if local == 0:           return 1   # ServiceScan
    if 4 <= local <= 8:      return 2   # Exploit
    if local in (1, 3):      return 3   # OSScan / ProcessScan
    if local == 9:           return 4   # PrivEsc
    return 0


# ── Collect p1 winning trajectories via NaSimKillChainWrapper ─────────────────
# Eager setup actions hidden in reset(): exploit entry (proftpd) + subnet scan
_EAGER_ATYPES = [2, 0]  # ExploitService, ScanNetwork

def collect_p1_trajectories(p1_model, encoder, n: int, max_steps: int = 200):
    """Run p1 in NaSimKillChainWrapper; prepend hidden reset() eager actions so
    the trajectory reflects the full attack path, not just the policy steps."""
    env = NaSimKillChainWrapper()
    trajs = []
    ep = 0
    while len(trajs) < n:
        obs, _ = env.reset()
        traj = list(_EAGER_ATYPES)   # include hidden exploit+scan from reset()
        for _ in range(max_steps):
            # NaSimKillChainWrapper gives KC_DIM=42; encode_obs needs 44 (KC+ctx)
            obs44 = np.concatenate([obs, np.zeros(2, dtype=np.float32)])
            enc = encode_obs(encoder, obs44, mask_is_target=True)
            raw_slot, _ = p1_model.predict(enc, deterministic=True)
            raw_slot = min(int(raw_slot), MAX_SLOTS)
            flat = env._translate(raw_slot)
            atype = flat_to_atype(flat)
            traj.append(atype)
            obs, r, term, trunc, info = env.step(raw_slot)
            if term or trunc:
                if info.get("win"):
                    trajs.append(traj)
                break
        ep += 1
        if ep % 20 == 0:
            print(f"  p1: ep={ep}  wins={len(trajs)}/{n}")
    env.close()
    return trajs


# ── Load pe winning trajectories from pre-collected JSON ─────────────────────
def load_pe_trajectories(json_path: str, n: int) -> list:
    """Load pe trajectories from the pre-collected JSON (collect_pe_trajectories.py).
    Using a fixed file ensures the pe panel is identical across all comparison plots."""
    with open(json_path) as f:
        data = json.load(f)
    trajs = data["trajectories"]
    if len(trajs) < n:
        print(f"  WARNING: only {len(trajs)} pe trajectories in {json_path} (requested {n})")
    return trajs[:n]


# ── Similarity metrics ────────────────────────────────────────────────────────
def levenshtein(a: list[int], b: list[int]) -> int:
    """Standard Levenshtein edit distance between two integer sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n]


def normalised_similarity(a: list[int], b: list[int]) -> float:
    """1 - edit_distance / max_length  (1 = identical, 0 = completely different)."""
    d = levenshtein(a, b)
    return 1.0 - d / max(len(a), len(b), 1)


def atype_distribution(trajs: list[list[int]]) -> np.ndarray:
    counts = np.zeros(N_ATYPES)
    for t in trajs:
        for a in t:
            counts[a] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log((a + eps) / (b + eps)))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


# ── Plot ──────────────────────────────────────────────────────────────────────
def _traj_to_matrix(trajs: list[list[int]], max_len: int) -> np.ndarray:
    """(N_traj, max_len) integer matrix; -1 for steps past end."""
    mat = np.full((len(trajs), max_len), -1, dtype=np.int8)
    for i, t in enumerate(trajs):
        for j, a in enumerate(t[:max_len]):
            mat[i, j] = a
    return mat


def plot_results(p1_trajs, pe_trajs, sim_scores, out: str):
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # ── row 0: trajectory heatmaps ────────────────────────────────────────────
    max_len = max(max(len(t) for t in p1_trajs), max(len(t) for t in pe_trajs))
    max_len = min(max_len, 30)
    show_n  = min(30, len(p1_trajs), len(pe_trajs))

    cmap = matplotlib.colors.ListedColormap(ATYPE_COLORS + ["#eeeeee"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    heatmap_axes = []
    for col, (trajs, title) in enumerate(
            [(p1_trajs[:show_n], "p1  (CW→NaSim transfer)"),
             (pe_trajs[:show_n], "pe  (NaSim invariant)")]):
        ax = fig.add_subplot(gs[0, col])
        heatmap_axes.append(ax)
        mat = _traj_to_matrix(trajs, max_len).astype(float)
        mat[mat == -1] = 5        # grey for padding
        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                  interpolation="nearest")
        ax.set_title(f"Winning trajectories — {title}", fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Episode", fontsize=8)
        ax.tick_params(labelsize=7)

    # colour legend inside second heatmap axis
    handles = [
        matplotlib.patches.Patch(color=ATYPE_COLORS[i], label=ATYPE_NAMES[i])
        for i in range(N_ATYPES)
    ]
    handles.append(matplotlib.patches.Patch(color="#eeeeee", label="(done)"))
    heatmap_axes[1].legend(handles=handles, loc="lower right", fontsize=7, frameon=True,
                           title="Action type", title_fontsize=7)

    # ── row 1 left: action-type distribution bars ────────────────────────────
    ax_dist = fig.add_subplot(gs[1, 0])
    p1_dist = atype_distribution(p1_trajs)
    pe_dist = atype_distribution(pe_trajs)
    x = np.arange(N_ATYPES)
    w = 0.35
    ax_dist.bar(x - w/2, p1_dist, w, color=ATYPE_COLORS, alpha=0.85,
                edgecolor="white")
    ax_dist.bar(x + w/2, pe_dist, w, color=ATYPE_COLORS, alpha=0.45,
                edgecolor="white", hatch="//")
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels([n.replace("Action", "") for n in ATYPE_NAMES],
                            rotation=15, ha="right", fontsize=8)
    ax_dist.set_ylabel("Fraction of actions", fontsize=8)
    ax_dist.set_title("Action-type distribution\nover winning trajectories", fontsize=10)
    legend_p1 = matplotlib.patches.Patch(facecolor="grey", edgecolor="white",
                                         alpha=0.85, label="p1")
    legend_pe = matplotlib.patches.Patch(facecolor="grey", edgecolor="white",
                                         alpha=0.45, hatch="//", label="pe")
    ax_dist.legend(handles=[legend_p1, legend_pe], fontsize=8)
    jsd = js_divergence(p1_dist, pe_dist)
    ax_dist.text(0.98, 0.97, f"JS divergence = {jsd:.3f}",
                 transform=ax_dist.transAxes, ha="right", va="top",
                 fontsize=8, color="#333333")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    # ── row 1 right: steps to goal ───────────────────────────────────────────
    ax_steps = fig.add_subplot(gs[1, 1])
    p1_lens = [len(t) for t in p1_trajs]
    pe_lens = [len(t) for t in pe_trajs]
    means = [np.mean(p1_lens), np.mean(pe_lens)]
    stds  = [np.std(p1_lens),  np.std(pe_lens)]
    bar_colors = ["#4c72b0", "#dd8452"]
    bars = ax_steps.bar(["p1\n(CW→NaSim)", "pe\n(NaSim invariant)"],
                        means, color=bar_colors, alpha=0.85,
                        edgecolor="white", width=0.45,
                        yerr=stds, capsize=6, error_kw={"elinewidth": 1.5})
    for bar, mean in zip(bars, means):
        ax_steps.text(bar.get_x() + bar.get_width() / 2, mean + max(stds) * 0.15,
                      f"{mean:.1f}", ha="center", va="bottom", fontsize=10,
                      fontweight="bold")
    ax_steps.set_ylabel("Steps to reach goal", fontsize=8)
    ax_steps.set_title("Steps to Goal", fontsize=10)
    ax_steps.spines["top"].set_visible(False)
    ax_steps.spines["right"].set_visible(False)


    fig.suptitle("Policy Trajectory Comparison — p1 vs pe", fontsize=12, y=1.01)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=50,
                    help="Number of winning trajectories to collect per policy")
    ap.add_argument("--max-steps",  type=int, default=200)
    ap.add_argument("--encoder",    default=str(_DEFAULT_ENCODER))
    ap.add_argument("--policy-p1",  default=str(_DEFAULT_P1))
    ap.add_argument("--pe-json",    default=_DEFAULT_PE_JSON,
                    help="Pre-collected pe trajectories JSON (from collect_pe_trajectories.py)")
    ap.add_argument("--out",        default=str(_DAPN / "artifacts/results/traj_similarity.png"))
    args = ap.parse_args()

    patch_nasim_load_scenario()

    print("Loading p1 + encoder …")
    encoder  = load_encoder(args.encoder, device="cpu")
    p1_model = PPO.load(args.policy_p1)

    print(f"\nCollecting {args.n} winning trajectories from p1 …")
    p1_trajs = collect_p1_trajectories(p1_model, encoder, args.n, args.max_steps)
    print(f"  done: {len(p1_trajs)} trajectories, "
          f"mean length = {np.mean([len(t) for t in p1_trajs]):.1f}")

    print(f"\nLoading pe trajectories from {args.pe_json} …")
    pe_trajs = load_pe_trajectories(args.pe_json, args.n)
    print(f"  done: {len(pe_trajs)} trajectories, "
          f"mean length = {np.mean([len(t) for t in pe_trajs]):.1f}")

    # Pairwise similarity: all p1 vs all pe combinations
    print("\nComputing pairwise trajectory similarities …")
    sim_scores = []
    for t1 in p1_trajs:
        for t2 in pe_trajs:
            sim_scores.append(normalised_similarity(t1, t2))
    print(f"  mean similarity = {np.mean(sim_scores):.3f} ± {np.std(sim_scores):.3f}")

    # Print a sample trajectory from each
    print("\nSample p1 winning trajectory (action types):")
    print("  " + " → ".join(ATYPE_NAMES[a] for a in p1_trajs[0]))
    print("Sample pe winning trajectory (action types):")
    print("  " + " → ".join(ATYPE_NAMES[a] for a in pe_trajs[0]))

    p1_dist = atype_distribution(p1_trajs)
    pe_dist = atype_distribution(pe_trajs)
    print("\nAction-type distributions:")
    print(f"  {'Type':<20} {'p1':>8} {'pe':>8}")
    for i, name in enumerate(ATYPE_NAMES):
        print(f"  {name:<20} {p1_dist[i]:>8.3f} {pe_dist[i]:>8.3f}")
    print(f"\n  JS divergence = {js_divergence(p1_dist, pe_dist):.4f}")

    plot_results(p1_trajs, pe_trajs, sim_scores, args.out)


if __name__ == "__main__":
    main()
