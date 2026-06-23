"""Plot CW (native training env) vs pe (NaSim invariant) trajectory comparison.

Loads pre-collected CW trajectories from cw_trajectories.json and
collects pe trajectories live, then produces the same 4-panel figure
as trajectory_similarity.py.

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \\
      transfer_dapn/plot_cw_vs_pe.py \\
      --out transfer_dapn/artifacts/results/cw_vs_pe.png
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

_SCENARIO = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_PE_MODEL = str(_DAGTS / "wandb/run-20260612_093816-biw5us4u/files/model.pt")
_CW_JSON  = str(_DAPN / "artifacts/results/cw_trajectories.json")

ATYPE_NAMES  = ["ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData"]
ATYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
N_ATYPES = len(ATYPE_NAMES)


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


def collect_pe_trajectories(pe_net, n: int, max_steps: int = 200):
    pe_net.set_force_continue(True)
    env = NASimEmuEnv(scenario_name=_SCENARIO, fully_obs=False)
    trajs = []
    ep = 0
    while len(trajs) < n:
        env._generate_env()
        s = env.reset()
        traj = []
        won = False
        for _ in range(max_steps):
            with torch.no_grad():
                acts, _, _, _ = pe_net.forward([s])
            _, act_id = acts[0]
            traj.append(emu_act_to_atype(env.action_list, int(act_id)))
            s, _, _, _ = env.step(acts[0])
            if env.env.goal_reached(env.env.current_state):
                won = True
                break
        if won:
            trajs.append(traj)
        ep += 1
        if ep % 20 == 0:
            print(f"  pe: ep={ep}  wins={len(trajs)}/{n}")
    env.close()
    return trajs


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n]


def normalised_similarity(a, b):
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


def atype_distribution(trajs):
    counts = np.zeros(N_ATYPES)
    for t in trajs:
        for a in t:
            counts[a] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def js_divergence(p, q):
    eps = 1e-10
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log((a + eps) / (b + eps)))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _traj_to_matrix(trajs, max_len):
    mat = np.full((len(trajs), max_len), -1, dtype=np.int8)
    for i, t in enumerate(trajs):
        for j, a in enumerate(t[:max_len]):
            mat[i, j] = a
    return mat


def plot(cw_trajs, pe_trajs, sim_scores, out):
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    max_len = max(max(len(t) for t in cw_trajs), max(len(t) for t in pe_trajs))
    max_len = min(max_len, 30)
    show_n  = min(30, len(cw_trajs), len(pe_trajs))

    cmap   = matplotlib.colors.ListedColormap(ATYPE_COLORS + ["#eeeeee"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    heatmap_axes = []
    for col, (trajs, title) in enumerate([
            (cw_trajs[:show_n], "p1  (CyberWheel — training env)"),
            (pe_trajs[:show_n], "pe  (NaSim invariant)")]):
        ax = fig.add_subplot(gs[0, col])
        heatmap_axes.append(ax)
        mat = _traj_to_matrix(trajs, max_len).astype(float)
        mat[mat == -1] = 5
        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"Winning trajectories — {title}", fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Episode", fontsize=8)
        ax.tick_params(labelsize=7)

    handles = [matplotlib.patches.Patch(color=ATYPE_COLORS[i], label=ATYPE_NAMES[i])
               for i in range(N_ATYPES)]
    handles.append(matplotlib.patches.Patch(color="#eeeeee", label="(done)"))
    heatmap_axes[1].legend(handles=handles, loc="lower right", fontsize=7,
                           frameon=True, title="Action type", title_fontsize=7)

    # ── distribution ─────────────────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[1, 0])
    cw_dist = atype_distribution(cw_trajs)
    pe_dist = atype_distribution(pe_trajs)
    x, w = np.arange(N_ATYPES), 0.35
    ax_dist.bar(x - w/2, cw_dist, w, color=ATYPE_COLORS, alpha=0.85, edgecolor="white")
    ax_dist.bar(x + w/2, pe_dist, w, color=ATYPE_COLORS, alpha=0.45,
                edgecolor="white", hatch="//")
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels([n.replace("Action", "") for n in ATYPE_NAMES],
                            rotation=15, ha="right", fontsize=8)
    ax_dist.set_ylabel("Fraction of actions", fontsize=8)
    ax_dist.set_title("Action-type distribution\nover winning trajectories", fontsize=10)
    legend_cw = matplotlib.patches.Patch(facecolor="grey", edgecolor="white",
                                         alpha=0.85, label="p1 (CW)")
    legend_pe = matplotlib.patches.Patch(facecolor="grey", edgecolor="white",
                                         alpha=0.45, hatch="//", label="pe (NaSim)")
    ax_dist.legend(handles=[legend_cw, legend_pe], fontsize=8)
    jsd = js_divergence(cw_dist, pe_dist)
    ax_dist.text(0.98, 0.97, f"JS divergence = {jsd:.3f}",
                 transform=ax_dist.transAxes, ha="right", va="top",
                 fontsize=8, color="#333333")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    # ── steps to goal ─────────────────────────────────────────────────────────
    ax_steps = fig.add_subplot(gs[1, 1])
    cw_lens = [len(t) for t in cw_trajs]
    pe_lens = [len(t) for t in pe_trajs]
    means = [np.mean(cw_lens), np.mean(pe_lens)]
    stds  = [np.std(cw_lens),  np.std(pe_lens)]
    bar_colors = ["#4c72b0", "#dd8452"]
    bars = ax_steps.bar(["p1\n(CyberWheel)", "pe\n(NaSim invariant)"],
                        means, color=bar_colors, alpha=0.85, edgecolor="white",
                        width=0.45, yerr=stds, capsize=6,
                        error_kw={"elinewidth": 1.5})
    for bar, mean in zip(bars, means):
        ax_steps.text(bar.get_x() + bar.get_width() / 2,
                      mean + max(stds) * 0.15,
                      f"{mean:.1f}", ha="center", va="bottom",
                      fontsize=10, fontweight="bold")
    ax_steps.set_ylabel("Steps to reach goal", fontsize=8)
    ax_steps.set_title("Steps to Goal", fontsize=10)
    ax_steps.spines["top"].set_visible(False)
    ax_steps.spines["right"].set_visible(False)

    fig.suptitle("Policy Trajectory Comparison", fontsize=12, y=1.01)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")

    mean_sim = float(np.mean(sim_scores))
    print(f"Mean sequence similarity (CW vs pe): {mean_sim:.3f}")
    print(f"JS divergence: {jsd:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cw-json",   default=_CW_JSON)
    ap.add_argument("--n-pe",      type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", default=str(_DAPN / "artifacts/results/cw_vs_pe.png"))
    args = ap.parse_args()

    patch_nasim_load_scenario()

    print("Loading CW trajectories …")
    with open(args.cw_json) as f:
        cw_data = json.load(f)
    cw_trajs = cw_data["trajectories"]
    print(f"  {len(cw_trajs)} trajectories, mean steps = {cw_data['mean_steps']:.1f}")

    print("Loading pe …")
    pe_net = NASimNetInvMAct()
    pe_net.load_state_dict(torch.load(_PE_MODEL, map_location="cpu"))
    pe_net.eval()

    print(f"\nCollecting {args.n_pe} pe winning trajectories …")
    pe_trajs = collect_pe_trajectories(pe_net, args.n_pe, args.max_steps)
    print(f"  done: mean steps = {np.mean([len(t) for t in pe_trajs]):.1f}")

    print("\nComputing pairwise similarities …")
    sim_scores = [normalised_similarity(t1, t2)
                  for t1 in cw_trajs for t2 in pe_trajs]

    plot(cw_trajs, pe_trajs, sim_scores, args.out)


if __name__ == "__main__":
    main()
