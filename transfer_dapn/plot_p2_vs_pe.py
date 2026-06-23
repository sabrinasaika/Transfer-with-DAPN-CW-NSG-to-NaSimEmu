"""Compare p2 (raw CW → NaSim transfer, 0% win rate) vs pe (NaSim invariant).

Since p2 wins 0 episodes, we show its episode trajectories (all failed) alongside
pe's winning trajectories to illustrate WHY naive transfer fails.

4-panel layout:
  Top-left : p2 episode action heatmap (all failed attempts)
  Top-right: pe winning trajectories heatmap (from pre-collected JSON)
  Bot-left : action-type distribution (p2 episodes vs pe wins)
  Bot-right: win rate bar chart (p1=100%, p2=0%, pe=100%)

Usage (nasimemu-env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  /home/ssaika@cs.utep.edu/nasimemu-env/bin/python \\
      transfer_dapn/plot_p2_vs_pe.py \\
      --out transfer_dapn/artifacts/results/p2_vs_pe.png
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

sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_DAPN))

from stable_baselines3 import PPO
from nasimemu.env import NASimEmuEnv
from envs.scenario_load import patch_nasim_load_scenario
from eval_p2_nasim import nasim_matrix_to_cw_obs, cw_action_to_nasim, _local_to_atype

_SCENARIO    = str(_REPO / "scenarios" / "fixed_dmz_two_subnet.v2.yaml")
_DEFAULT_P2  = str(_DAPN / "artifacts/policies/cw_raw_policy/best_model.zip")
_DEFAULT_PE_JSON = str(_DAPN / "artifacts/results/pe_trajectories.json")
_DEFAULT_OUT = str(_DAPN / "artifacts/results/p2_vs_pe.png")

ATYPE_NAMES  = ["ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData"]
ATYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
N_ATYPES = 5


def collect_p2_episodes(p2_model, n: int = 30, max_steps: int = 50):
    """Collect p2 episode trajectories in NaSim (win or lose, fixed length)."""
    env = NASimEmuEnv(scenario_name=_SCENARIO, fully_obs=False)
    episodes = []
    for ep in range(n):
        env._generate_env()
        s = env.reset()
        traj = []
        for _ in range(max_steps):
            cw_obs = nasim_matrix_to_cw_obs(s)
            raw_action, _ = p2_model.predict(cw_obs, deterministic=True)
            nasim_action = cw_action_to_nasim(int(raw_action), env.action_list)
            _, act_id = nasim_action
            local = act_id % 10
            traj.append(_local_to_atype(local))
            s, _, _, _ = env.step(nasim_action)
        episodes.append(traj)
    env.close()
    return episodes


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


def plot_results(p2_episodes, pe_trajs, out: str):
    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    show_n  = min(30, len(p2_episodes), len(pe_trajs))
    max_len = max(
        max(len(t) for t in p2_episodes[:show_n]),
        max(len(t) for t in pe_trajs[:show_n]),
    )
    max_len = min(max_len, 50)

    cmap   = matplotlib.colors.ListedColormap(ATYPE_COLORS + ["#eeeeee"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Top-left: p2 episodes (all failed)
    ax0 = fig.add_subplot(gs[0, 0])
    mat0 = _traj_to_matrix(p2_episodes[:show_n], max_len).astype(float)
    mat0[mat0 == -1] = 5
    ax0.imshow(mat0, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax0.set_title("p2 episodes  (original CW → NaSim)  —  0% win rate", fontsize=10)
    ax0.set_xlabel("Step", fontsize=8)
    ax0.set_ylabel("Episode", fontsize=8)
    ax0.tick_params(labelsize=7)
    ax0.text(0.98, 0.02, "No wins in 500 episodes",
             transform=ax0.transAxes, ha="right", va="bottom",
             fontsize=8, color="#cc0000", fontweight="bold")

    # Top-right: pe winning trajectories
    ax1 = fig.add_subplot(gs[0, 1])
    mat1 = _traj_to_matrix(pe_trajs[:show_n], max_len).astype(float)
    mat1[mat1 == -1] = 5
    ax1.imshow(mat1, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax1.set_title("pe winning trajectories  (NaSim invariant)", fontsize=10)
    ax1.set_xlabel("Step", fontsize=8)
    ax1.set_ylabel("Episode", fontsize=8)
    ax1.tick_params(labelsize=7)

    handles = [matplotlib.patches.Patch(color=ATYPE_COLORS[i], label=ATYPE_NAMES[i])
               for i in range(N_ATYPES)]
    handles.append(matplotlib.patches.Patch(color="#eeeeee", label="(done)"))
    ax1.legend(handles=handles, loc="lower right", fontsize=7, frameon=True,
               title="Action type", title_fontsize=7)

    # Bottom-left: action-type distribution
    ax_dist = fig.add_subplot(gs[1, 0])
    p2_dist = atype_distribution(p2_episodes)
    pe_dist = atype_distribution(pe_trajs)
    x, w = np.arange(N_ATYPES), 0.35
    ax_dist.bar(x - w/2, p2_dist, w, color=ATYPE_COLORS, alpha=0.85, edgecolor="white")
    ax_dist.bar(x + w/2, pe_dist, w, color=ATYPE_COLORS, alpha=0.45,
                edgecolor="white", hatch="//")
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels([n.replace("Action", "") for n in ATYPE_NAMES],
                            rotation=15, ha="right", fontsize=8)
    ax_dist.set_ylabel("Fraction of actions", fontsize=8)
    ax_dist.set_title("Action-type distribution\np2 (original CW) episodes vs pe wins", fontsize=10)
    lp2 = matplotlib.patches.Patch(facecolor="grey", alpha=0.85, edgecolor="white", label="p2")
    lpe = matplotlib.patches.Patch(facecolor="grey", alpha=0.45, edgecolor="white",
                                   hatch="//", label="pe")
    ax_dist.legend(handles=[lp2, lpe], fontsize=8)
    jsd = js_divergence(p2_dist, pe_dist)
    ax_dist.text(0.98, 0.97, f"JS divergence = {jsd:.3f}",
                 transform=ax_dist.transAxes, ha="right", va="top", fontsize=8)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    # Bottom-right: win rate comparison (p1, p2, pe)
    ax_wr = fig.add_subplot(gs[1, 1])
    labels    = ["p2\n(original CW)", "pe\n(NaSim)"]
    win_rates = [0.0, 100.0]
    bar_colors = ["#e05c5c", "#dd8452"]
    bars = ax_wr.bar(labels, win_rates, color=bar_colors, alpha=0.85,
                     edgecolor="white", width=0.45)
    for bar, val in zip(bars, win_rates):
        ax_wr.text(bar.get_x() + bar.get_width() / 2,
                   val + 1.5,
                   f"{val:.0f}%", ha="center", va="bottom",
                   fontsize=10, fontweight="bold")
    ax_wr.set_ylim(0, 115)
    ax_wr.set_ylabel("Win rate (%)", fontsize=8)
    ax_wr.set_title("Win Rate in NaSim", fontsize=10)
    ax_wr.spines["top"].set_visible(False)
    ax_wr.spines["right"].set_visible(False)

    fig.suptitle("Policy Trajectory Comparison — p2 (original CW) vs pe",
                 fontsize=12, y=1.01)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    print(f"JS divergence (p2 episodes vs pe wins): {jsd:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=30,
                    help="Number of p2 episodes and pe trajectories to show")
    ap.add_argument("--max-steps",  type=int, default=50)
    ap.add_argument("--policy-p2",  default=_DEFAULT_P2)
    ap.add_argument("--pe-json",    default=_DEFAULT_PE_JSON)
    ap.add_argument("--out",        default=_DEFAULT_OUT)
    args = ap.parse_args()

    patch_nasim_load_scenario()

    print(f"Loading p2 from {args.policy_p2} …")
    p2_model = PPO.load(args.policy_p2)

    print(f"\nCollecting {args.n} p2 episodes in NaSim …")
    p2_episodes = collect_p2_episodes(p2_model, n=args.n, max_steps=args.max_steps)
    p2_dist = [0]*5
    for t in p2_episodes:
        for a in t: p2_dist[a] += 1
    total = sum(p2_dist)
    print("p2 action distribution:")
    for i, name in enumerate(ATYPE_NAMES):
        print(f"  {name:<20} {p2_dist[i]/total:.3f}")

    print(f"\nLoading pe trajectories from {args.pe_json} …")
    with open(args.pe_json) as f:
        data = json.load(f)
    pe_trajs = data["trajectories"][:args.n]
    print(f"  {len(pe_trajs)} trajectories, mean steps = {data['mean_steps']:.1f}")

    plot_results(p2_episodes, pe_trajs, args.out)


if __name__ == "__main__":
    main()
