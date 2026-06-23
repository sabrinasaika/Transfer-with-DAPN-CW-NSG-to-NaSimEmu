"""p3 vs pe trajectory comparison (NSG + DAPN/KC → NaSim transfer).

Layout matches trajectory_similarity.py / traj_similarity.png:
  - p3 (left): winning trajectories collected live on NSG-aligned NaSim KC env
  - pe (right): canonical NaSim-invariant reference from pe_trajectories.json
    (graph policy on two_subnet via collect_pe_trajectories.py — same pe panel
    as p1/p2 comparison plots; NOT the NSG KC PPO checkpoint, which collapses
    to the same 4-step path as p3 on this scenario)

Usage:
  cd transfer_dapn
  conda run -n cyberwheel python plot_p3_vs_pe.py --n 50
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

_REPO = Path(__file__).resolve().parents[1]
_DAPN = Path(__file__).resolve().parent

sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_DAPN))

from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
from envs.host_map_nsg import EXPLOIT_LOCAL, flat_action, flat_to_atype
from envs.kc_envs import make_nasim_nsg_kc_env
from envs.kill_chain_nsg import ENTRY_SLOT, MAX_SLOTS, NASIM_NSG_ENTRY
from policy_compat import load_ppo_policy_weights

ATYPE_NAMES = [
    "ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData",
]
ATYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
N_ATYPES = len(ATYPE_NAMES)
_SCAN_SUBNET = 2

_DEFAULT_ENCODER = str(_DAPN / "artifacts/models/dapn_encoder_nsg.pt.best.pt")
_DEFAULT_P3 = str(_DAPN / "artifacts/policies/nsg_dapn_policy_final.zip")
_DEFAULT_PE_JSON = str(_DAPN / "artifacts/results/pe_trajectories.json")
_DEFAULT_OUT = str(_DAPN / "artifacts/results/traj_similarity_p3.png")


def _eager_traj(inner) -> list[int]:
    """Hidden reset actions on entry (http exploit + subnet scan)."""
    ei = inner._layout[ENTRY_SLOT]
    return [
        flat_to_atype(flat_action(ei, EXPLOIT_LOCAL[NASIM_NSG_ENTRY])),
        flat_to_atype(flat_action(ei, _SCAN_SUBNET)),
    ]


def load_pe_trajectories(json_path: str, n: int) -> list[list[int]]:
    """Canonical pe wins (graph policy); shared across p1/p2/p3 comparison plots."""
    with open(json_path) as f:
        data = json.load(f)
    trajs = data["trajectories"]
    if len(trajs) < n:
        print(f"  WARNING: only {len(trajs)} pe trajectories in {json_path} (requested {n})")
    return trajs[:n]


def collect_kc_trajectories(model, encoder_path: str, n: int, max_steps: int = 200,
                            max_episodes: int = 5000, *, stochastic: bool = False):
    base = make_nasim_nsg_kc_env()
    env = DAPNEncoderWrapper(base, encoder_path, mask_is_target=True)
    inner = base.env  # NaSimNSGKillChainWrapper
    trajs = []
    ep = 0
    while len(trajs) < n and ep < max_episodes:
        obs, _ = env.reset()
        traj = _eager_traj(inner)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=not stochastic)
            raw_slot = min(int(action), MAX_SLOTS)
            flat = inner._translate(raw_slot)
            traj.append(flat_to_atype(flat))
            obs, _r, term, trunc, info = env.step(int(action))
            if term or trunc:
                if info.get("win"):
                    trajs.append(traj)
                break
        ep += 1
        if ep % 20 == 0:
            print(f"  ep={ep}  wins={len(trajs)}/{n}")
    env.close()
    if len(trajs) < n:
        print(f"  WARNING: only collected {len(trajs)}/{n} wins in {ep} episodes")
    return trajs


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


def plot_results(p3_trajs, pe_trajs, out: str):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    max_len = max(max(len(t) for t in p3_trajs), max(len(t) for t in pe_trajs))
    max_len = min(max_len, 30)
    show_n = min(30, len(p3_trajs), len(pe_trajs))

    cmap = matplotlib.colors.ListedColormap(ATYPE_COLORS + ["#eeeeee"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    heatmap_axes = []
    for col, (trajs, title) in enumerate([
        (p3_trajs[:show_n], "p3  (NSG→NaSim transfer)"),
        (pe_trajs[:show_n], "pe  (NaSim invariant)"),
    ]):
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

    ax_dist = fig.add_subplot(gs[1, 0])
    p3_dist = atype_distribution(p3_trajs)
    pe_dist = atype_distribution(pe_trajs)
    x, w = np.arange(N_ATYPES), 0.35
    ax_dist.bar(x - w / 2, p3_dist, w, color=ATYPE_COLORS, alpha=0.85, edgecolor="white")
    ax_dist.bar(x + w / 2, pe_dist, w, color=ATYPE_COLORS, alpha=0.45,
                edgecolor="white", hatch="//")
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(ATYPE_NAMES, rotation=15, ha="right", fontsize=8)
    ax_dist.set_ylabel("Fraction of actions", fontsize=8)
    ax_dist.set_title("Action-type distribution\nover winning trajectories", fontsize=10)
    legend_p3 = matplotlib.patches.Patch(facecolor="grey", edgecolor="white",
                                         alpha=0.85, label="p3")
    legend_pe = matplotlib.patches.Patch(facecolor="grey", edgecolor="white",
                                         alpha=0.45, hatch="//", label="pe")
    ax_dist.legend(handles=[legend_p3, legend_pe], fontsize=8, loc="upper left")
    jsd = js_divergence(p3_dist, pe_dist)
    ax_dist.text(0.98, 0.97, f"JS divergence = {jsd:.3f}",
                 transform=ax_dist.transAxes, ha="right", va="top", fontsize=8)

    ax_steps = fig.add_subplot(gs[1, 1])
    p3_lens = [len(t) for t in p3_trajs]
    pe_lens = [len(t) for t in pe_trajs]
    means = [np.mean(p3_lens), np.mean(pe_lens)]
    stds = [np.std(p3_lens), np.std(pe_lens)]
    bars = ax_steps.bar(["p3\n(NSG→NaSim)", "pe\n(NaSim invariant)"],
                        means, color=["#2ca02c", "#dd8452"], alpha=0.85,
                        width=0.45, yerr=stds, capsize=6)
    for bar, mean in zip(bars, means):
        ax_steps.text(bar.get_x() + bar.get_width() / 2, mean + max(stds) * 0.15,
                      f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_steps.set_ylabel("Steps to reach goal", fontsize=8)
    ax_steps.set_title("Steps to Goal", fontsize=10)

    fig.suptitle("Policy Trajectory Comparison — p3 vs pe", fontsize=12, y=1.01)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    print(f"JS divergence: {jsd:.4f}")


def main():
    ap = argparse.ArgumentParser(description="p3 vs pe trajectory comparison (NSG)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--encoder", default=_DEFAULT_ENCODER)
    ap.add_argument("--policy-p3", default=_DEFAULT_P3)
    ap.add_argument("--pe-json", default=_DEFAULT_PE_JSON,
                    help="Canonical pe trajectories (collect_pe_trajectories.py)")
    ap.add_argument("--out", default=_DEFAULT_OUT)
    args = ap.parse_args()

    print("Scenario: NSG-aligned NaSim (nasim_two_subnet)")
    print("Loading models …")

    dummy_env = DAPNEncoderWrapper(
        make_nasim_nsg_kc_env(), args.encoder, mask_is_target=True)
    p3_model = load_ppo_policy_weights(args.policy_p3, dummy_env)
    dummy_env.close()

    print(f"\nCollecting {args.n} p3 winning trajectories …")
    p3_trajs = collect_kc_trajectories(
        p3_model, args.encoder, args.n, args.max_steps, stochastic=False)
    if p3_trajs:
        print(f"  mean steps = {np.mean([len(t) for t in p3_trajs]):.1f}")
    else:
        print("  mean steps = n/a (no wins)")

    print(f"\nLoading pe reference trajectories from {args.pe_json} …")
    pe_trajs = load_pe_trajectories(args.pe_json, args.n)
    if pe_trajs:
        print(f"  mean steps = {np.mean([len(t) for t in pe_trajs]):.1f}")
    else:
        print("  mean steps = n/a (empty JSON)")

    if not p3_trajs:
        raise SystemExit(
            f"No p3 winning trajectories — retrain p3 or check {args.policy_p3}."
        )
    if not pe_trajs:
        raise SystemExit(
            f"No pe trajectories in {args.pe_json} — run collect_pe_trajectories.py."
        )

    plot_results(p3_trajs, pe_trajs, args.out)


if __name__ == "__main__":
    main()
