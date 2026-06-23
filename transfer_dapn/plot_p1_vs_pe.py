"""p1 vs pe trajectory comparison (CW + DAPN/KC → NaSim transfer).

Produces the 4-panel "Policy Trajectory Comparison" figure:
  - Winning trajectory heatmaps (p1 and pe)
  - Action-type distribution + JS divergence
  - Steps to goal

Usage:
  cd transfer_dapn
  conda run -n cyberwheel python plot_p1_vs_pe.py --scenario two_subnet
  conda run -n cyberwheel python plot_p1_vs_pe.py --scenario one_subnet --n 50
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(_REPO / "cyberwheel"))
sys.path.insert(0, str(_DAPN))

from stable_baselines3 import PPO

from envs.scenario_cfg import get_scenario
from envs.nasim_wrapper import NaSimKillChainWrapper
from envs.scenario_load import patch_nasim_load_scenario
from models.encoder import load_encoder, encode_obs

ATYPE_NAMES = [
    "ScanNetwork", "FindServices", "ExploitService", "FindData", "ExfiltrateData",
]
ATYPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
N_ATYPES = len(ATYPE_NAMES)


def flat_to_atype(flat: int, actions_per_host: int = 10) -> int:
    local = flat % actions_per_host
    if local == 2:
        return 0
    if local == 0:
        return 1
    if 4 <= local <= 8:
        return 2
    if local in (1, 3):
        return 3
    if local == 9:
        return 4
    return 0


def collect_kc_trajectories(model, encoder, cfg, n: int, max_steps: int = 200,
                            max_episodes: int = 5000):
    env = NaSimKillChainWrapper(scenario=cfg)
    trajs = []
    ep = 0
    eager = list(cfg.eager_atypes)
    aph = cfg.actions_per_host
    while len(trajs) < n and ep < max_episodes:
        obs, _ = env.reset()
        traj = list(eager)
        for _ in range(max_steps):
            obs44 = np.concatenate([obs, np.zeros(2, dtype=np.float32)])
            enc = encode_obs(encoder, obs44, mask_is_target=True)
            raw_slot, _ = model.predict(enc, deterministic=True)
            raw_slot = min(int(raw_slot), cfg.max_slots)
            flat = env._translate(raw_slot)
            traj.append(flat_to_atype(flat, aph))
            obs, _r, term, trunc, info = env.step(raw_slot)
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


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j - 1], prev[j], dp[j - 1])
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


def plot_results(p1_trajs, pe_trajs, sim_scores, out: str, scenario_label: str):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    max_len = max(max(len(t) for t in p1_trajs), max(len(t) for t in pe_trajs))
    max_len = min(max_len, 30)
    show_n = min(30, len(p1_trajs), len(pe_trajs))

    cmap = matplotlib.colors.ListedColormap(ATYPE_COLORS + ["#eeeeee"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    heatmap_axes = []
    for col, (trajs, title) in enumerate([
        (p1_trajs[:show_n], "p1  (CW→NaSim transfer)"),
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
    p1_dist = atype_distribution(p1_trajs)
    pe_dist = atype_distribution(pe_trajs)
    x, w = np.arange(N_ATYPES), 0.35
    ax_dist.bar(x - w / 2, p1_dist, w, color=ATYPE_COLORS, alpha=0.85, edgecolor="white")
    ax_dist.bar(x + w / 2, pe_dist, w, color=ATYPE_COLORS, alpha=0.45,
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
                 transform=ax_dist.transAxes, ha="right", va="top", fontsize=8)

    ax_steps = fig.add_subplot(gs[1, 1])
    p1_lens = [len(t) for t in p1_trajs]
    pe_lens = [len(t) for t in pe_trajs]
    means = [np.mean(p1_lens), np.mean(pe_lens)]
    stds = [np.std(p1_lens), np.std(pe_lens)]
    bars = ax_steps.bar(["p1\n(CW→NaSim)", "pe\n(NaSim invariant)"],
                        means, color=["#4c72b0", "#dd8452"], alpha=0.85,
                        width=0.45, yerr=stds, capsize=6)
    for bar, mean in zip(bars, means):
        ax_steps.text(bar.get_x() + bar.get_width() / 2, mean + max(stds) * 0.15,
                      f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_steps.set_ylabel("Steps to reach goal", fontsize=8)
    ax_steps.set_title("Steps to Goal", fontsize=10)

    fig.suptitle(f"Policy Trajectory Comparison — {scenario_label}", fontsize=12, y=1.01)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    print(f"Mean sequence similarity: {np.mean(sim_scores):.3f}")
    print(f"JS divergence: {jsd:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="two_subnet", choices=["one_subnet", "two_subnet"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--policy-p1", default=None)
    ap.add_argument("--policy-pe", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = get_scenario(args.scenario)
    patch_nasim_load_scenario(cfg)

    encoder_path = args.encoder or str(_DAPN / f"{cfg.encoder_out}.best.pt")
    p1_path = args.policy_p1 or str(_DAPN / cfg.p1_policy_dir / "best_model.zip")
    pe_path = args.policy_pe or str(_DAPN / cfg.pe_policy_dir / "best_model.zip")
    out_path = args.out or str(_DAPN / cfg.traj_out)

    print(f"Scenario: {cfg.name}  ({cfg.nasim_yaml.name})")
    print("Loading models …")
    encoder = load_encoder(encoder_path, device="cpu")
    p1_model = PPO.load(p1_path)
    pe_model = PPO.load(pe_path)

    print(f"\nCollecting {args.n} p1 winning trajectories …")
    p1_trajs = collect_kc_trajectories(p1_model, encoder, cfg, args.n, args.max_steps)
    if p1_trajs:
        print(f"  mean steps = {np.mean([len(t) for t in p1_trajs]):.1f}")
    else:
        print("  mean steps = n/a (no wins)")

    print(f"\nCollecting {args.n} pe winning trajectories …")
    pe_trajs = collect_kc_trajectories(pe_model, encoder, cfg, args.n, args.max_steps)
    if pe_trajs:
        print(f"  mean steps = {np.mean([len(t) for t in pe_trajs]):.1f}")
    else:
        print("  mean steps = n/a (no wins)")

    if not p1_trajs:
        raise SystemExit(
            "No p1 winning trajectories collected — retrain p1 or check the policy "
            f"({p1_path}). pe collected {len(pe_trajs)} wins."
        )
    if not pe_trajs:
        raise SystemExit("No pe winning trajectories collected — check the pe policy.")

    sim_scores = [normalised_similarity(t1, t2) for t1 in p1_trajs for t2 in pe_trajs]
    plot_results(p1_trajs, pe_trajs, sim_scores, out_path, cfg.name.replace("_", " "))


if __name__ == "__main__":
    main()
