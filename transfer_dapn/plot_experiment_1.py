"""Plot Experiment 1 similarity results.

Reads the JSON written by experiment_1_similarity.py and produces:
  1. Performance bar chart (win rate + mean return for p1 and pe)
  2. Similarity metrics bar chart (action agreement, cosine sim, JS div)
  3. Action frequency comparison (side-by-side bars per action slot)
  4. Episode-level scatter: return_p1 vs. return_pe with regression line

Usage
-----
  cd transfer_dapn
  python plot_experiment_1.py --in artifacts/results/experiment_1_similarity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent
DEFAULT_IN = str(_ROOT / "artifacts/results/experiment_1_similarity.json")

COLOUR_P1 = "#6C8EBF"   # steel blue — p1 (DAPN translated)
COLOUR_PE = "#D4884B"   # warm orange — pe (NaSim native)
COLOUR_SIM = "#6BBF8A"  # green — similarity metrics

plt.rcParams.update({
    "figure.dpi": 180,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "font.family": "DejaVu Sans",
})


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _annotate_bar(ax, x, h, fmt="{:.1f}", offset=6):
    ax.annotate(fmt.format(h), xy=(x, h), xytext=(0, offset),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# Panel 1: Performance bar chart
# ---------------------------------------------------------------------------

def plot_performance(agg: dict, ax_win, ax_ret):
    labels = ["p1\n(CW→NaSim DAPN)", "pe\n(NaSim native)"]
    colours = [COLOUR_P1, COLOUR_PE]
    x = np.arange(2)
    w = 0.55

    win  = [100 * agg["win_rate_p1"],  100 * agg["win_rate_pe"]]
    ci   = [100 * agg["win_rate_p1_ci95"], 100 * agg["win_rate_pe_ci95"]]
    ret  = [agg["mean_return_p1"], agg["mean_return_pe"]]
    ret_e = [agg["std_return_p1"], agg["std_return_pe"]]

    ekw = dict(capsize=5, ecolor="#333", error_kw=dict(lw=1.3, capthick=1.3))

    b = ax_win.bar(x, win, w, yerr=ci, color=colours, edgecolor="white",
                   linewidth=1.2, zorder=3, **ekw)
    ax_win.set_ylabel("Win rate (%)")
    ax_win.set_ylim(0, 120)
    ax_win.set_yticks([0, 20, 40, 60, 80, 100])
    ax_win.set_title("Goal-reach rate")
    ax_win.set_xticks(x)
    ax_win.set_xticklabels(labels, fontsize=10)
    for xi, bi in zip(x, b):
        _annotate_bar(ax_win, xi, bi.get_height(), fmt="{:.0f}%")
    _despine(ax_win)

    b2 = ax_ret.bar(x, ret, w, yerr=ret_e, color=colours, edgecolor="white",
                    linewidth=1.2, zorder=3, **ekw)
    ax_ret.set_ylabel("Mean return")
    ax_ret.set_title("Solution efficiency")
    ax_ret.set_xticks(x)
    ax_ret.set_xticklabels(labels, fontsize=10)
    for xi, bi in zip(x, b2):
        h = bi.get_height()
        if h > 0:
            _annotate_bar(ax_ret, xi, h, fmt="{:.1f}")
    _despine(ax_ret)


# ---------------------------------------------------------------------------
# Panel 2: Similarity metrics
# ---------------------------------------------------------------------------

def plot_similarity_metrics(agg: dict, global_dist: dict, ax):
    metrics = {
        "Action\nagreement": 100 * agg["action_agreement_rate"],
        "Cosine\nsimilarity\n×100": 100 * agg["mean_cosine_sim"],
        "JS div\n×100 (↓)": 100 * agg["mean_js_divergence"],
        "KL div\n×100 (↓)": 100 * min(agg["mean_kl_divergence"], 2.0),
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    colours = [COLOUR_SIM, COLOUR_SIM, "#D47E6C", "#D47E6C"]
    x = np.arange(len(labels))

    b = ax.bar(x, values, 0.55, color=colours, edgecolor="white",
               linewidth=1.2, zorder=3)
    ax.set_title("Behavioural similarity (p1 vs. pe)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Value (%  or  ×100)")
    for xi, bi in zip(x, b):
        _annotate_bar(ax, xi, bi.get_height(), fmt="{:.1f}")
    _despine(ax)


# ---------------------------------------------------------------------------
# Panel 3: Action frequency comparison
# ---------------------------------------------------------------------------

def plot_action_freq(global_dist: dict, ax):
    names   = global_dist["action_names"]
    freq_p1 = np.array(global_dist["freq_p1"])
    freq_pe = np.array(global_dist["freq_pe"])
    x = np.arange(len(names))
    w = 0.35

    b1 = ax.bar(x - w/2, freq_p1, w, label="p1 (DAPN)", color=COLOUR_P1,
                edgecolor="white", linewidth=1.0, zorder=3)
    b2 = ax.bar(x + w/2, freq_pe, w, label="pe (native)", color=COLOUR_PE,
                edgecolor="white", linewidth=1.0, zorder=3)

    ax.set_title("Action frequency per slot")
    ax.set_ylabel("Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=9)
    _despine(ax)


# ---------------------------------------------------------------------------
# Panel 4: Episode return scatter
# ---------------------------------------------------------------------------

def plot_return_scatter(ep_results: list[dict], agg: dict, ax):
    r_p1 = [e["return_p1"] for e in ep_results]
    r_pe  = [e["return_pe"]  for e in ep_results]

    ax.scatter(r_pe, r_p1, alpha=0.35, s=18, color=COLOUR_P1, edgecolors="none")

    # regression line
    if len(r_p1) >= 2:
        m, b = np.polyfit(r_pe, r_p1, 1)
        xs = np.linspace(min(r_pe), max(r_pe), 100)
        ax.plot(xs, m * xs + b, color="#333333", lw=1.2, linestyle="--")

    rho = agg["return_correlation"]
    ax.set_xlabel("Return — pe (NaSim native)")
    ax.set_ylabel("Return — p1 (DAPN)")
    ax.set_title(f"Episode returns  (r = {rho:.3f})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot Experiment 1 similarity results")
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", default=None, help="Output .pdf/.png path")
    ap.add_argument("--title", default="Experiment 1 — CW→NaSim (DAPN) vs. NaSim-native policy")
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)

    agg         = data["aggregate"]
    global_dist = data["global_action_distributions"]
    ep_results  = data.get("episodes", [])

    has_scatter = len(ep_results) >= 2

    ncols = 3 if not has_scatter else 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))

    plot_performance(agg, axes[0, 0], axes[0, 1])
    plot_similarity_metrics(agg, global_dist, axes[0, 2])
    plot_action_freq(global_dist, axes[1, 0])

    if has_scatter:
        plot_return_scatter(ep_results, agg, axes[1, 1])
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "Run without --no-per-ep\nfor scatter plot",
                        ha="center", va="center", transform=axes[1, 1].transAxes,
                        fontsize=9, color="grey")

    # Summary text box in bottom-right
    ax_text = axes[1, 2]
    ax_text.axis("off")
    summary = (
        f"n = {agg['n_episodes']} episodes\n\n"
        f"Win rate\n"
        f"  p1 : {100*agg['win_rate_p1']:.1f}% ± {100*agg['win_rate_p1_ci95']:.1f}%\n"
        f"  pe  : {100*agg['win_rate_pe']:.1f}% ± {100*agg['win_rate_pe_ci95']:.1f}%\n\n"
        f"Similarity\n"
        f"  Agreement : {100*agg['action_agreement_rate']:.1f}%\n"
        f"  Cosine    : {agg['mean_cosine_sim']:.4f}\n"
        f"  JS div    : {agg['mean_js_divergence']:.4f}\n"
        f"  r(ret)    : {agg['return_correlation']:.4f}"
    )
    ax_text.text(0.05, 0.95, summary, transform=ax_text.transAxes,
                 va="top", fontsize=9.5, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                           edgecolor="#cccccc", linewidth=0.8))

    fig.suptitle(args.title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    base = Path(args.inp).with_suffix("")
    out_fig = args.out or str(base.parent / "experiment_1_similarity.pdf")
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    png = str(Path(out_fig).with_suffix(".png"))
    if not out_fig.endswith(".png"):
        fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")
    if not out_fig.endswith(".png"):
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
