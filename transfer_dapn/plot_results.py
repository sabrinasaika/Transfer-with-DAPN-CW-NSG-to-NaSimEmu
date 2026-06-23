"""Plot the DAPN transfer benchmark headline figure + emit a results table.

Reads the JSON written by benchmark.py and produces:
  - a grouped bar chart (win rate %, with 95% CI error bars; mean return annotated)
  - a Markdown table  (stdout + .md file)
  - a LaTeX table     (.tex file, ready to \input{} in a paper)

Usage
-----
  cd transfer_dapn
  python plot_results.py --in artifacts/results/benchmark.json
  python plot_results.py --in artifacts/results/benchmark.json --out artifacts/results/transfer_bar.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401
import numpy as np

from bench_common import load_json

_ROOT = Path(__file__).resolve().parent
DEFAULT_IN = str(_ROOT / "artifacts/results/benchmark.json")

# consistent ordering + colours for the paper figure (modern, slightly muted)
ORDER = ["nasim", "dapn_sim", "dapn_emu"]
COLOURS = {
    "nasim": "#6C8EBF",      # steel blue   - reference (no translator)
    "dapn_sim": "#6BBF8A",   # green        - DAPN in simulation
    "dapn_emu": "#2E8B57",   # deep green   - DAPN on live emulator
}

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.family": "DejaVu Sans",
})


def _ordered(results):
    by_cond = {r["condition"]: r for r in results}
    out = [by_cond[c] for c in ORDER if c in by_cond]
    # append any unknown conditions at the end
    out += [r for r in results if r["condition"] not in ORDER]
    return out


def _style_axis(ax):
    """Despine + light horizontal grid for a clean modern look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def make_bar(results, out_path: str, title: str):
    results = _ordered(results)
    labels = [r.get("label", r["condition"]) for r in results]
    colours = [COLOURS.get(r["condition"], "#777777") for r in results]
    x = np.arange(len(results))
    bar_w = 0.62

    win = np.array([100 * r["win_rate"] for r in results])
    win_err = np.array([100 * r.get("win_rate_ci95", 0.0) for r in results])
    ret = np.array([r["mean_return"] for r in results])
    ret_err = np.array([r.get("std_return", 0.0) for r in results])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(9.5, 3.1 * len(results)), 5.0))
    ekw = dict(capsize=5, ecolor="#333333",
               error_kw=dict(lw=1.4, capthick=1.4))
    # CI error bars are only informative when n is reasonably large; with very
    # few episodes (e.g. the slow emulator run) the interval is huge and just
    # visual noise, so suppress it and mark small-n bars instead.
    small_n = np.array([r["episodes"] < 10 for r in results])
    win_err = np.where(small_n, 0.0, win_err)

    # --- panel 1: win rate ---
    b1 = ax1.bar(x, win, bar_w, yerr=win_err, color=colours,
                 edgecolor="white", linewidth=1.2, zorder=3, **ekw)
    ax1.set_ylabel("Win rate (%)")
    ax1.set_ylim(0, 116)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title("Goal-reach rate")
    for xi, b in zip(x, b1):
        h = b.get_height()
        ax1.annotate(f"{h:.0f}%", xy=(xi, h), xytext=(0, 7),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

    # --- panel 2: mean return (efficiency) ---
    ret_top = max(100.0, float(ret.max()) + float(ret_err.max()) + 8)
    b2 = ax2.bar(x, ret, bar_w, yerr=ret_err, color=colours,
                 edgecolor="white", linewidth=1.2, zorder=3, **ekw)
    ax2.set_ylabel("Mean return")
    ax2.set_ylim(0, ret_top)
    ax2.set_title("Solution efficiency")
    for xi, r, b in zip(x, results, b2):
        h = b.get_height()
        ax2.annotate(f"{h:.1f}", xy=(xi, h), xytext=(0, 7),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
        ax2.annotate(f"n={r['episodes']}", xy=(xi, 0), xytext=(0, 6),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8, color="white", fontweight="bold")

    # wrap "DAPN -> Emulator" etc. onto two lines so labels don't collide
    wrapped = [lbl.replace(" -> ", "\n→ ") for lbl in labels]
    for ax in (ax1, ax2):
        _style_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped, fontsize=10.5)
        ax.set_xlim(-0.7, len(results) - 0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    png = str(Path(out_path).with_suffix(".png"))
    if not out_path.endswith(".png"):
        fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    if not out_path.endswith(".png"):
        print(f"  wrote {png}")
    return png if not out_path.endswith(".png") else out_path


def make_markdown(results) -> str:
    results = _ordered(results)
    lines = [
        "| Condition | Domain | Win rate | 95% CI | Mean return | Episodes |",
        "|-----------|--------|---------:|-------:|------------:|---------:|",
    ]
    for r in results:
        label = r.get("label", r["condition"]).replace("\n", " ")
        lines.append(
            f"| {label} | {r.get('note','')} | "
            f"{100*r['win_rate']:.1f}% | "
            f"±{100*r.get('win_rate_ci95',0.0):.1f}% | "
            f"{r['mean_return']:.2f} ± {r.get('std_return',0.0):.2f} | "
            f"{r['episodes']} |"
        )
    return "\n".join(lines)


def make_latex(results) -> str:
    results = _ordered(results)
    rows = []
    for r in results:
        label = r.get("label", r["condition"]).replace("\n", " ")
        rows.append(
            f"{label} & {100*r['win_rate']:.1f}\\% & "
            f"$\\pm${100*r.get('win_rate_ci95',0.0):.1f}\\% & "
            f"{r['mean_return']:.2f} $\\pm$ {r.get('std_return',0.0):.2f} & "
            f"{r['episodes']} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{DAPN cross-domain transfer. A CyberWheel-trained policy is "
        "transferred zero-shot via the domain translator $G$.}\n"
        "\\label{tab:dapn-transfer}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Condition & Win rate & 95\\% CI & Mean return & Episodes \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _as_list(payload):
    if isinstance(payload, dict):
        return [payload]
    return list(payload)


def merge_results(*paths):
    """Combine several result JSONs; later files override earlier by condition."""
    by_cond = {}
    order = []
    for p in paths:
        for rec in _as_list(load_json(p)):
            c = rec["condition"]
            if c not in by_cond:
                order.append(c)
            by_cond[c] = rec
    return [by_cond[c] for c in order]


def main():
    ap = argparse.ArgumentParser(description="Plot DAPN transfer benchmark")
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--merge", nargs="*", default=None,
                    help="Extra result JSON(s) to merge in (e.g. emulator runs). "
                         "Merged result is written back to --in.")
    ap.add_argument("--out", default=None, help="Figure path (.pdf/.png). Default next to JSON.")
    ap.add_argument("--title", default="DAPN zero-shot cross-domain transfer")
    args = ap.parse_args()

    if args.merge:
        results = merge_results(args.inp, *args.merge)
        from bench_common import write_json as _wj
        _wj(args.inp, results)
    else:
        results = load_json(args.inp)
        if isinstance(results, dict):
            results = [results]
    if not results:
        print("No results in JSON; nothing to plot.")
        return

    base = Path(args.inp).with_suffix("")
    out_fig = args.out or str(base.parent / "transfer_bar.pdf")
    make_bar(results, out_fig, args.title)

    md = make_markdown(results)
    tex = make_latex(results)

    md_path = str(base.parent / "transfer_table.md")
    tex_path = str(base.parent / "transfer_table.tex")
    Path(md_path).write_text(md + "\n")
    Path(tex_path).write_text(tex)
    print(f"  wrote {md_path}")
    print(f"  wrote {tex_path}")

    print("\n" + md + "\n")


if __name__ == "__main__":
    main()
