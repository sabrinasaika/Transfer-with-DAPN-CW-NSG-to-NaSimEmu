"""Generate NASimEmu-style training curves from offline JSON data or W&B.

Produces a figure matching the paper:
  - Left panel : training scenario reward-per-step vs epoch
  - Right panel: novel scenario reward-per-step vs epoch  (optional)
  - Each policy drawn as a solid line with a shaded ±1 std band
  - Black = p1 (CW→NaSim transferred), Blue = pe (NaSim invariant)

Single-run usage (no std band):
  python plot_training_curves.py --pe artifacts/results/training_curves_pe.json

Multi-seed usage (shaded bands — run training with different --seed values first):
  python plot_training_curves.py \\
      --pe  artifacts/results/training_curves_pe_s42.json \\
            artifacts/results/training_curves_pe_s0.json  \\
            artifacts/results/training_curves_pe_s7.json  \\
      --p1  artifacts/results/training_curves_p1_s42.json \\
      --out artifacts/results/training_curves.pdf

W&B usage:
  python plot_training_curves.py --wandb-project nasimemu-exp1 \\
      --wandb-runs pe-nasim-invariant p1-cw-transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = {
    "pe": "#1f77b4",   # blue  — NaSim invariant
    "p1": "#000000",   # black — CW transfer
    "p2": "#d62728",   # red
    "p3": "#2ca02c",   # green
}
_LABELS = {
    "pe": "NaSim Invariant (pe)",
    "p1": "CW→NaSim Transfer (p1)",
    "p2": "CBS→NaSim Transfer (p2)",
    "p3": "NSG→NaSim Transfer (p3)",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _curves_from_files(paths: list[str]) -> dict:
    """Return {epoch: [rps, ...]} averaged across runs."""
    if not paths:
        return {}
    runs = [_load_json(p) for p in paths]
    label = runs[0].get("label", Path(paths[0]).stem)
    by_epoch: dict[int, list[float]] = {}
    steps_map: dict[int, int] = {}
    for run in runs:
        for rec in run.get("epochs", []):
            e = rec["epoch"]
            by_epoch.setdefault(e, []).append(rec["reward_per_step"])
            if e not in steps_map:
                steps_map[e] = rec.get("timestep", e)
    epochs = sorted(by_epoch)
    mean   = [float(np.mean(by_epoch[e])) for e in epochs]
    std    = [float(np.std(by_epoch[e]))  for e in epochs]
    steps  = [steps_map[e] for e in epochs]
    return {"label": label, "epochs": epochs, "steps": steps, "mean": mean, "std": std}


def _curves_from_wandb(project: str, run_names: list[str]) -> list[dict]:
    import wandb
    api   = wandb.Api()
    out   = []
    for name in run_names:
        runs = api.runs(project, filters={"display_name": name})
        by_epoch: dict[int, list[float]] = {}
        label = name.split("-")[0]   # "pe", "p1", …
        for run in runs:
            hist = run.history(keys=[f"{label}/reward_per_step", "epoch"])
            for _, row in hist.iterrows():
                e = int(row.get("epoch", 0))
                v = row.get(f"{label}/reward_per_step", np.nan)
                if not np.isnan(v):
                    by_epoch.setdefault(e, []).append(float(v))
        if not by_epoch:
            continue
        epochs = sorted(by_epoch)
        out.append({
            "label":  label,
            "epochs": epochs,
            "mean":   [float(np.mean(by_epoch[e])) for e in epochs],
            "std":    [float(np.std(by_epoch[e]))  for e in epochs],
        })
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def _smooth(arr: list[float], k: int = 5) -> list[float]:
    if k <= 1 or len(arr) <= k:
        return arr
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return list(np.convolve(padded, kernel, mode="valid")[:len(arr)])


def _normalize(arr: list[float]) -> list[float]:
    """Scale to [0, 1] using curve's own min/max."""
    lo, hi = min(arr), max(arr)
    if hi == lo:
        return [0.0] * len(arr)
    return [(v - lo) / (hi - lo) for v in arr]


def _plot_panel(ax, curves: list[dict], title: str, smooth: int = 5,
                normalize: bool = False, use_steps: bool = True):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Env steps" if use_steps else "Epoch", fontsize=9)
    ylabel = "Normalized reward" if normalize else "Reward per step"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)

    for c in curves:
        lbl   = c["label"]
        color = _COLORS.get(lbl, "#888888")
        name  = _LABELS.get(lbl, lbl)
        xs    = np.array(c.get("steps", c["epochs"])) if use_steps else np.array(c["epochs"])
        mean  = _smooth(c["mean"], smooth)
        ys    = np.array(_normalize(mean) if normalize else mean)
        stds  = np.array(c["std"])

        ax.plot(xs, ys, color=color, linewidth=1.8, label=name)
        if stds.sum() > 0 and not normalize:
            ax.fill_between(xs, ys - stds, ys + stds,
                            alpha=0.15, color=color, linewidth=0)

    ax.legend(fontsize=8, loc="lower right")


def build_figure(curves_train: list[dict],
                 curves_novel: Optional[list[dict]] = None,
                 smooth: int = 5,
                 normalize: bool = False,
                 use_steps: bool = True) -> plt.Figure:
    n_panels = 2 if curves_novel else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    _plot_panel(axes[0], curves_train, "Training scenario", smooth, normalize, use_steps)
    if curves_novel:
        _plot_panel(axes[1], curves_novel, "Novel scenario", smooth, normalize, use_steps)

    fig.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Plot training curves")
    ap.add_argument("--pe",  nargs="*", default=[],
                    help="JSON curve files for pe (NaSim invariant)")
    ap.add_argument("--p1",  nargs="*", default=[],
                    help="JSON curve files for p1 (CW transfer)")
    ap.add_argument("--p2",  nargs="*", default=[],
                    help="JSON curve files for p2")
    ap.add_argument("--p3",  nargs="*", default=[],
                    help="JSON curve files for p3")
    ap.add_argument("--novel-pe", nargs="*", default=[],
                    help="Novel-scenario curve files for pe")
    ap.add_argument("--novel-p1", nargs="*", default=[],
                    help="Novel-scenario curve files for p1")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-runs",    nargs="*", default=[])
    ap.add_argument("--smooth", type=int, default=5,
                    help="Rolling-average window (epochs). Set 1 to disable.")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize each curve to [0,1] (needed when rewards differ in scale)")
    ap.add_argument("--epochs", action="store_true",
                    help="Use epochs on x-axis instead of env steps")
    ap.add_argument("--out",  default="artifacts/results/training_curves.png")
    args = ap.parse_args()

    curves_train: list[dict] = []
    curves_novel: list[dict] = []

    if args.wandb_project and args.wandb_runs:
        curves_train = _curves_from_wandb(args.wandb_project, args.wandb_runs)
    else:
        for key, paths in [("pe", args.pe), ("p1", args.p1),
                            ("p2", args.p2), ("p3", args.p3)]:
            if paths:
                c = _curves_from_files(paths)
                c["label"] = key
                curves_train.append(c)
        for key, paths in [("pe", args.novel_pe), ("p1", args.novel_p1)]:
            if paths:
                c = _curves_from_files(paths)
                c["label"] = key
                curves_novel.append(c)

    if not curves_train:
        ap.error("Provide at least one --pe / --p1 file (or --wandb-project).")

    fig = build_figure(curves_train,
                       curves_novel if curves_novel else None,
                       smooth=args.smooth,
                       normalize=args.normalize,
                       use_steps=not args.epochs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
