#!/usr/bin/env bash
# p2 — Raw CyberWheel (no KC, no DAPN) → NaSim naive-transfer baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TD="$ROOT/transfer_dapn"
CW_PY="$HOME/.conda/envs/cyberwheel/bin/python"
export PYTHONPATH="$TD:$ROOT/src:$ROOT/cyberwheel"

TIMESTEPS="${TIMESTEPS:-300000}"
N_TRAJ="${N_TRAJ:-30}"

cd "$TD"

echo "=== p2: Train raw CW policy (no KC, no DAPN) ==="
"$CW_PY" train_cw_raw_policy.py --timesteps "$TIMESTEPS"

echo "=== p2: Zero-shot eval on NaSim ==="
"$CW_PY" eval_p2_nasim.py --episodes 100

echo "=== p2: Collect pe trajectories (if missing) ==="
if [[ ! -f artifacts/results/pe_trajectories.json ]]; then
    "$CW_PY" collect_pe_trajectories.py
fi

echo "=== p2: Trajectory comparison plot ==="
"$CW_PY" plot_p2_vs_pe.py --n "$N_TRAJ"

echo "Done → artifacts/results/p2_vs_pe.png"
