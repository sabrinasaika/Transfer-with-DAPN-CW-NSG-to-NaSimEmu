#!/usr/bin/env bash
# p1 — CyberWheel + DAPN/KC → NaSim zero-shot transfer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TD="$ROOT/transfer_dapn"
CW_PY="$HOME/.conda/envs/cyberwheel/bin/python"
export PYTHONPATH="$TD:$ROOT/src:$ROOT/cyberwheel"

SCENARIO="${SCENARIO:-two_subnet}"
SAMPLES="${SAMPLES:-50000}"
TIMESTEPS="${TIMESTEPS:-300000}"
PE_STEPS="${PE_STEPS:-300000}"
N_TRAJ="${N_TRAJ:-50}"

cd "$TD"

echo "=== p1: Collect KC observations (CW + NaSim, scenario=$SCENARIO) ==="
"$CW_PY" collect_kc_obs.py --scenario "$SCENARIO" \
    --cw-samples "$SAMPLES" --nasim-samples 0 \
    --out "data/kc_obs_${SCENARIO}.npz"
"$CW_PY" collect_kc_obs.py --scenario "$SCENARIO" \
    --cw-samples 0 --nasim-samples "$SAMPLES" \
    --merge "data/kc_obs_${SCENARIO}.partial.npz" \
    --out "data/kc_obs_${SCENARIO}.npz"

echo "=== p1: Train DAPN encoder ==="
"$CW_PY" train_dapn_encoder.py --scenario "$SCENARIO" \
    --data "data/kc_obs_${SCENARIO}.npz" \
    --out "artifacts/models/dapn_encoder_${SCENARIO}.pt" \
    --epochs 100

ENC="${TD}/artifacts/models/dapn_encoder_${SCENARIO}.pt.best.pt"

echo "=== p1: Train CW DAPN policy ==="
"$CW_PY" train_cw_dapn_policy.py --scenario "$SCENARIO" \
    --encoder "$ENC" \
    --timesteps "$TIMESTEPS" --no-eval

echo "=== p1: Train pe (NaSim invariant baseline) ==="
"$CW_PY" train_nasim_invariant.py --scenario "$SCENARIO" \
    --encoder "$ENC" \
    --timesteps "$PE_STEPS" --no-eval

echo "=== p1: Zero-shot eval on NaSim ==="
"$CW_PY" eval_p1_nasim.py --scenario "$SCENARIO" \
    --encoder "$ENC" --episodes 100

echo "=== p1: Trajectory comparison plot ==="
"$CW_PY" plot_p1_vs_pe.py --scenario "$SCENARIO" \
    --encoder "$ENC" --n "$N_TRAJ"

echo "Done → artifacts/results/traj_similarity${SCENARIO:+_${SCENARIO}}.png"
