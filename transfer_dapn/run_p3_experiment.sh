#!/usr/bin/env bash
# p3 — NetSecGame + DAPN/KC → NaSim zero-shot transfer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TD="$ROOT/transfer_dapn"
NSG_PY="$HOME/.conda/envs/nsg/bin/python"
CW_PY="$HOME/.conda/envs/cyberwheel/bin/python"
export PYTHONPATH="$TD:$ROOT/src"

SAMPLES="${SAMPLES:-50000}"
TIMESTEPS="${TIMESTEPS:-300000}"
PE_STEPS="${PE_STEPS:-300000}"
N_TRAJ="${N_TRAJ:-50}"

ENC="${TD}/artifacts/models/dapn_encoder_nsg.pt.best.pt"
P3="${TD}/artifacts/policies/nsg_dapn_policy_final.zip"
PE="${TD}/artifacts/policies/nasim_nsg_invariant/best_model.zip"
OUT="${TD}/artifacts/results/traj_similarity_p3.png"

cd "$TD"

echo "=== p3: Collect KC observations (NSG + NaSim) ==="
"$NSG_PY" collect_nsg_kc_obs.py --nsg-samples "$SAMPLES" --nasim-samples 0 \
    --out data/nsg_kc_obs.npz
"$CW_PY" collect_nsg_kc_obs.py --nsg-samples 0 --nasim-samples "$SAMPLES" \
    --merge data/nsg_kc_obs.partial.npz --out data/nsg_kc_obs.npz

echo "=== p3: Train DAPN encoder ==="
"$CW_PY" train_dapn_encoder.py \
    --data data/nsg_kc_obs.npz \
    --out artifacts/models/dapn_encoder_nsg.pt \
    --epochs 100

echo "=== p3: Train NSG DAPN policy ==="
"$NSG_PY" train_nsg_dapn_policy.py \
    --encoder "$ENC" \
    --timesteps "$TIMESTEPS" --device cpu --no-eval

echo "=== p3: Train pe (NaSim NSG-aligned invariant baseline) ==="
"$CW_PY" train_nasim_invariant.py --scenario nsg \
    --encoder "$ENC" \
    --timesteps "$PE_STEPS" --no-eval

echo "=== p3: Zero-shot eval on NaSim ==="
"$CW_PY" eval_p3_nasim.py \
    --encoder "$ENC" \
    --policy "$P3" \
    --episodes 100

echo "=== p3: Trajectory comparison plot ==="
if [[ ! -f artifacts/results/pe_trajectories.json ]]; then
    echo "=== Collect canonical pe reference trajectories ==="
    /home/ssaika@cs.utep.edu/nasimemu-env/bin/python collect_pe_trajectories.py --n "$N_TRAJ"
fi
"$CW_PY" plot_p3_vs_pe.py \
    --encoder "$ENC" \
    --policy-p3 "$P3" \
    --pe-json "${TD}/artifacts/results/pe_trajectories.json" \
    --n "$N_TRAJ" \
    --out "$OUT"

echo "Done → $OUT"
