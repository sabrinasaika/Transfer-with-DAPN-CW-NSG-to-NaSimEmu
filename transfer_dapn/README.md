# DAPN Transfer: CyberWheel & NSG → NaSimEmu

Cross-domain policy transfer experiments using DAPN + kill-chain (KC) alignment.

## Policies

| ID | Source | Target | Script |
|----|--------|--------|--------|
| **p1** | CyberWheel + DAPN/KC | NaSim (two-subnet) | `run_p1_experiment.sh` |
| **p2** | CyberWheel raw | NaSim | `run_p2_experiment.sh` |
| **p3** | NetSecGame + DAPN/KC | NaSim (NSG-aligned) | `run_p3_experiment.sh` |
| **pe** | NaSim invariant baseline | NaSim | trained via `train_nasim_invariant.py` |

## Prerequisites

- [NASimEmu](https://github.com/jaromiru/NASimEmu) `src/` on `PYTHONPATH`
- Conda envs: `cyberwheel` (NaSim/CW), `nsg` (NetSecGame for p3 training)
- Optional: [CyberWheel](https://github.com/aicis/cyberwheel) for CW env YAMLs
- Optional: [NASimEmu-agents](https://github.com/) graph policy for `collect_pe_trajectories.py`

## Quick start

```bash
cd transfer_dapn

# p1 (CW → NaSim)
bash run_p1_experiment.sh

# p3 (NSG → NaSim)
bash run_p3_experiment.sh

# Plots
conda run -n cyberwheel python plot_p3_vs_pe.py --n 30
conda run -n cyberwheel python plot_action_distribution.py --n 50
```

## Key outputs

- `artifacts/results/traj_similarity.png` — p1 vs pe
- `artifacts/results/traj_similarity_p3.png` — p3 vs pe
- `artifacts/results/action_distribution_three.png` — combined action-type bars
- `artifacts/results/action_distribution_three_panels.png` — three separate bar charts

Trained weights are not in git; run the experiment scripts to regenerate them.
