"""Extract pe training curve from NASimEmu-agents output.log → our JSON format."""

import ast
import json
import re
import sys
from pathlib import Path

LOG = Path("/home/ssaika@cs.utep.edu/NASimEmu-agents/wandb/run-20260612_093816-biw5us4u/files/output.log")
OUT = Path(__file__).parent / "artifacts/results/training_curves_pe.json"

def _clean(line: str) -> str:
    """Strip np.float64(...) wrappers and wandb objects so ast can parse."""
    line = re.sub(r"np\.float64\(([^)]+)\)", r"\1", line)
    line = re.sub(r"np\.int64\(([^)]+)\)", r"\1", line)
    # Replace wandb Histogram objects with None
    line = re.sub(r"<wandb\.sdk[^>]+>", "None", line)
    return line

records = []
epoch = 0
for line in LOG.read_text().splitlines():
    line = line.strip()
    if not line.startswith("{'env_steps'"):
        continue
    try:
        d = ast.literal_eval(_clean(line))
    except Exception:
        continue

    ep = d.get("eval_perf", {})
    trn = ep.get("eval_trn", {})
    tst = ep.get("eval_tst", {})
    env_steps = d.get("env_steps", 0)
    rps_trn = float(trn.get("reward_avg", 0.0))
    rps_tst = float(tst.get("reward_avg", 0.0))
    captured = float(trn.get("captured_avg", 0.0))

    epoch += 1
    records.append({
        "epoch":           epoch,
        "timestep":        env_steps,
        "reward_per_step": rps_trn,
        "reward_per_step_tst": rps_tst,
        "mean_return":     float(trn.get("reward_avg_episodes", 0.0)),
        "win_rate":        1.0 if captured >= 1.0 else captured,
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
payload = {"label": "pe", "policy": "NaSim Invariant (NASimNetInvMAct)", "epochs": records}
OUT.write_text(json.dumps(payload, indent=2))
print(f"Extracted {len(records)} epochs → {OUT}")
if records:
    print(f"  first: steps={records[0]['timestep']}  rps={records[0]['reward_per_step']:.4f}")
    print(f"  last:  steps={records[-1]['timestep']}  rps={records[-1]['reward_per_step']:.4f}")
