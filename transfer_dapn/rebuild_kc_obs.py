"""Rebuild kc_obs.npz using pe-driven NaSim obs (more diverse than random/sequential)."""
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

src_path  = _ROOT / "data/kc_obs.npz"        # has CW source_obs
tgt_path  = _ROOT / "data/nasim_kc_obs_pe.npy"
out_path  = _ROOT / "data/kc_obs.npz"

src_obs = np.load(str(src_path))["source_obs"].astype(np.float32)
tgt_obs = np.load(str(tgt_path)).astype(np.float32)

# Trim to equal size
n = min(len(src_obs), len(tgt_obs))
src_obs = src_obs[:n]
tgt_obs = tgt_obs[:n]

print(f"source_obs (CW)    : {src_obs.shape}")
print(f"target_obs (NaSim) : {tgt_obs.shape}")

# Diversity check
from envs.kill_chain import KC_DIM, KC_FEATS, MAX_SLOTS
kc = tgt_obs[:, :KC_DIM].reshape(-1, MAX_SLOTS, KC_FEATS)
print("\nDiversity (pe-driven NaSim) — phase per slot:")
for s in range(MAX_SLOTS):
    ph = kc[:, s, 0]
    uniq = sorted(set(round(float(x), 2) for x in ph))
    print(f"  slot {s}: {uniq}  nonzero={float((ph>0).mean()):.3f}")

np.savez_compressed(str(out_path), source_obs=src_obs, target_obs=tgt_obs)
print(f"\nSaved → {out_path}")
