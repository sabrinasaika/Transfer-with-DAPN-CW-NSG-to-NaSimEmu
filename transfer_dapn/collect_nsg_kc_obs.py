"""Collect 62-D KC+ctx observations from NetSecGame and NSG-aligned NaSim.

10-slot layout (kill_chain_nsg): KC_DIM=60, +2 ctx = 62-D obs.

Output: data/nsg_kc_obs.npz
  source_obs  (N_nsg, 62)  — NetSecGame KC+ctx
  target_obs  (N_nas, 62)  — NaSim (nasim_two_subnet) KC+ctx

Usage:
  # NSG side (Python 3.12+):
  python3.12 collect_nsg_kc_obs.py --nsg-samples 5000 --nasim-samples 0

  # NaSim side (cyberwheel):
  conda run -n cyberwheel python collect_nsg_kc_obs.py --nsg-samples 0 --nasim-samples 5000

  # Both (if python3.12 has nasimemu on PYTHONPATH):
  python3.12 collect_nsg_kc_obs.py --nsg-samples 5000 --nasim-samples 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.kill_chain_nsg import MAX_SLOTS, KC_DIM, KC_FEATS

_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = str(_ROOT / "data/nsg_kc_obs.npz")


def collect_sequential(env_fn, n_samples: int, label: str) -> np.ndarray:
    env = env_fn()
    obs_list = []
    obs, _ = env.reset()
    obs_list.append(obs.copy())
    slot = 0
    while len(obs_list) < n_samples:
        action = slot % (MAX_SLOTS + 1)
        obs, _r, term, trunc, _ = env.step(action)
        obs_list.append(obs.copy())
        slot += 1
        if term or trunc:
            obs, _ = env.reset()
            obs_list.append(obs.copy())
            slot = 0
        if len(obs_list) % 500 == 0:
            print(f"  {label}: {len(obs_list)}/{n_samples}")
    env.close()
    arr = np.array(obs_list[:n_samples], dtype=np.float32)
    print(f"  {label}: collected {len(arr)} obs, shape={arr.shape}")
    return arr


def collect_nsg_sequential(n_samples: int, label: str = "NSG") -> np.ndarray:
    from envs.kc_envs import make_nsg_kc_env
    return collect_sequential(make_nsg_kc_env, n_samples, label)


def collect_nasim_nsg_sequential(n_samples: int, label: str = "NaSim-NSG") -> np.ndarray:
    from envs.kc_envs import make_nasim_nsg_kc_env
    return collect_sequential(make_nasim_nsg_kc_env, n_samples, label)


def main():
    ap = argparse.ArgumentParser(description="Collect NSG + NaSim KC obs for p3")
    ap.add_argument("--nsg-samples", type=int, default=50000)
    ap.add_argument("--nasim-samples", type=int, default=50000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", default=None,
                    help="Merge into existing npz (provide path)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    source = target = None
    if args.merge and Path(args.merge).exists():
        prev = np.load(args.merge)
        source = prev["source_obs"] if "source_obs" in prev else None
        target = prev["target_obs"] if "target_obs" in prev else None

    if args.nsg_samples > 0:
        print(f"\nCollecting NSG observations ({args.nsg_samples})…")
        source = collect_nsg_sequential(args.nsg_samples)

    if args.nasim_samples > 0:
        print(f"\nCollecting NaSim-NSG observations ({args.nasim_samples})…")
        target = collect_nasim_nsg_sequential(args.nasim_samples)

    if source is None and args.nsg_samples == 0 and args.merge:
        raise SystemExit("NSG samples missing — run with --nsg-samples > 0 first")
    if target is None and args.nasim_samples == 0 and args.merge:
        raise SystemExit("NaSim samples missing — run with --nasim-samples > 0")

    if source is None or target is None:
        # Save partial for two-pass collection across Python envs.
        partial = out.with_suffix(".partial.npz")
        save_kw = {}
        if source is not None:
            save_kw["source_obs"] = source
        if target is not None:
            save_kw["target_obs"] = target
        np.savez_compressed(str(partial), **save_kw)
        print(f"\nPartial save → {partial}  (run the other domain, then --merge {partial})")
        if source is None or target is None:
            return

    np.savez_compressed(str(out), source_obs=source, target_obs=target)
    print(f"\nSaved → {out}")
    print(f"  source_obs : {source.shape}  (NSG)")
    print(f"  target_obs : {target.shape}  (NaSim-NSG)")


if __name__ == "__main__":
    main()
