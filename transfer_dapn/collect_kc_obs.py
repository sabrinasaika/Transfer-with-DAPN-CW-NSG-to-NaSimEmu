"""Step 1 — Collect 44-D KC+ctx observations from CW and NaSim.

Runs random rollouts in each domain and saves game-state features for
adversarial encoder training.

Output: data/kc_obs.npz with keys:
  source_obs  (N_cw,  44)  — CW KC+ctx observations
  target_obs  (N_nas, 44)  — NaSim KC+ctx observations

Usage:
  cd transfer_dapn
  conda run -n cyberwheel python collect_kc_obs.py \
      --cw-samples 3000 --nasim-samples 3000 --out data/kc_obs.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cyberwheel"))

from envs.kc_envs import make_cw_kc_env, make_nasim_kc_env
from envs.kill_chain import MAX_SLOTS
from envs.scenario_cfg import get_scenario

_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = str(_ROOT / "data/kc_obs.npz")


def collect(env_fn, n_samples: int, label: str) -> np.ndarray:
    """Run random rollouts; collect n_samples observations."""
    env  = env_fn()
    obs_list = []
    obs, _ = env.reset()
    obs_list.append(obs.copy())

    while len(obs_list) < n_samples:
        action = env.action_space.sample()
        obs, _r, term, trunc, _ = env.step(action)
        obs_list.append(obs.copy())
        if term or trunc:
            obs, _ = env.reset()
            obs_list.append(obs.copy())
        if len(obs_list) % 500 == 0:
            print(f"  {label}: {len(obs_list)}/{n_samples}")

    env.close()
    arr = np.array(obs_list[:n_samples], dtype=np.float32)
    print(f"  {label}: collected {len(arr)} obs, shape={arr.shape}")
    return arr


def collect_sequential(env_fn, n_samples: int, label: str) -> np.ndarray:
    """Collect obs by cycling KC slots 0→6 so all kill-chain stages appear."""
    env = env_fn()
    obs_list = []
    obs, _ = env.reset()
    obs_list.append(obs.copy())

    slot = 0
    while len(obs_list) < n_samples:
        action = slot % (MAX_SLOTS + 1)   # cycle 0‥7 (7 = noop)
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
    _print_slot_diversity(arr, label)
    return arr


def _print_slot_diversity(arr: np.ndarray, label: str) -> None:
    from envs.kill_chain import KC_DIM, KC_FEATS
    kc = arr[:, :KC_DIM].reshape(-1, MAX_SLOTS, KC_FEATS)
    print(f"  Slot phase diversity ({label}):")
    for s in range(MAX_SLOTS):
        ph = kc[:, s, 0]
        uniq = sorted(set(round(float(x), 3) for x in ph if x > 0))
        print(f"    slot {s}: nonzero={float((ph>0).mean()):.3f}  unique phases={uniq}")


def collect_nasim_sequential(n_samples: int, scenario: str, label: str = "NaSim") -> np.ndarray:
    return collect_sequential(lambda: make_nasim_kc_env(scenario), n_samples, label)


def main():
    ap = argparse.ArgumentParser(description="Collect KC+ctx obs from CW and NaSim")
    ap.add_argument("--scenario", default="two_subnet", choices=["two_subnet", "one_subnet"])
    ap.add_argument("--cw-samples",    type=int, default=50000)
    ap.add_argument("--nasim-samples", type=int, default=50000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", default=None, help="Merge into existing partial npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = get_scenario(args.scenario)
    out = Path(args.out or str(_ROOT / cfg.data_kc_obs))
    np.random.seed(args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)

    source = target = None
    if args.merge and Path(args.merge).exists():
        prev = np.load(args.merge)
        source = prev["source_obs"] if "source_obs" in prev else None
        target = prev["target_obs"] if "target_obs" in prev else None

    if args.cw_samples > 0:
        print(f"\nCollecting CW observations ({args.cw_samples}) — {cfg.name}…")
        source = collect_sequential(lambda: make_cw_kc_env(args.scenario),
                                    args.cw_samples, "CW")

    if args.nasim_samples > 0:
        print(f"\nCollecting NaSim observations ({args.nasim_samples}) — {cfg.name}…")
        target = collect_nasim_sequential(args.nasim_samples, args.scenario)

    if source is None or target is None:
        partial = out.with_suffix(".partial.npz")
        save_kw = {}
        if source is not None:
            save_kw["source_obs"] = source
        if target is not None:
            save_kw["target_obs"] = target
        np.savez_compressed(str(partial), **save_kw)
        print(f"\nPartial save → {partial}")
        return

    np.savez_compressed(str(out), source_obs=source, target_obs=target)
    print(f"\nSaved → {out}")
    print(f"  source_obs : {source.shape}  (CW)")
    print(f"  target_obs : {target.shape}  (NaSim)")


if __name__ == "__main__":
    main()
