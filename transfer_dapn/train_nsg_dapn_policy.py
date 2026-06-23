"""Step 3b — Train NSG policy (p3 source) using DAPN encoder output.

Same architecture as train_cw_dapn_policy.py but trains in NetSecGame.

Requires Python 3.12+ with netsecgame[server], stable-baselines3, torch.

Usage:
  cd transfer_dapn
  python3.12 train_nsg_dapn_policy.py --device cpu --timesteps 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.kc_envs import make_nsg_kc_env
from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
from callbacks import WinRateCallback

_ROOT = Path(__file__).resolve().parent
DEFAULT_ENCODER  = str(_ROOT / "artifacts/models/dapn_encoder_nsg.pt.best.pt")
DEFAULT_OUT      = str(_ROOT / "artifacts/policies/nsg_dapn_policy_final.zip")
DEFAULT_BEST_DIR = str(_ROOT / "artifacts/policies/nsg_dapn_policy")
DEFAULT_CURVE    = str(_ROOT / "artifacts/results/training_curves_nsg_dapn.json")


def make_env(encoder_path: str):
    _here = str(Path(__file__).resolve().parent)
    _src  = str(Path(__file__).resolve().parents[1] / "src")

    def _f():
        import sys
        for p in [_here, _src]:
            if p not in sys.path:
                sys.path.insert(0, p)
        from envs.kc_envs import make_nsg_kc_env
        from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
        base = make_nsg_kc_env()
        return DAPNEncoderWrapper(base, encoder_path, mask_is_target=True)
    return _f


class CurveCallback(BaseCallback):
    def __init__(self, label: str, window: int = 100, verbose=0):
        super().__init__(verbose)
        self._label = label
        self._rewards: deque = deque(maxlen=window)
        self._steps: deque = deque(maxlen=window)
        self._wins: deque = deque(maxlen=window)
        self.epochs: list = []
        self._epoch = 0

    def _on_rollout_end(self):
        self._epoch += 1
        if not self._rewards:
            return
        mean_r = float(np.mean(self._rewards))
        mean_w = float(np.mean(self._wins))
        mean_s = float(np.mean(self._steps)) if self._steps else 1.0
        self.epochs.append({
            "epoch": self._epoch, "timestep": self.num_timesteps,
            "reward_per_step": mean_r / max(mean_s, 1.0),
            "mean_return": mean_r, "win_rate": mean_w,
        })

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._rewards.append(ep["r"])
                self._steps.append(ep["l"])
                win = bool(info.get("win", False)) or ep.get("r", -999) > 0
                self._wins.append(1.0 if win else 0.0)
        return True


def main():
    ap = argparse.ArgumentParser(description="Train NSG DAPN policy (p3 source)")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--best-dir", default=DEFAULT_BEST_DIR)
    ap.add_argument("--curve-out", default=DEFAULT_CURVE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    if not Path(args.encoder).exists():
        print(f"ERROR: encoder not found at {args.encoder}")
        print("Run collect_nsg_kc_obs.py then train_dapn_encoder.py --data data/nsg_kc_obs.npz")
        sys.exit(1)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    os.makedirs(args.best_dir, exist_ok=True)

    train_env = VecMonitor(DummyVecEnv([make_env(args.encoder)]))
    eval_env = VecMonitor(DummyVecEnv([make_env(args.encoder)]))

    print(f"\nTraining p3: NSG policy with DAPN encoder")
    print(f"  encoder   : {args.encoder}")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  device    : {args.device}\n")

    model = PPO(
        "MlpPolicy", train_env,
        verbose=1, seed=args.seed,
        n_steps=2048, batch_size=64, n_epochs=10,
        learning_rate=3e-4, gamma=0.99,
        device=args.device,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    curve_cb = CurveCallback("p3")
    callbacks = [WinRateCallback(window=50), curve_cb]
    if not args.no_eval:
        callbacks.append(EvalCallback(
            eval_env, n_eval_episodes=5, eval_freq=max(5000, args.timesteps // 20),
            best_model_save_path=args.best_dir, verbose=1))

    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))
    model.save(args.out)
    if args.no_eval:
        model.save(str(Path(args.best_dir) / "best_model"))

    with open(args.curve_out, "w") as f:
        json.dump({"label": "p3", "policy": "NSG DAPN", "epochs": curve_cb.epochs}, f, indent=2)

    print(f"\nFinal policy → {args.out}")
    print(f"Best ckpt    → {args.best_dir}/best_model.zip")


if __name__ == "__main__":
    main()
