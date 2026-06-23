"""
Step 5 — Baseline: PPO trained directly on NaSim (no CW, no DAPN).

This is the comparison point for the DAPN transfer experiment.
The same scenario and hyperparameters as the CW training (Step 3) are used
so the only difference is the training domain.

Saves:
  artifacts/policies/nasim_baseline_policy.zip
  artifacts/policies/nasim_best/best_model.zip

Usage:
  conda run -n cyberwheel python 5_baseline_nasim.py
  conda run -n cyberwheel python 5_baseline_nasim.py --timesteps 50000  # smoke test
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cyberwheel"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList)

from envs.nasim_native_wrapper import NaSimNativeWrapper
from callbacks import WinRateCallback


def make_env():
    def _f():
        return NaSimNativeWrapper()
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", default=300_000, type=int)
    ap.add_argument("--out",       default="artifacts/policies/nasim_baseline_policy.zip")
    ap.add_argument("--seed",      default=42, type=int)
    ap.add_argument("--ent-coef",  default=0.01, type=float)
    args = ap.parse_args()

    out_dir   = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(out_dir / "nasim_best")
    ckpt_path = str(out_dir / "nasim_checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)

    print(f"\n{'='*60}")
    print("Baseline: PPO trained directly on NaSim (150-D obs)")
    print(f"  scenario  : fixed_dmz_one_subnet_4host.v2.yaml")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  output    : {args.out}")
    print(f"{'='*60}\n")

    train_env = VecMonitor(DummyVecEnv([make_env()]))
    eval_env  = VecMonitor(DummyVecEnv([make_env()]))

    print(f"  obs dim    : {train_env.observation_space.shape[0]}")
    print(f"  action dim : {train_env.action_space.n}\n")

    model = PPO(
        "MlpPolicy", train_env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    eval_freq = max(1, args.timesteps // 20)
    callbacks = CallbackList([
        WinRateCallback(window=100),
        EvalCallback(
            eval_env,
            n_eval_episodes=10,
            eval_freq=eval_freq,
            best_model_save_path=best_path,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(1, args.timesteps // 30),
            save_path=ckpt_path,
            name_prefix="nasim_baseline",
            verbose=0,
        ),
    ])

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[interrupted — saving]")

    model.save(args.out)
    print(f"\nBaseline policy saved  → {args.out}")
    print(f"Best checkpoint        → {best_path}/best_model.zip")
    print(f"\nCompare with DAPN transfer result from 4_eval_nasim.py")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
