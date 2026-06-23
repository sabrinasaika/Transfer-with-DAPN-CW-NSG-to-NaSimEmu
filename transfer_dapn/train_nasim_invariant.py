"""Step 4 — Train pe: NaSim-native invariant policy (Experiment 1).

Trains PPO directly on NaSim using the DAPN encoder output (73-D) — the same
observation space as p1 — so both policies can be compared fairly.

Pipeline:
  NaSim KC+ctx (44-D) → frozen DAPN encoder → 73-D obs → PPO (pe)

The encoder must be trained first (train_dapn_encoder.py).

Saved artefacts
---------------
  artifacts/policies/nasim_kc_invariant/best_model.zip   — best checkpoint
  artifacts/policies/nasim_kc_invariant_final.zip        — final weights
  artifacts/results/training_curves_pe.json              — offline curve data

Usage
-----
  cd transfer_dapn
  conda run -n cyberwheel python train_nasim_invariant.py \
      --encoder artifacts/models/dapn_encoder_kc7.pt.best.pt

  # with W&B:
  conda run -n cyberwheel python train_nasim_invariant.py \
      --encoder artifacts/models/dapn_encoder_kc7.pt.best.pt --wandb

  # smoke test (fast):
  conda run -n cyberwheel python train_nasim_invariant.py \
      --encoder artifacts/models/dapn_encoder_kc7.pt.best.pt --timesteps 20000
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cyberwheel"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.nasim_native_wrapper import NaSimNativeWrapper
from envs.scenario_cfg import get_scenario
from callbacks import WinRateCallback

_ROOT = Path(__file__).resolve().parent
DEFAULT_ENCODER    = str(_ROOT / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
DEFAULT_OUT        = str(_ROOT / "artifacts/policies/nasim_kc_invariant_final.zip")
DEFAULT_BEST       = str(_ROOT / "artifacts/policies/nasim_kc_invariant")
DEFAULT_CURVE_OUT  = str(_ROOT / "artifacts/results/training_curves_pe.json")


def make_env(encoder_path: str, scenario: str = "two_subnet"):
    def _f():
        if scenario == "nsg":
            from envs.kc_envs import make_nasim_nsg_kc_env
            from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
            return DAPNEncoderWrapper(
                make_nasim_nsg_kc_env(), encoder_path, mask_is_target=True)
        return NaSimNativeWrapper(encoder_path=encoder_path, scenario=scenario)
    return _f


class CurveCallback(BaseCallback):
    """Records reward-per-step and win-rate per rollout for offline plotting."""

    def __init__(self, label: str, window: int = 100,
                 wandb_run=None, verbose: int = 0):
        super().__init__(verbose)
        self._label = label
        self._wandb = wandb_run
        self._ep_rewards: deque = deque(maxlen=window)
        self._ep_steps:   deque = deque(maxlen=window)
        self._ep_wins:    deque = deque(maxlen=window)
        self.epochs: list[dict] = []
        self._epoch = 0

    def _on_rollout_end(self) -> None:
        self._epoch += 1
        if not self._ep_rewards:
            return

        mean_r   = float(np.mean(self._ep_rewards))
        std_r    = float(np.std(self._ep_rewards))
        mean_win = float(np.mean(self._ep_wins))
        mean_stp = float(np.mean(self._ep_steps)) if self._ep_steps else 0.0
        rps      = mean_r / max(mean_stp, 1.0)  # reward per step

        rec = {
            "epoch":           self._epoch,
            "timestep":        self.num_timesteps,
            "reward_per_step": rps,
            "mean_return":     mean_r,
            "std_return":      std_r,
            "win_rate":        mean_win,
        }
        self.epochs.append(rec)

        if self._wandb is not None:
            self._wandb.log({
                f"{self._label}/reward_per_step": rps,
                f"{self._label}/win_rate":        mean_win,
                f"{self._label}/mean_return":     mean_r,
                "epoch": self._epoch,
            }, step=self.num_timesteps)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(ep["r"])
                self._ep_steps.append(ep["l"])
                win = bool(info.get("win", False))
                if not win and ep.get("r", -999) > 0:
                    win = True
                self._ep_wins.append(1.0 if win else 0.0)
        return True


def main():
    ap = argparse.ArgumentParser(
        description="Train pe: NaSim-native KC policy for Experiment 1")
    ap.add_argument("--scenario",     default="two_subnet",
                    choices=["two_subnet", "one_subnet", "nsg"])
    ap.add_argument("--encoder",      default=None,
                    help="Path to trained DAPN encoder (.pt)")
    ap.add_argument("--timesteps",    type=int,   default=300_000)
    ap.add_argument("--out",          default=None)
    ap.add_argument("--best-dir",     default=None)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--ent-coef",     type=float, default=0.01)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--curve-out",    default=DEFAULT_CURVE_OUT,
                    help="Path to save offline training-curve JSON")
    ap.add_argument("--wandb",        action="store_true",
                    help="Enable Weights & Biases logging")
    ap.add_argument("--wandb-project", default="nasimemu-exp1")
    ap.add_argument("--wandb-name",    default="pe-nasim-invariant")
    ap.add_argument("--no-eval",       action="store_true",
                    help="Skip EvalCallback (faster training)")
    args = ap.parse_args()
    if args.scenario == "nsg":
        if args.encoder is None:
            args.encoder = str(_ROOT / "artifacts/models/dapn_encoder_nsg.pt.best.pt")
        if args.out is None:
            args.out = str(_ROOT / "artifacts/policies/nasim_nsg_invariant_final.zip")
        if args.best_dir is None:
            args.best_dir = str(_ROOT / "artifacts/policies/nasim_nsg_invariant")
        if args.curve_out == DEFAULT_CURVE_OUT:
            args.curve_out = str(_ROOT / "artifacts/results/training_curves_pe_nsg.json")
    else:
        cfg = get_scenario(args.scenario)
        if args.encoder is None:
            args.encoder = str(_ROOT / f"{cfg.encoder_out}.best.pt")
        if args.out is None:
            args.out = str(_ROOT / cfg.pe_policy_final)
        if args.best_dir is None:
            args.best_dir = str(_ROOT / cfg.pe_policy_dir)

    if not Path(args.encoder).exists():
        print(f"ERROR: encoder not found at {args.encoder}")
        print("Run collect_kc_obs.py then train_dapn_encoder.py first.")
        import sys; sys.exit(1)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    os.makedirs(args.best_dir, exist_ok=True)
    ckpt_path = str(Path(args.out).parent / "nasim_kc_invariant_ckpts")
    os.makedirs(ckpt_path, exist_ok=True)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    "policy": "pe_nasim_invariant",
                    "scenario": "fixed_dmz_two_subnet",
                    "timesteps": args.timesteps,
                    "seed": args.seed,
                    "ent_coef": args.ent_coef,
                    "lr": args.lr,
                },
            )
            print(f"W&B run: {wandb_run.url}")
        except ImportError:
            print("wandb not installed — skipping W&B logging. pip install wandb")

    train_env = VecMonitor(DummyVecEnv([make_env(args.encoder, args.scenario)]))
    eval_env  = VecMonitor(DummyVecEnv([make_env(args.encoder, args.scenario)]))

    obs_dim = train_env.observation_space.shape[0]
    n_act   = train_env.action_space.n

    print(f"\n{'='*60}")
    print("Training pe: NaSim-native invariant policy (DAPN encoded)")
    print(f"  scenario   : {args.scenario}")
    print(f"  encoder    : {args.encoder}")
    print(f"  obs dim    : {obs_dim}  (73-D encoded)")
    print(f"  action dim : {n_act}  (Discrete(8) slots)")
    print(f"  timesteps  : {args.timesteps:,}")
    print(f"  seed       : {args.seed}")
    print(f"  output     : {args.out}")
    print(f"  best       : {args.best_dir}/best_model.zip")
    print(f"  W&B        : {'on' if wandb_run else 'off'}")
    print(f"{'='*60}\n")

    model = PPO(
        "MlpPolicy", train_env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    eval_freq = max(1, args.timesteps // 20)
    save_freq = max(1, args.timesteps // 30)

    curve_cb = CurveCallback(label="pe", wandb_run=wandb_run)

    callbacks = CallbackList([
        WinRateCallback(window=100),
        curve_cb,
        EvalCallback(
            eval_env,
            n_eval_episodes=20,
            eval_freq=eval_freq,
            best_model_save_path=args.best_dir,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=ckpt_path,
            name_prefix="nasim_kc_inv",
            verbose=0,
        ),
    ])

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[interrupted — saving current weights]")

    model.save(args.out)
    if args.no_eval:
        model.save(str(Path(args.best_dir) / "best_model"))

    # Save offline curve data for plot_training_curves.py
    curve_out = Path(args.curve_out)
    curve_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"label": "pe", "policy": "NaSim invariant KC", "epochs": curve_cb.epochs}
    with open(curve_out, "w") as f:
        json.dump(payload, f, indent=2)

    if wandb_run:
        wandb_run.finish()

    print(f"\nFinal policy saved  → {args.out}")
    print(f"Best checkpoint     → {args.best_dir}/best_model.zip")
    print(f"Training curves     → {curve_out}")
    print(f"\nNext: python plot_training_curves.py --pe {curve_out}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
