"""Step 3 — Train CW policy (p1 source) using DAPN encoder output.

Trains PPO in CyberWheel with observations passed through the frozen DAPN
encoder → 73-D policy input [latent(64) | is_target(7) | ctx(2)].

At inference in NaSim, the same frozen encoder processes NaSim KC+ctx obs
to produce the same 73-D input — enabling zero-shot transfer.

Saved artefacts:
  artifacts/policies/cw_dapn_policy/best_model.zip  — best checkpoint
  artifacts/policies/cw_dapn_policy_final.zip        — final weights
  artifacts/results/training_curves_cw_dapn.json     — training curve data

Usage:
  cd transfer_dapn
  conda run -n cyberwheel python train_cw_dapn_policy.py
  conda run -n cyberwheel python train_cw_dapn_policy.py --wandb
  conda run -n cyberwheel python train_cw_dapn_policy.py --timesteps 20000  # smoke
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
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from envs.kc_envs import make_cw_kc_env
from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
from envs.scenario_cfg import get_scenario
from callbacks import WinRateCallback

_ROOT = Path(__file__).resolve().parent
DEFAULT_ENCODER  = str(_ROOT / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
DEFAULT_OUT      = str(_ROOT / "artifacts/policies/cw_dapn_policy_final.zip")
DEFAULT_BEST_DIR = str(_ROOT / "artifacts/policies/cw_dapn_policy")
DEFAULT_CURVE    = str(_ROOT / "artifacts/results/training_curves_cw_dapn.json")


def make_env(encoder_path: str, scenario: str = "two_subnet"):
    _here = str(Path(__file__).resolve().parent)
    _src  = str(Path(__file__).resolve().parents[1] / "src")
    _cw   = str(Path(__file__).resolve().parents[1] / "cyberwheel")

    def _f():
        import sys
        for p in [_here, _src, _cw]:
            if p not in sys.path:
                sys.path.insert(0, p)
        from envs.kc_envs import make_cw_kc_env
        from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
        base = make_cw_kc_env(scenario)
        return DAPNEncoderWrapper(base, encoder_path, mask_is_target=True)
    return _f


class CurveCallback(BaseCallback):
    def __init__(self, label: str, window: int = 100, wandb_run=None, verbose=0):
        super().__init__(verbose)
        self._label   = label
        self._wandb   = wandb_run
        self._rewards: deque = deque(maxlen=window)
        self._steps:   deque = deque(maxlen=window)
        self._wins:    deque = deque(maxlen=window)
        self.epochs:   list  = []
        self._epoch   = 0

    def _on_rollout_end(self):
        self._epoch += 1
        if not self._rewards:
            return
        mean_r  = float(np.mean(self._rewards))
        std_r   = float(np.std(self._rewards))
        mean_w  = float(np.mean(self._wins))
        mean_s  = float(np.mean(self._steps)) if self._steps else 1.0
        rps     = mean_r / max(mean_s, 1.0)
        rec = {"epoch": self._epoch, "timestep": self.num_timesteps,
               "reward_per_step": rps, "mean_return": mean_r,
               "std_return": std_r, "win_rate": mean_w}
        self.epochs.append(rec)
        if self._wandb:
            self._wandb.log(
                {f"{self._label}/reward_per_step": rps,
                 f"{self._label}/win_rate": mean_w,
                 f"{self._label}/mean_return": mean_r,
                 "epoch": self._epoch},
                step=self.num_timesteps)

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
    ap = argparse.ArgumentParser(description="Train CW DAPN policy (p1 source)")
    ap.add_argument("--scenario",     default="two_subnet",
                    choices=["two_subnet", "one_subnet"])
    ap.add_argument("--encoder",      default=None)
    ap.add_argument("--timesteps",    type=int,   default=300_000)
    ap.add_argument("--out",          default=None)
    ap.add_argument("--best-dir",     default=None)
    ap.add_argument("--curve-out",    default=DEFAULT_CURVE)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--ent-coef",     type=float, default=0.01)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--n-envs",        type=int,   default=8,
                    help="Parallel envs (SubprocVecEnv). Use 1 for DummyVecEnv.")
    ap.add_argument("--device",        default="auto",
                    help="PyTorch device: auto, cpu, cuda")
    ap.add_argument("--no-eval",       action="store_true",
                    help="Skip EvalCallback (much faster; saves final model as best)")
    ap.add_argument("--wandb",        action="store_true")
    ap.add_argument("--wandb-project", default="nasimemu-exp1")
    ap.add_argument("--wandb-name",    default="p1-cw-dapn")
    args = ap.parse_args()
    cfg = get_scenario(args.scenario)
    if args.encoder is None:
        args.encoder = str(_ROOT / f"{cfg.encoder_out}.best.pt")
    if args.out is None:
        args.out = str(_ROOT / cfg.p1_policy_final)
    if args.best_dir is None:
        args.best_dir = str(_ROOT / cfg.p1_policy_dir)

    if not Path(args.encoder).exists():
        print(f"ERROR: encoder not found at {args.encoder}")
        print("Run collect_kc_obs.py then train_dapn_encoder.py first.")
        sys.exit(1)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    os.makedirs(args.best_dir, exist_ok=True)
    ckpt_dir = str(Path(args.out).parent / "cw_dapn_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, name=args.wandb_name,
                config={"policy": "p1_cw_dapn", "timesteps": args.timesteps,
                        "seed": args.seed})
            print(f"W&B: {wandb_run.url}")
        except ImportError:
            print("wandb not installed — skipping")

    n_envs = 1  # CyberWheel doesn't survive multiprocessing; single env is safest
    train_env = VecMonitor(DummyVecEnv([make_env(args.encoder, args.scenario)]))
    eval_env  = VecMonitor(DummyVecEnv([make_env(args.encoder, args.scenario)]))

    obs_dim = train_env.observation_space.shape[0]
    n_act   = train_env.action_space.n
    n_steps = 4096  # larger rollout buffer compensates for single env

    print(f"\n{'='*60}")
    print("Training p1: CW policy with DAPN encoder (7-slot KC)")
    print(f"  encoder    : {args.encoder}")
    print(f"  obs dim    : {obs_dim}  (73-D encoded, is_target masked for CW)")
    print(f"  action dim : {n_act}")
    print(f"  timesteps  : {args.timesteps:,}")
    print(f"  n_envs     : {n_envs}  (n_steps={n_steps} each)")
    print(f"  device     : {args.device}")
    print(f"  seed       : {args.seed}")
    print(f"  W&B        : {'on' if wandb_run else 'off'}")
    print(f"{'='*60}\n")
    model = PPO(
        "MlpPolicy", train_env,
        verbose=1, seed=args.seed,
        n_steps=n_steps, batch_size=64, n_epochs=10,
        learning_rate=args.lr, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=args.ent_coef,
        device=args.device,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    eval_freq = max(1, args.timesteps // 20)
    save_freq = max(1, args.timesteps // 30)
    curve_cb  = CurveCallback("p1", wandb_run=wandb_run)

    cb_list = [WinRateCallback(window=100), curve_cb,
               CheckpointCallback(save_freq=save_freq, save_path=ckpt_dir,
                                  name_prefix="cw_dapn", verbose=0)]
    if not args.no_eval:
        cb_list.insert(2, EvalCallback(eval_env, n_eval_episodes=20,
                                       eval_freq=eval_freq,
                                       best_model_save_path=args.best_dir,
                                       verbose=1))
    callbacks = CallbackList(cb_list)

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[interrupted — saving]")

    model.save(args.out)
    if args.no_eval:
        model.save(str(Path(args.best_dir) / "best_model"))

    curve_out = Path(args.curve_out)
    curve_out.parent.mkdir(parents=True, exist_ok=True)
    with open(curve_out, "w") as f:
        json.dump({"label": "p1", "policy": "CW DAPN", "epochs": curve_cb.epochs}, f, indent=2)

    if wandb_run:
        wandb_run.finish()

    print(f"\nFinal policy → {args.out}")
    print(f"Best ckpt    → {args.best_dir}/best_model.zip")
    print(f"Curves       → {curve_out}")
    print(f"\nNext: python train_nasim_invariant.py --encoder {args.encoder}")


if __name__ == "__main__":
    main()
