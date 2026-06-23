"""Train p2: vanilla PPO in raw CyberWheel (no KC wrapper, no DAPN encoder).

Observation : 50-D native CW obs_vec
Action      : Discrete(42) = 7 hosts × 6 kill-chain phases

This is the naive-transfer baseline — no domain adaptation.

Usage (cyberwheel conda env):
  cd /home/ssaika@cs.utep.edu/NASimEmu
  conda run -n cyberwheel python transfer_dapn/train_cw_raw_policy.py \\
      --timesteps 300000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DAPN = _REPO / "transfer_dapn"

sys.path.insert(0, str(_REPO / "cyberwheel"))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_DAPN))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.cw_native_wrapper import CWNativeWrapper

_OUT_DIR = str(_DAPN / "artifacts/policies/cw_raw_policy")
_FINAL   = str(_DAPN / "artifacts/policies/cw_raw_policy_final.zip")


class WinRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._wins = self._eps = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if info.get("win"):
                self._wins += 1
            if self.locals.get("dones", [False])[0]:
                self._eps += 1
        return True

    def _on_rollout_end(self):
        if self._eps > 0:
            self.logger.record("win_rate", self._wins / self._eps)
            self._wins = self._eps = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out-dir",   default=_OUT_DIR)
    ap.add_argument("--final",     default=_FINAL)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    print("=" * 55)
    print("Training p2: raw CW policy (no KC, no DAPN)")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  out dir   : {args.out_dir}")
    print("=" * 55)

    def make_env():
        return CWNativeWrapper()

    train_env = DummyVecEnv([make_env])
    eval_env  = DummyVecEnv([make_env])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.out_dir,
        log_path=args.out_dir,
        eval_freq=15_000,
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )
    win_cb = WinRateCallback()

    model = PPO(
        "MlpPolicy", train_env,
        n_steps=4096, batch_size=256,
        n_epochs=10, learning_rate=3e-4,
        ent_coef=0.01, gamma=0.99,
        verbose=1, seed=args.seed,
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps,
                callback=[eval_cb, win_cb])

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save(args.final)
    print(f"\nFinal policy → {args.final}")
    print(f"Best ckpt    → {args.out_dir}/best_model.zip")


if __name__ == "__main__":
    main()
