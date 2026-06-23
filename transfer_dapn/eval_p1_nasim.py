"""Zero-shot evaluation of p1 (CW-trained DAPN policy) in NaSim.

Tests whether the policy trained entirely in CyberWheel transfers to NaSim
without any additional training, using the frozen DAPN encoder.

Usage:
  cd transfer_dapn
  conda run -n cyberwheel python eval_p1_nasim.py
  conda run -n cyberwheel python eval_p1_nasim.py --episodes 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cyberwheel"))

from stable_baselines3 import PPO
from envs.nasim_native_wrapper import NaSimNativeWrapper
from envs.scenario_cfg import get_scenario

_ROOT = Path(__file__).resolve().parent
DEFAULT_ENCODER = str(_ROOT / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
DEFAULT_POLICY  = str(_ROOT / "artifacts/policies/cw_dapn_policy/best_model.zip")


def run_episode(model, env, deterministic=True):
    obs, _ = env.reset()
    total_r, steps = 0.0, 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(int(action))
        total_r += r
        steps += 1
        if terminated or truncated:
            win = bool(info.get("win", False) or terminated)
            return win, total_r, steps


def main():
    ap = argparse.ArgumentParser(description="Zero-shot p1 → NaSim evaluation")
    ap.add_argument("--scenario", default="two_subnet", choices=["two_subnet", "one_subnet"])
    ap.add_argument("--policy",   default=None)
    ap.add_argument("--encoder",  default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    cfg = get_scenario(args.scenario)
    encoder_path = args.encoder or str(_ROOT / f"{cfg.encoder_out}.best.pt")
    policy_path = args.policy or str(_ROOT / cfg.p1_policy_dir / "best_model.zip")

    env   = NaSimNativeWrapper(encoder_path=encoder_path, scenario=args.scenario)
    model = PPO.load(policy_path)

    print(f"\n{'='*55}")
    print("Zero-shot transfer: p1 (CW-trained) → NaSim")
    print(f"  scenario : {args.scenario}")
    print(f"  policy   : {policy_path}")
    print(f"  encoder  : {encoder_path}")
    print(f"  episodes : {args.episodes}")
    print(f"{'='*55}\n")

    rng = np.random.default_rng(args.seed)
    wins, rewards, steps_list = 0, [], []

    for ep in range(args.episodes):
        env.reset(seed=int(rng.integers(0, 2**31)))
        win, ret, steps = run_episode(model, env)
        wins += int(win)
        rewards.append(ret)
        steps_list.append(steps)
        if (ep + 1) % max(1, args.episodes // 10) == 0:
            print(f"  ep {ep+1:4d}/{args.episodes}  "
                  f"wins={wins}/{ep+1}  "
                  f"win_rate={100*wins/(ep+1):.1f}%  "
                  f"last_return={ret:.1f}  steps={steps}")

    print(f"\n{'='*55}")
    print(f"Result: {wins}/{args.episodes} wins  ({100*wins/args.episodes:.1f}%)")
    print(f"  mean return : {np.mean(rewards):.2f}")
    print(f"  std return  : {np.std(rewards):.2f}")
    print(f"  mean steps  : {np.mean(steps_list):.1f}")
    print(f"{'='*55}")
    env.close()


if __name__ == "__main__":
    main()
