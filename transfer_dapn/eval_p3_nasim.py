"""Zero-shot evaluation of p3 (NSG-trained DAPN policy) in NSG-aligned NaSim.

Usage:
  conda run -n cyberwheel python eval_p3_nasim.py --episodes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3 import PPO
from envs.kc_envs import make_nasim_nsg_kc_env
from envs.dapn_encoder_wrapper import DAPNEncoderWrapper
from policy_compat import load_ppo_policy_weights

_ROOT = Path(__file__).resolve().parent
DEFAULT_ENCODER = str(_ROOT / "artifacts/models/dapn_encoder_nsg.pt.best.pt")
DEFAULT_POLICY  = str(_ROOT / "artifacts/policies/nsg_dapn_policy_final.zip")


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
    ap = argparse.ArgumentParser(description="Zero-shot p3 → NaSim-NSG evaluation")
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = make_nasim_nsg_kc_env()
    env = DAPNEncoderWrapper(base, args.encoder, mask_is_target=True)
    model = load_ppo_policy_weights(args.policy, env)

    print(f"\n{'='*55}")
    print("Zero-shot transfer: p3 (NSG-trained) → NaSim (nasim_two_subnet)")
    print(f"  policy   : {args.policy}")
    print(f"  encoder  : {args.encoder}")
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
            print(f"  ep {ep+1:4d}/{args.episodes}  wins={wins}/{ep+1}  "
                  f"win_rate={100*wins/(ep+1):.1f}%  return={ret:.1f}  steps={steps}")

    print(f"\n{'='*55}")
    print(f"Result: {wins}/{args.episodes} wins  ({100*wins/args.episodes:.1f}%)")
    print(f"  mean return : {np.mean(rewards):.2f}")
    print(f"  mean steps  : {np.mean(steps_list):.1f}")
    print(f"{'='*55}")
    env.close()


if __name__ == "__main__":
    main()
