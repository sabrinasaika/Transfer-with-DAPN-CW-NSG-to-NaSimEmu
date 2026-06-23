"""
Step 4 — Evaluate DAPN transfer on NaSim (simulation).

Paper-faithful DAPN (Zhao et al., arXiv:2003.08626):
  NaSim KC+ctx obs → translator G → CW-like obs → frozen CW policy π

Usage:
  cd transfer_dapn
  conda run -n cyberwheel python 4_eval_nasim.py --episodes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3 import PPO

from envs.kc_envs import make_nasim_kc_env
from envs.dapn_translator_wrapper import DAPNTranslatorWrapper

_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = str(_ROOT / "artifacts/policies/cw_kc_policy/best/best_model.zip")
DEFAULT_TRANSLATOR = str(_ROOT / "artifacts/models/dapn_kc_translator.pt.best.pt")


def build_env(translator: str):
    env = make_nasim_kc_env()
    return DAPNTranslatorWrapper(env, translator)


def run_episode(model, env, deterministic: bool = True):
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_r += reward
        steps += 1
        if terminated or truncated:
            win = bool(info.get("win", False) or terminated)
            return win, total_r, steps


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate DAPN (translator G) transfer on NaSim")
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="CW-trained SB3 .zip policy")
    ap.add_argument(
        "--translator", default=DEFAULT_TRANSLATOR,
        help="DAPN domain translator G (.pt)",
    )
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None,
                    help="If set, write summary statistics to this JSON path")
    args = ap.parse_args()

    env = build_env(args.translator)
    model = PPO.load(args.policy)

    exp_obs = env.observation_space.shape[0]
    pol_obs = model.observation_space.shape[0]
    if exp_obs != pol_obs:
        print(f"WARNING: env obs dim {exp_obs} != policy obs dim {pol_obs}")

    print(f"\n{'='*60}")
    print(f"DAPN NaSim eval  episodes={args.episodes}")
    print(f"  policy     : {args.policy}")
    print(f"  translator : {args.translator}")
    print(f"  obs dim    : {exp_obs} (policy expects {pol_obs})")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(args.seed)
    wins = 0
    rewards = []
    for ep in range(args.episodes):
        env.reset(seed=int(rng.integers(0, 2**31)))
        win, ret, steps = run_episode(model, env)
        wins += int(win)
        rewards.append(ret)
        if (ep + 1) % max(1, args.episodes // 10) == 0:
            print(f"  ep {ep+1:4d}/{args.episodes}  wins={wins}/{ep+1}  "
                  f"win_rate={100*wins/(ep+1):.1f}%  last_return={ret:.1f}  steps={steps}")

    print(f"\nResults: {wins}/{args.episodes} wins ({100*wins/args.episodes:.1f}%)")
    print(f"  mean return : {np.mean(rewards):.2f}")
    print(f"  std return  : {np.std(rewards):.2f}")

    if args.json:
        from bench_common import EvalResult, wilson_halfwidth, write_json
        result = EvalResult(
            condition="dapn_sim",
            label="DAPN -> Sim",
            episodes=args.episodes,
            wins=wins,
            win_rate=wins / args.episodes if args.episodes else 0.0,
            win_rate_ci95=wilson_halfwidth(wins, args.episodes),
            mean_return=float(np.mean(rewards)) if rewards else 0.0,
            std_return=float(np.std(rewards)) if rewards else 0.0,
            mean_steps=0.0,
            wall_time_s=0.0,
            returns=[float(r) for r in rewards],
            note="DAPN translator transfer on NaSim simulation",
        )
        write_json(args.json, result)

    env.close()


if __name__ == "__main__":
    main()
