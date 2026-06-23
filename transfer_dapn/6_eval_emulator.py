"""
Step 6 — Evaluate DAPN transfer on the live NASimEmu emulator (Vagrant + MSF).

Paper-faithful DAPN (Zhao et al., arXiv:2003.08626):
  emulator KC+ctx obs → translator G → CW-like obs → frozen CW policy π

Prerequisites:
  1. ./setup_vagrant.sh scenarios/fixed_dmz_one_subnet_4host.v2.yaml
  2. cd vagrant && vagrant up
  3. msfrpcd reachable on localhost:55553 (attacker VM)

Usage:
  cd transfer_dapn
  conda run -n cyberwheel python 6_eval_emulator.py --episodes 1

Note: each emulator step runs real Metasploit actions (~2s sleep per step).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3 import PPO

from envs.kc_envs import make_emu_kc_env
from envs.dapn_translator_wrapper import DAPNTranslatorWrapper
from envs.kill_chain import HOST_NAMES, SLOT_ORDER

_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = str(_ROOT / "artifacts/policies/cw_kc_policy/best/best_model.zip")
DEFAULT_TRANSLATOR = str(_ROOT / "artifacts/models/dapn_kc_translator.pt.best.pt")

_NOOP = 4


def build_env(translator: str):
    env = make_emu_kc_env()
    return DAPNTranslatorWrapper(env, translator)


def _unwrap_kc_wrapper(env):
    cur = env
    while cur is not None:
        if cur.__class__.__name__ == "NaSimKillChainWrapper":
            return cur
        cur = getattr(cur, "env", None)
    return None


def _slot_label(action: int) -> str:
    if action == _NOOP:
        return "noop"
    if 0 <= action < len(HOST_NAMES):
        return HOST_NAMES[action]
    return f"slot_{action}"


def _flat_action_name(kc_env, flat: int) -> str:
    try:
        a = kc_env._env.action_space.get_action(flat)
        return str(a)
    except Exception:
        return f"flat_{flat}"


def _host_state_lines(kc_env) -> list[str]:
    lines = []
    for addr, host in kc_env._env.current_state.hosts:
        lines.append(
            f"      {addr}  disc={int(host.discovered)}  "
            f"reach={int(host.reachable)}  access={int(host.access)}  "
            f"comp={int(host.compromised)}  value={host.value:.0f}"
        )
    return lines


def run_episode(model, env, ep: int = 1, seed: int | None = None,
                verbose: bool = True, deterministic: bool = True):
    reset_kw = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kw)
    kc = _unwrap_kc_wrapper(env)
    total_r = 0.0
    steps = 0

    if verbose:
        print(f"\n--- episode {ep} reset ---")
        if kc is not None:
            for line in _host_state_lines(kc):
                print(line)
        print(f"  obs (policy input): {np.round(obs, 2)}")

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)

        if verbose and kc is not None:
            flat = kc._translate(action)
            print(f"\n  step {steps + 1}  policy_slot={action} ({_slot_label(action)})")
            print(f"           msf_action={_flat_action_name(kc, flat)}")
            if kc._last_step_info:
                ok = kc._last_step_info.get("success", "?")
                print(f"           prev_msf_ok={ok}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        steps += 1

        if verbose:
            msf_ok = kc._last_step_info.get("success") if kc is not None else None
            msf_str = f"  msf_ok={msf_ok}" if msf_ok is not None else ""
            print(f"           reward={reward:.1f}  total={total_r:.1f}  "
                  f"win={info.get('win', False)}{msf_str}")
            if kc is not None:
                for line in _host_state_lines(kc):
                    print(line)

        if terminated or truncated:
            win = bool(info.get("win", False) or terminated)
            if verbose:
                reason = "goal" if win else ("step_limit" if truncated else "done")
                print(f"\n  episode {ep} ended: {reason}  steps={steps}  return={total_r:.1f}")
            return win, total_r, steps


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate DAPN (translator G) on live NASimEmu emulator")
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="CW-trained SB3 .zip policy")
    ap.add_argument(
        "--translator", default=DEFAULT_TRANSLATOR,
        help="DAPN domain translator G (.pt)",
    )
    ap.add_argument("--episodes", type=int, default=5,
                    help="Keep small — each step hits real MSF (~2s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-step trace (episode summary only)")
    ap.add_argument("--json", default=None,
                    help="If set, write summary statistics to this JSON path")
    args = ap.parse_args()

    env = build_env(args.translator)
    model = PPO.load(args.policy)
    verbose = not args.quiet

    exp_obs = env.observation_space.shape[0]
    pol_obs = model.observation_space.shape[0]
    if exp_obs != pol_obs:
        print(f"WARNING: env obs dim {exp_obs} != policy obs dim {pol_obs}")

    print(f"\n{'='*60}")
    print(f"DAPN emulator eval  episodes={args.episodes}")
    print(f"  policy     : {args.policy}")
    print(f"  translator : {args.translator}")
    print(f"  obs dim    : {exp_obs} (policy expects {pol_obs})")
    print(f"  slots      : {list(zip(HOST_NAMES, SLOT_ORDER))}")
    print(f"{'='*60}")

    import time as _time
    rng = np.random.default_rng(args.seed)
    wins = 0
    rewards = []
    steps_list = []
    t0 = _time.time()
    for ep in range(args.episodes):
        win, ret, steps = run_episode(
            model, env,
            ep=ep + 1,
            seed=int(rng.integers(0, 2**31)),
            verbose=verbose,
        )
        wins += int(win)
        rewards.append(ret)
        steps_list.append(steps)
        tag = "WIN " if win else "lose"
        print(f"\n  ep {ep+1:3d}/{args.episodes}  {tag}  return={ret:7.1f}  steps={steps}")
    wall = _time.time() - t0

    print(f"\nResults: {wins}/{args.episodes} wins ({100*wins/args.episodes:.1f}%)")
    print(f"  mean return : {np.mean(rewards):.2f}")
    print(f"  std return  : {np.std(rewards):.2f}")

    if args.json:
        from bench_common import EvalResult, wilson_halfwidth, write_json
        result = EvalResult(
            condition="dapn_emu",
            label="DAPN -> Emulator",
            episodes=args.episodes,
            wins=wins,
            win_rate=wins / args.episodes if args.episodes else 0.0,
            win_rate_ci95=wilson_halfwidth(wins, args.episodes),
            mean_return=float(np.mean(rewards)) if rewards else 0.0,
            std_return=float(np.std(rewards)) if rewards else 0.0,
            mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
            wall_time_s=wall,
            returns=[float(r) for r in rewards],
            steps=[int(s) for s in steps_list],
            note="DAPN translator transfer on live NASimEmu emulator (MSF)",
        )
        write_json(args.json, result)

    env.close()


if __name__ == "__main__":
    main()
