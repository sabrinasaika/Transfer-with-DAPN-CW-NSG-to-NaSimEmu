"""DAPN transfer benchmark harness.

Runs the policy under several (policy, domain) conditions and writes a single
results JSON consumed by plot_results.py to produce the headline figure.

Conditions
----------
  nasim     : policy on NaSim observations (no translator)        -- simulation reference
  dapn_sim  : policy + translator G on NaSim simulation           (zero-shot transfer)
  dapn_emu  : policy + translator G on the live NASimEmu emulator (zero-shot transfer)

The nasim/dapn_sim conditions are fast (run many episodes).
The emulator condition hits real Metasploit (slow) -- keep --episodes-emu small,
and it is skipped unless the Vagrant network + msfrpcd are reachable.

Usage
-----
  cd transfer_dapn
  conda activate cyberwheel

  # fast (no emulator):
  PYTHONPATH=../src:../cyberwheel:. python benchmark.py \
      --episodes-sim 200 --conditions nasim dapn_sim

  # full (with live emulator -- VMs + msfrpcd must be up, run alone):
  PYTHONUNBUFFERED=1 PYTHONPATH=../src:../cyberwheel:. python benchmark.py \
      --episodes-sim 200 --episodes-emu 5 --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cyberwheel"))

from stable_baselines3 import PPO

from bench_common import evaluate, write_json

_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = str(_ROOT / "artifacts/policies/cw_kc_policy/best/best_model.zip")
DEFAULT_TRANSLATOR = str(_ROOT / "artifacts/models/dapn_kc_translator.pt.best.pt")
DEFAULT_OUT = str(_ROOT / "artifacts/results/benchmark.json")

ALL_CONDITIONS = ["nasim", "dapn_sim", "dapn_emu"]

LABELS = {
    "nasim": "NaSim",
    "dapn_sim": "DAPN -> Sim",
    "dapn_emu": "DAPN -> Emulator",
}


def _build_dapn_sim(translator):
    from envs.kc_envs import make_nasim_kc_env
    from envs.dapn_translator_wrapper import DAPNTranslatorWrapper
    return DAPNTranslatorWrapper(make_nasim_kc_env(), translator)


def _build_nasim():
    # Feed RAW NaSim KC+ctx (26-D) straight to the policy (no domain adaptation).
    from envs.kc_envs import make_nasim_kc_env
    return make_nasim_kc_env()


def _build_dapn_emu(translator):
    from envs.kc_envs import make_emu_kc_env
    from envs.dapn_translator_wrapper import DAPNTranslatorWrapper
    return DAPNTranslatorWrapper(make_emu_kc_env(), translator)


def run_condition(cond, model, args):
    """Build env + (policy/random), evaluate, return EvalResult or None on skip."""
    try:
        if cond == "nasim":
            env = _build_nasim()
            res = evaluate(model, env, cond, LABELS[cond], args.episodes_sim,
                           seed=args.seed, progress_every=max(1, args.episodes_sim // 5),
                           note="Policy on NaSim observations (no translator)")
        elif cond == "dapn_sim":
            env = _build_dapn_sim(args.translator)
            res = evaluate(model, env, cond, LABELS[cond], args.episodes_sim,
                           seed=args.seed, progress_every=max(1, args.episodes_sim // 5),
                           note="DAPN translator transfer on NaSim simulation")
        elif cond == "dapn_emu":
            env = _build_dapn_emu(args.translator)
            res = evaluate(model, env, cond, LABELS[cond], args.episodes_emu,
                           seed=args.seed, progress_every=1,
                           note="DAPN transfer on live emulator (MSF)")
        else:
            print(f"  unknown condition '{cond}', skipping")
            return None
        try:
            env.close()
        except Exception:
            pass
        return res
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] condition '{cond}' failed: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser(description="DAPN transfer benchmark harness")
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--translator", default=DEFAULT_TRANSLATOR)
    ap.add_argument("--episodes-sim", type=int, default=200,
                    help="Episodes for the fast simulation conditions (nasim, dapn_sim)")
    ap.add_argument("--episodes-emu", type=int, default=5,
                    help="Episodes for the slow live-emulator condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help=f"Subset of {ALL_CONDITIONS}; default = all except dapn_emu")
    ap.add_argument("--all", action="store_true",
                    help="Run every condition including the slow emulator")
    args = ap.parse_args()

    if args.conditions:
        conditions = args.conditions
    elif args.all:
        conditions = ALL_CONDITIONS
    else:
        conditions = [c for c in ALL_CONDITIONS if c != "dapn_emu"]

    print(f"\n{'='*60}")
    print("DAPN transfer benchmark")
    print(f"  policy     : {args.policy}")
    print(f"  translator : {args.translator}")
    print(f"  conditions : {conditions}")
    print(f"  episodes   : sim={args.episodes_sim}  emu={args.episodes_emu}")
    print(f"{'='*60}\n")

    model = PPO.load(args.policy)

    results = []
    for cond in conditions:
        print(f"--- {cond} ---", flush=True)
        res = run_condition(cond, model, args)
        if res is not None:
            results.append(res)
            print(f"    win_rate={100*res.win_rate:.1f}% (+/-{100*res.win_rate_ci95:.1f}) "
                  f"mean_return={res.mean_return:.2f}  n={res.episodes}  "
                  f"{res.wall_time_s:.1f}s\n", flush=True)

    write_json(args.out, results)

    print(f"\n{'='*60}\nSummary")
    print(f"{'condition':<16}{'win%':>8}{'+/-95%':>9}{'mean_ret':>10}{'n':>6}")
    for r in results:
        print(f"{r.condition:<16}{100*r.win_rate:>7.1f}%{100*r.win_rate_ci95:>8.1f}%"
              f"{r.mean_return:>10.2f}{r.episodes:>6}")
    print(f"{'='*60}")
    print(f"\nNext: python plot_results.py --in {args.out}")


if __name__ == "__main__":
    main()
