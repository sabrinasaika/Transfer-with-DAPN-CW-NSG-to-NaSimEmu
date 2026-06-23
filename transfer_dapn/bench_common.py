"""Shared evaluation utilities for the DAPN transfer benchmark.

Used by the standalone eval scripts (4_eval_nasim.py, 6_eval_emulator.py) and
by benchmark.py to produce the headline transfer figure.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


@dataclass
class EvalResult:
    """Summary statistics for one (policy, domain) condition."""

    condition: str                      # short id, e.g. "dapn_sim"
    label: str                          # human label for plots, e.g. "DAPN -> Sim"
    episodes: int
    wins: int
    win_rate: float                     # in [0, 1]
    win_rate_ci95: float                # +/- half-width (Wilson) in [0, 1]
    mean_return: float
    std_return: float
    mean_steps: float
    wall_time_s: float
    returns: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def wilson_halfwidth(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval half-width for a binomial proportion."""
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    # report symmetric half-width around p (clamped)
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (hi - lo) / 2.0


class RandomPolicy:
    """Drop-in replacement for an SB3 model exposing .predict()."""

    def __init__(self, action_space, seed: int = 0):
        self._space = action_space
        self._rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        return int(self._rng.integers(0, self._space.n)), None


def run_episode(model, env, seed=None, deterministic: bool = True):
    """Run one episode; return (win, total_return, steps)."""
    reset_kw = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kw)
    total_r = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_r += reward
        steps += 1
        if terminated or truncated:
            win = bool(info.get("win", False) or terminated)
            return win, float(total_r), steps


def evaluate(model, env, condition: str, label: str, episodes: int,
             seed: int = 0, deterministic: bool = True,
             progress_every: int = 0, note: str = "") -> EvalResult:
    """Evaluate a policy on an env for N episodes and aggregate statistics."""
    rng = np.random.default_rng(seed)
    wins = 0
    returns: list[float] = []
    steps_list: list[int] = []
    t0 = time.time()
    for ep in range(episodes):
        win, ret, steps = run_episode(
            model, env, seed=int(rng.integers(0, 2**31)),
            deterministic=deterministic)
        wins += int(win)
        returns.append(ret)
        steps_list.append(steps)
        if progress_every and (ep + 1) % progress_every == 0:
            print(f"    [{condition}] ep {ep+1}/{episodes}  "
                  f"wins={wins}/{ep+1}  win_rate={100*wins/(ep+1):.1f}%",
                  flush=True)
    wall = time.time() - t0
    return EvalResult(
        condition=condition,
        label=label,
        episodes=episodes,
        wins=wins,
        win_rate=wins / episodes if episodes else 0.0,
        win_rate_ci95=wilson_halfwidth(wins, episodes),
        mean_return=float(np.mean(returns)) if returns else 0.0,
        std_return=float(np.std(returns)) if returns else 0.0,
        mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
        wall_time_s=wall,
        returns=returns,
        steps=steps_list,
        note=note,
    )


def write_json(path: str | Path, payload) -> None:
    """Write a dict, an EvalResult, or a list of EvalResults to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, EvalResult):
        data = payload.as_dict()
    elif isinstance(payload, list):
        data = [p.as_dict() if isinstance(p, EvalResult) else p for p in payload]
    else:
        data = payload
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
