"""Experiment 1: How similar is p1 (CW→NaSim via DAPN) to pe (NaSim-native)?

Method — Shadow rollout
-----------------------
A single NaSim KC+ctx env (44-D) drives each episode.
At every step both policies receive the SAME 73-D DAPN-encoded observation:

  encoded_obs = encoder( kc_ctx_obs )    →  73-D  [latent | is_target | ctx]
  probs_p1    = p1( encoded_obs )        →  Discrete(8) probs
  probs_pe    = pe( encoded_obs )        →  Discrete(8) probs

The env advances with p1's action (shadow). A separate pe-only rollout with
the same seed captures pe's standalone performance.

Similarity metrics
------------------
  action_agreement_rate : fraction of steps where argmax(p1) == argmax(pe)
  mean_js_divergence    : Jensen-Shannon divergence between action-prob vectors
  mean_kl_divergence    : KL(p1‖pe) per step
  mean_cosine_sim       : cosine similarity of action-prob vectors
  return_correlation    : Pearson r between per-episode returns of p1 and pe
  win_rate_p1 / pe      : raw performance

Usage
-----
  cd transfer_dapn
  conda run -n cyberwheel python experiment_1_similarity.py \\
      --encoder artifacts/models/dapn_encoder_kc7.pt.best.pt \\
      --policy-p1 artifacts/policies/cw_dapn_policy/best_model.zip \\
      --policy-pe artifacts/policies/nasim_kc_invariant/best_model.zip \\
      --episodes 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cyberwheel"))

from stable_baselines3 import PPO

from envs.kc_envs import make_nasim_kc_env
from models.encoder import load_encoder, encode_obs

_ROOT = Path(__file__).resolve().parent
DEFAULT_ENCODER   = str(_ROOT / "artifacts/models/dapn_encoder_kc7.pt.best.pt")
DEFAULT_POLICY_P1 = str(_ROOT / "artifacts/policies/cw_dapn_policy/best_model.zip")
DEFAULT_POLICY_PE = str(_ROOT / "artifacts/policies/nasim_kc_invariant/best_model.zip")
DEFAULT_OUT       = str(_ROOT / "artifacts/results/experiment_1_similarity.json")

ACTION_NAMES = ["entry", "target", "pivot1", "pivot2", "extra1", "extra2", "extra3", "noop"]


# ── Policy utilities ──────────────────────────────────────────────────────────

def get_action_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.cpu().numpy()[0]
    return probs


# ── Divergence helpers ────────────────────────────────────────────────────────

def _safe(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = p + eps
    return p / p.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = _safe(p), _safe(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = _safe(p), _safe(q)
    return float(np.sum(p * np.log(p / q)))


def cosine_sim(p: np.ndarray, q: np.ndarray) -> float:
    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)
    if norm_p < 1e-12 or norm_q < 1e-12:
        return 0.0
    return float(np.dot(p, q) / (norm_p * norm_q))


# ── Episode runners ───────────────────────────────────────────────────────────

def shadow_episode(model_p1: PPO, model_pe: PPO, encoder, env,
                   seed: int, device: str = "cpu"):
    """Drive with p1; pe shadows at every step. Both see same encoded obs."""
    obs_raw, _ = env.reset(seed=seed)

    steps = 0
    total_r    = 0.0
    agreements = []
    js_divs    = []
    kl_divs    = []
    cos_sims   = []
    actions_p1 = []
    actions_pe = []
    probs_p1_all = []
    probs_pe_all = []
    terminated = truncated = False

    while not (terminated or truncated):
        enc = encode_obs(encoder, obs_raw, device=device)  # 73-D — same for both

        probs_p1 = get_action_probs(model_p1, enc)
        probs_pe = get_action_probs(model_pe, enc)

        action_p1 = int(np.argmax(probs_p1))
        action_pe = int(np.argmax(probs_pe))

        actions_p1.append(action_p1)
        actions_pe.append(action_pe)
        probs_p1_all.append(probs_p1.tolist())
        probs_pe_all.append(probs_pe.tolist())
        agreements.append(int(action_p1 == action_pe))
        js_divs.append(js_divergence(probs_p1, probs_pe))
        kl_divs.append(kl_divergence(probs_p1, probs_pe))
        cos_sims.append(cosine_sim(probs_p1, probs_pe))

        obs_raw, reward, terminated, truncated, info = env.step(action_p1)
        total_r += reward
        steps += 1

    win_p1 = bool(info.get("win", False) or terminated)
    return {
        "win_p1":                win_p1,
        "return_p1":             float(total_r),
        "steps_p1":              steps,
        "action_agreement_rate": float(np.mean(agreements)),
        "mean_js_divergence":    float(np.mean(js_divs)),
        "mean_kl_divergence":    float(np.mean(kl_divs)),
        "mean_cosine_sim":       float(np.mean(cos_sims)),
        "actions_p1":            actions_p1,
        "actions_pe":            actions_pe,
        "probs_p1":              probs_p1_all,
        "probs_pe":              probs_pe_all,
    }


def pe_episode(model_pe: PPO, encoder, env, seed: int, device: str = "cpu"):
    """Standalone pe rollout. Returns (win, return, steps)."""
    obs_raw, _ = env.reset(seed=seed)
    total_r = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        enc    = encode_obs(encoder, obs_raw, device=device)
        probs  = get_action_probs(model_pe, enc)
        action = int(np.argmax(probs))
        obs_raw, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        steps += 1

    win = bool(info.get("win", False) or terminated)
    return win, float(total_r), steps


# ── Aggregate statistics ──────────────────────────────────────────────────────

def wilson_ci(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom  = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (min(1.0, centre + margin) - max(0.0, centre - margin)) / 2.0


def pearson_r(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    denom  = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float(np.dot(xm, ym) / denom) if denom > 1e-12 else float("nan")


def aggregate(ep_results: list[dict]) -> dict:
    wins_p1    = sum(e["win_p1"] for e in ep_results)
    wins_pe    = sum(e["win_pe"] for e in ep_results)
    n          = len(ep_results)
    returns_p1 = [e["return_p1"] for e in ep_results]
    returns_pe = [e["return_pe"] for e in ep_results]
    return {
        "n_episodes":            n,
        "win_rate_p1":           wins_p1 / n,
        "win_rate_p1_ci95":      wilson_ci(wins_p1, n),
        "win_rate_pe":           wins_pe / n,
        "win_rate_pe_ci95":      wilson_ci(wins_pe, n),
        "mean_return_p1":        float(np.mean(returns_p1)),
        "std_return_p1":         float(np.std(returns_p1)),
        "mean_return_pe":        float(np.mean(returns_pe)),
        "std_return_pe":         float(np.std(returns_pe)),
        "return_correlation":    pearson_r(returns_p1, returns_pe),
        "action_agreement_rate": float(np.mean([e["action_agreement_rate"] for e in ep_results])),
        "mean_js_divergence":    float(np.mean([e["mean_js_divergence"]    for e in ep_results])),
        "mean_kl_divergence":    float(np.mean([e["mean_kl_divergence"]    for e in ep_results])),
        "mean_cosine_sim":       float(np.mean([e["mean_cosine_sim"]       for e in ep_results])),
    }


def global_action_distributions(ep_results: list[dict]) -> dict:
    all_probs_p1, all_probs_pe = [], []
    all_a_p1,    all_a_pe     = [], []
    for e in ep_results:
        all_probs_p1.extend(e["probs_p1"])
        all_probs_pe.extend(e["probs_pe"])
        all_a_p1.extend(e["actions_p1"])
        all_a_pe.extend(e["actions_pe"])

    n_act       = len(ACTION_NAMES)
    freq_p1     = np.bincount(all_a_p1, minlength=n_act) / max(len(all_a_p1), 1)
    freq_pe     = np.bincount(all_a_pe, minlength=n_act) / max(len(all_a_pe), 1)
    mean_p1     = np.mean(all_probs_p1, axis=0) if all_probs_p1 else np.zeros(n_act)
    mean_pe     = np.mean(all_probs_pe, axis=0) if all_probs_pe else np.zeros(n_act)
    global_js   = js_divergence(mean_p1, mean_pe)
    return {
        "action_names":  ACTION_NAMES,
        "freq_p1":       freq_p1.tolist(),
        "freq_pe":       freq_pe.tolist(),
        "mean_prob_p1":  mean_p1.tolist(),
        "mean_prob_pe":  mean_pe.tolist(),
        "global_js_div": global_js,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(agg: dict, global_dist: dict) -> None:
    print(f"\n{'='*60}")
    print("Experiment 1 — Policy Similarity Summary")
    print(f"{'='*60}")
    print(f"  {'Policy':<22} {'Win%':>7}  {'±95%':>7}  {'Mean return':>12}")
    print(f"  {'-'*53}")
    print(f"  {'p1 (CW→NaSim DAPN)':<22} "
          f"{100*agg['win_rate_p1']:>6.1f}%  "
          f"±{100*agg['win_rate_p1_ci95']:>5.1f}%  "
          f"{agg['mean_return_p1']:>12.2f}")
    print(f"  {'pe (NaSim native)':<22} "
          f"{100*agg['win_rate_pe']:>6.1f}%  "
          f"±{100*agg['win_rate_pe_ci95']:>5.1f}%  "
          f"{agg['mean_return_pe']:>12.2f}")
    print(f"\nBehavioural Similarity (p1 shadow vs pe):")
    print(f"  Action agreement   : {100*agg['action_agreement_rate']:.1f}%")
    print(f"  Mean JS divergence : {agg['mean_js_divergence']:.4f}")
    print(f"  Mean KL(p1‖pe)    : {agg['mean_kl_divergence']:.4f}")
    print(f"  Cosine similarity  : {agg['mean_cosine_sim']:.4f}")
    print(f"  Return correlation : {agg['return_correlation']:.4f}")
    print(f"  Global JS          : {global_dist['global_js_div']:.4f}")
    print(f"\nAction frequencies:")
    print(f"  {'Action':<12} {'p1':>8} {'pe':>8}")
    print(f"  {'-'*30}")
    for name, fp1, fpe in zip(global_dist["action_names"],
                               global_dist["freq_p1"],
                               global_dist["freq_pe"]):
        print(f"  {name:<12} {fp1:>7.3f}  {fpe:>7.3f}")
    print(f"{'='*60}\n")


def make_markdown_table(agg: dict) -> str:
    return "\n".join([
        "## Experiment 1 — CW→NaSim DAPN (p1) vs NaSim-native (pe)",
        "",
        "| Policy | Win% | ±95% CI | Mean return | Std |",
        "|--------|-----:|--------:|------------:|----:|",
        f"| p1 (CW→NaSim DAPN) | {100*agg['win_rate_p1']:.1f}% | "
        f"±{100*agg['win_rate_p1_ci95']:.1f}% | "
        f"{agg['mean_return_p1']:.2f} | {agg['std_return_p1']:.2f} |",
        f"| pe (NaSim native) | {100*agg['win_rate_pe']:.1f}% | "
        f"±{100*agg['win_rate_pe_ci95']:.1f}% | "
        f"{agg['mean_return_pe']:.2f} | {agg['std_return_pe']:.2f} |",
        "",
        "| Similarity metric | Value |",
        "|-------------------|------:|",
        f"| Action agreement | {100*agg['action_agreement_rate']:.1f}% |",
        f"| Mean JS divergence | {agg['mean_js_divergence']:.4f} |",
        f"| Mean KL(p1‖pe) | {agg['mean_kl_divergence']:.4f} |",
        f"| Mean cosine similarity | {agg['mean_cosine_sim']:.4f} |",
        f"| Return correlation (r) | {agg['return_correlation']:.4f} |",
        f"| N episodes | {agg['n_episodes']} |",
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Experiment 1: p1 (CW DAPN) vs pe (NaSim native) similarity")
    ap.add_argument("--encoder",    default=DEFAULT_ENCODER)
    ap.add_argument("--policy-p1",  default=DEFAULT_POLICY_P1)
    ap.add_argument("--policy-pe",  default=DEFAULT_POLICY_PE)
    ap.add_argument("--episodes",   type=int, default=200)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--out",        default=DEFAULT_OUT)
    ap.add_argument("--no-per-ep",  action="store_true",
                    help="Drop per-episode trajectory data from JSON")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print("Experiment 1: p1 (CW DAPN) vs pe (NaSim native)")
    print(f"  encoder   : {args.encoder}")
    print(f"  policy p1 : {args.policy_p1}")
    print(f"  policy pe : {args.policy_pe}")
    print(f"  episodes  : {args.episodes}")
    print(f"{'='*60}\n")

    encoder  = load_encoder(args.encoder, device=args.device)
    model_p1 = PPO.load(args.policy_p1)
    model_pe = PPO.load(args.policy_pe)

    # Both policies must have the same obs/action space (73-D, Discrete(8))
    for name, m in [("p1", model_p1), ("pe", model_pe)]:
        obs_n = m.observation_space.shape[0]
        act_n = m.action_space.n
        if obs_n != 73:
            print(f"WARNING: {name} expects {obs_n}-D obs (expected 73)")
        if act_n != 8:
            print(f"WARNING: {name} has {act_n} actions (expected 8)")

    env = make_nasim_kc_env()    # produces 44-D KC+ctx obs; encoder applied manually
    rng = np.random.default_rng(args.seed)

    ep_results = []
    wins_p1 = wins_pe = 0
    t0 = time.time()

    print("Running shadow rollouts…")
    for ep in range(args.episodes):
        ep_seed = int(rng.integers(0, 2**31))

        shadow = shadow_episode(model_p1, model_pe, encoder, env,
                                seed=ep_seed, device=args.device)
        win_pe, ret_pe, steps_pe = pe_episode(model_pe, encoder, env,
                                              seed=ep_seed, device=args.device)

        rec = {**shadow, "win_pe": win_pe, "return_pe": ret_pe, "steps_pe": steps_pe}
        if args.no_per_ep:
            for k in ("probs_p1", "probs_pe", "actions_p1", "actions_pe"):
                rec.pop(k, None)

        ep_results.append(rec)
        wins_p1 += int(shadow["win_p1"])
        wins_pe += int(win_pe)

        if (ep + 1) % max(1, args.episodes // 10) == 0:
            n = ep + 1
            print(f"  ep {n:4d}/{args.episodes}  "
                  f"p1={100*wins_p1/n:.0f}%  pe={100*wins_pe/n:.0f}%  "
                  f"agree={100*np.mean([e['action_agreement_rate'] for e in ep_results]):.0f}%  "
                  f"JS={np.mean([e['mean_js_divergence'] for e in ep_results]):.3f}",
                  flush=True)

    env.close()

    agg         = aggregate(ep_results)
    global_dist = global_action_distributions(ep_results)
    print_summary(agg, global_dist)

    payload = {
        "experiment": "1_cw_dapn_vs_nasim_native",
        "config": {
            "encoder":   args.encoder,
            "policy_p1": args.policy_p1,
            "policy_pe": args.policy_pe,
            "episodes":  args.episodes,
            "seed":      args.seed,
        },
        "aggregate":                 agg,
        "global_action_distributions": global_dist,
        "wall_time_s":               time.time() - t0,
        "episodes":                  ep_results if not args.no_per_ep else [],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    md = out.with_suffix(".md")
    md.write_text(make_markdown_table(agg) + "\n")
    print(f"Results → {out}")
    print(f"Markdown → {md}")
    print(f"\nNext: python plot_experiment_1.py --in {out}")


if __name__ == "__main__":
    main()
