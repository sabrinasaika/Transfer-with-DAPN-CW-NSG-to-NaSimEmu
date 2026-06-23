"""
CWNativeWrapper — CyberwheelRL with NaSim-aligned reward/goal/topology.

NaSim alignment:
  - Topology : fixed-dmz-two-subnet (7 hosts, matches NaSim fixed_dmz_two_subnet)
  - Goal     : compromise target_drupal (= NaSim's sensitive host (2,0))
  - Reward   : -1 per step (all actions), +100 bonus when goal achieved
  - Terminal : episode ends immediately on goal achievement

Observation: native RedObservation obs_vec
  - max_num_hosts = 5  →  obs_size = 5×7 + 1 = 36-D float32 in [0, 1]

Action: Discrete(30) = 5 hosts × 6 kill-chain actions
  Actions for undiscovered hosts are silently mapped to pingsweep on the
  entry host so the policy always takes a meaningful step (no free noop).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cyberwheel"))

_ENV_CFG   = "fixed_dmz_two_subnet_transfer.yaml"
_MAX_HOSTS  = 7
_HOST_FEAT  = 7
_STANDALONE = 1
_OBS_SIZE   = _MAX_HOSTS * _HOST_FEAT + _STANDALONE   # 50
_MAX_ATTR   = 4
_ACT_SIZE   = _MAX_HOSTS * 6                          # 42

# NaSim-style reward constants (match fixed_dmz_one_subnet_4host.v2.yaml)
_STEP_COST    = -1.0    # cost paid on every step (success or failure)
_GOAL_REWARD  = 100.0   # bonus when impact achieved on target host
_MAX_STEPS    = 200     # truncate at 200 steps (matches NaSim step_limit)


def _make_cw_env(scenario: str = "two_subnet") -> gym.Env:
    from envs.scenario_cfg import get_scenario
    cfg = get_scenario(scenario)
    env_cfg = cfg.cw_env_yaml
    max_hosts = cfg.cw_max_hosts
    import yaml
    from types import SimpleNamespace
    from cyberwheel.network.network_base import Network
    from cyberwheel.cyberwheel_envs.cyberwheel_rl import CyberwheelRL
    from cyberwheel.utils.get_service_map import get_service_map
    from importlib.resources import files

    def _load_yaml(pkg_path, fallback):
        try:
            with files(pkg_path[0]).joinpath(pkg_path[1]).open("r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            with open(fallback) as f:
                return yaml.safe_load(f) or {}

    cw_root = Path(__file__).resolve().parents[2] / "cyberwheel" / "cyberwheel"
    cfg = _load_yaml(
        ("cyberwheel.data.configs.environment", env_cfg),
        cw_root / "data" / "configs" / "environment" / env_cfg,
    )

    def _to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_ns(v) for v in obj]
        return obj

    args = _to_ns(cfg)

    # Force valid_targets="leader" so only target_drupal gives impact reward.
    # Must set BEFORE the hasattr defaults below since the yaml already has "all".
    args.valid_targets = "leader"

    for attr, val in [
        ("red_reward_function",        "reward_decoy_hits"),
        ("blue_reward_function",       "reward_red_delay"),
        ("network_size_compatibility", "small"),
        ("max_decoys",                 0),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, val)

    args.max_num_hosts = _MAX_HOSTS

    net_cfg_name = getattr(args, "network_config", "fixed-dmz-4host.yaml")
    try:
        net_cfg_path = str(files("cyberwheel.data.configs.network") / net_cfg_name)
    except Exception:
        net_cfg_path = str(cw_root / "data" / "configs" / "network" / net_cfg_name)

    host_cfg = getattr(args, "host_config", "host_defs_services.yaml")
    network  = Network.create_network_from_yaml(net_cfg_path, host_cfg)
    service_map = get_service_map(network)
    args.service_mapping = {network.name: service_map}

    red_yaml  = getattr(args, "red_agent",  "rl_red_dmz.yaml")
    blue_yaml = getattr(args, "blue_agent", "inactive_blue_agent.yaml")

    red_cfg  = _load_yaml(("cyberwheel.data.configs.red_agent",  red_yaml),
                           cw_root / "data" / "configs" / "red_agent"  / red_yaml)
    blue_cfg = _load_yaml(("cyberwheel.data.configs.blue_agent", blue_yaml),
                           cw_root / "data" / "configs" / "blue_agent" / blue_yaml)

    host_keys = sorted(network.hosts.keys())
    entry = red_cfg.get("entry_host", "dmz_entry")
    if entry not in network.hosts:
        red_cfg["entry_host"] = host_keys[0]

    args.agent_config = {"red": red_cfg, "blue": blue_cfg}
    return CyberwheelRL(args, network=network)


class CWNativeWrapper(gym.Env):
    """
    Thin wrapper that normalises reward/termination to match NaSim:
      - step cost  = -1 every step
      - goal bonus = +100 when impact succeeds on the target host
      - terminates immediately on goal achievement
    """

    metadata = {"render_modes": []}

    def __init__(self, env: Optional[gym.Env] = None):
        super().__init__()
        self._env = env if env is not None else _make_cw_env()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(_OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(_ACT_SIZE)
        self._step_count = 0

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._step_count = 0
        raw, info = self._env.reset(seed=seed, options=options)
        return self._process_obs(raw), info or {}

    def step(self, action: int):
        action = int(action)
        cw_action = self._translate_action(action)
        raw, _cw_reward, _cw_term, _cw_trunc, info = self._env.step(
            {"red": cw_action, "blue": 0})

        cw_r = info.get("red_reward", _cw_reward) if isinstance(info, dict) else _cw_reward
        self._step_count += 1

        # NaSim-style reward: flat step cost + goal bonus
        goal_achieved = float(cw_r) >= 100.0   # CW gives ≥100 when impact hits
        reward     = _STEP_COST + (_GOAL_REWARD if goal_achieved else 0.0)
        terminated = bool(goal_achieved)
        # Truncate at _MAX_STEPS; ignore CW's own done (it truncates at 50 steps)
        truncated  = (not terminated) and (self._step_count >= _MAX_STEPS)

        return self._process_obs(raw), reward, terminated, truncated, info or {}

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _translate_action(self, action: int) -> int:
        """Map wrapper action (0..29) → valid CW action index."""
        try:
            red_as   = self._env.red_agent.action_space
            cur_size = red_as._action_space_size   # grows as hosts discovered
            if action >= cur_size:
                # Host not discovered yet — redirect to pingsweep on entry host
                action = 0
        except Exception:
            pass
        return action

    def _process_obs(self, raw) -> np.ndarray:
        vec = raw.get("red") if isinstance(raw, dict) else raw
        if vec is None:
            return np.zeros(_OBS_SIZE, dtype=np.float32)
        arr = np.asarray(vec, dtype=np.float32).ravel()
        if len(arr) < _OBS_SIZE:
            arr = np.pad(arr, (0, _OBS_SIZE - len(arr)), constant_values=0.0)
        else:
            arr = arr[:_OBS_SIZE]
        return ((arr + 1.0) / (_MAX_ATTR + 1.0)).astype(np.float32)
