"""In-process NetSecGame client: spawns WhiteBoxNetSecGame server + socket agent."""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from netsecgame import Action, ActionType, AgentRole, BaseAgent, IP, Observation
from netsecgame.game_components import AgentStatus


def nasim_goal_reached(target_stage: int) -> bool:
    """NaSim-aligned win: ROOT on target — stage 2 in the KC wrapper."""
    return target_stage >= 2

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = str(_ROOT / "scenarios" / "nsg_two_networks_task.yaml")
_NSG_SERVER = str(_ROOT / "run_nsg_server.py")


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"NSG server did not start on {host}:{port} within {timeout}s")


class NSGClient:
    """Manages a local WhiteBoxNetSecGame subprocess and attacker socket."""

    def __init__(
        self,
        task_config: str = DEFAULT_TASK,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        seed: int = 0,
        log_level: int = logging.WARNING,
    ):
        self.host = host
        self.port = port or _pick_free_port(host)
        self.task_config = str(Path(task_config).resolve())
        self.seed = int(seed)
        self._proc: Optional[subprocess.Popen] = None
        self._agent: Optional[BaseAgent] = None
        self._log_level = log_level
        self._step_count = 0

    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = [
            sys.executable,
            _NSG_SERVER,
            "-gh",
            self.host,
            "-gp",
            str(self.port),
            "-c",
            self.task_config,
            "-s",
            str(self.seed),
            "-l",
            "WARNING",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_port(self.host, self.port)
        time.sleep(3.0)
        self._agent = BaseAgent(self.host, self.port, AgentRole.Attacker)
        obs = self._agent.register()
        if obs is None:
            self.close()
            raise RuntimeError("NSG agent registration failed")
        logging.getLogger("NSGClient").info("Registered with NSG server on port %s", self.port)

    def reset(self, seed: Optional[int] = None) -> Observation:
        if self._agent is None:
            self.start()
        assert self._agent is not None
        self._step_count = 0
        obs = self._agent.request_game_reset(
            request_trajectory=False,
            randomize_topology=False,
            seed=self.seed,
        )
        if obs is None:
            raise RuntimeError("NSG reset failed")
        return obs

    def step(self, action: Action) -> Observation:
        assert self._agent is not None
        obs = self._agent.make_step(action)
        if obs is None:
            raise RuntimeError("NSG step returned no observation")
        self._step_count += 1
        return obs

    @property
    def agent(self) -> BaseAgent:
        assert self._agent is not None
        return self._agent

    def close(self) -> None:
        if self._agent is not None:
            try:
                self._agent.terminate_connection()
            except Exception:
                pass
            self._agent = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def nsg_win(obs: Observation) -> bool:
    if not obs.end:
        return False
    info = obs.info or {}
    return info.get("end_reason") == AgentStatus.Success
