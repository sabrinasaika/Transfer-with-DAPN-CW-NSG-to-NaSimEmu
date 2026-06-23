#!/usr/bin/env python3
"""Launch WhiteBoxNetSecGame with transfer_dapn NSG scenario patches applied."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from envs.nsg_scenario_patch import apply_nsg_scenario_patches

apply_nsg_scenario_patches()

from netsecgame.game.worlds import WhiteBoxNetSecGame as _mod  # noqa: E402

if __name__ == "__main__":
    import argparse
    import logging
    import os
    from netsecgame.utils.utils import get_logging_level

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_level", default="WARNING")
    parser.add_argument("-gh", "--game_host", default="127.0.0.1")
    parser.add_argument("-gp", "--game_port", type=int, default=9000)
    parser.add_argument("-c", "--task_config", required=True)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-l", "--log_level", default="WARNING")
    args = parser.parse_args()

    log_filename = Path("logs/WhiteBox_NSG_coordinator.log")
    log_filename.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=get_logging_level(args.log_level),
    )

    game_server = _mod.WhiteBoxNetSecGame(
        args.game_host, args.game_port, args.task_config, seed=args.seed)
    game_server.run()
