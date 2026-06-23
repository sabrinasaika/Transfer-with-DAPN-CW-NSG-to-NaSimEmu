"""Patch NetSecGame two_networks scenario for NaSim-aligned KC transfer.

NSG's bundled smb_exploit uses a typo'd service version (``10.0. 19041``) so
``microsoft-ds`` is never exploitable.  We fix that in-process before the NSG
server loads the scenario (see ``run_nsg_server.py``).
"""

from __future__ import annotations

_PATCHED = False


def apply_nsg_scenario_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from cyst.api.configuration import ExploitConfig
    from netsecgame.game.scenarios import SCENARIO_REGISTRY
    import netsecgame.game.scenarios.two_nets as two_nets

    def _fix_smb_exploit(config_objects) -> None:
        for obj in config_objects:
            if not isinstance(obj, ExploitConfig):
                continue
            for svc in obj.services:
                if svc.service == "microsoft-ds" and "19041" in svc.min_version:
                    svc.min_version = "10.0.19041"

    _fix_smb_exploit(two_nets.configuration_objects)
    if "two_networks" in SCENARIO_REGISTRY:
        _fix_smb_exploit(SCENARIO_REGISTRY["two_networks"])

    _PATCHED = True
