"""Load fixed NaSim scenarios for the transfer pipeline."""

from __future__ import annotations

from pathlib import Path

from nasimemu.nasim.scenarios.loader_v2 import ScenarioLoaderV2
from nasimemu.nasim.scenarios.scenario import Scenario
import nasimemu.nasim.scenarios.utils as u

from envs.scenario_cfg import ScenarioCfg, TWO_SUBNET, get_scenario

_NSG_NASIM_SCENARIO = (
    Path(__file__).resolve().parents[1] / "scenarios" / "nasim_two_subnet.yaml"
)


class FixedScenarioLoaderV2(ScenarioLoaderV2):
    """Parse fixed/reproducible v2 YAML and disable subnet permutation."""

    def _parse_sensitive_hosts(self):
        sensitive_hosts = self.yaml_dict[u.SENSITIVE_HOSTS]
        self.sensitive_hosts = {}

        explicit = all(
            isinstance(k, tuple)
            or (isinstance(k, str) and k.strip().startswith("("))
            for k in sensitive_hosts
        )
        if explicit:
            for address, value in sensitive_hosts.items():
                addr = address if isinstance(address, tuple) else eval(address)
                self.sensitive_hosts[addr] = float(value)
            return

        return super()._parse_sensitive_hosts()

    def _parse_host_configs(self):
        if u.HOST_CONFIGS in self.yaml_dict:
            raw_configs = self.yaml_dict[u.HOST_CONFIGS]
            self.host_configs = {
                (addr if isinstance(addr, tuple) else eval(addr)): cfg
                for addr, cfg in raw_configs.items()
            }
            self._validate_host_configs(self.host_configs)
            return

        return super()._parse_host_configs()

    def _construct_scenario(self):
        scenario_dict = dict()
        scenario_dict[u.SUBNETS] = self.subnets
        scenario_dict[u.TOPOLOGY] = self.topology
        scenario_dict[u.OS] = self.os
        scenario_dict[u.SERVICES] = self.services
        scenario_dict[u.PROCESSES] = self.processes
        scenario_dict[u.SENSITIVE_HOSTS] = self.sensitive_hosts
        scenario_dict[u.EXPLOITS] = self.exploits
        scenario_dict[u.PRIVESCS] = self.privescs
        scenario_dict[u.OS_SCAN_COST] = self.os_scan_cost
        scenario_dict[u.SERVICE_SCAN_COST] = self.service_scan_cost
        scenario_dict[u.SUBNET_SCAN_COST] = self.subnet_scan_cost
        scenario_dict[u.PROCESS_SCAN_COST] = self.process_scan_cost
        scenario_dict[u.FIREWALL] = self.firewall
        scenario_dict[u.HOSTS] = self.hosts
        scenario_dict[u.STEP_LIMIT] = self.step_limit
        scenario_dict["address_space_bounds"] = self.address_space_bounds

        return Scenario(
            scenario_dict,
            name=self.name,
            generated=False,
            permute_subnets=False,
        )


def load_scenario_cfg(cfg: ScenarioCfg | str = "two_subnet"):
    if isinstance(cfg, str):
        cfg = get_scenario(cfg)
    return FixedScenarioLoaderV2().load(str(cfg.nasim_yaml))


def load_fixed_dmz_scenario():
    return load_scenario_cfg(TWO_SUBNET)


def load_nsg_nasim_scenario():
    return FixedScenarioLoaderV2().load(str(_NSG_NASIM_SCENARIO))


def patch_nasim_load_scenario(active: ScenarioCfg | str = "two_subnet") -> None:
    """Route scenario loads through transfer_dapn loader (for NASimEmuEnv)."""
    import nasimemu.nasim as nasim_pkg
    import nasimemu.nasim.scenarios as scenarios_pkg

    if isinstance(active, str):
        active = get_scenario(active)

    from envs.scenario_cfg import SCENARIOS
    orig = scenarios_pkg.load_scenario
    watched = {str(cfg.nasim_yaml): cfg for cfg in SCENARIOS.values()}

    def _load(path, name=None):
        key = str(path)
        if key in watched:
            return load_scenario_cfg(watched[key])
        return orig(path, name=name)

    scenarios_pkg.load_scenario = _load
    nasim_pkg.load_scenario = _load
    patch_nasim_load_scenario._active = active  # type: ignore[attr-defined]
