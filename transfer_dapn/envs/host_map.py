"""Fixed host layout and NaSim flat-action indices for the two-subnet DMZ scenario.

Scenario: fixed_dmz_two_subnet.v2.yaml  (7 hosts across 3 subnets)
  subnets: [1, 3, 3]  — DMZ:1, user:3, service:3

Exploit ordering in YAML (determines local offset = 4 + global_index):
  index 0: e_proftpd       -> local 4
  index 1: e_drupal        -> local 5
  index 2: e_phpwiki       -> local 6
  index 3: e_elasticsearch -> local 7
  index 4: e_wp_ninja      -> local 8
  privesc pe_kernel        -> local 9

ACTIONS_PER_HOST = 4 scans + 5 exploits + 1 privesc = 10
"""

from __future__ import annotations

from envs.kill_chain import SLOT_ORDER, ENTRY_ADDR, ENTRY_SLOT

ACTIONS_PER_HOST = 10

EXPLOIT_LOCAL = {
    (1, 0): 4,   # e_proftpd  (DMZ entry)
    (2, 0): 6,   # e_phpwiki  (user_server)
    (2, 1): 8,   # e_wp_ninja (user_worker — Windows)
    (2, 2): 5,   # e_drupal   (user_extra)
    (3, 0): 5,   # e_drupal   (service_target — TARGET)
    (3, 1): 6,   # e_phpwiki  (service_extra_1)
    (3, 2): 7,   # e_elasticsearch (service_extra_2 — Windows)
}

EXPLOIT_SERVICE = {
    (1, 0): "21_linux_proftpd",
    (2, 0): "80_linux_phpwiki",
    (2, 1): "80_windows_wp_ninja",
    (2, 2): "80_linux_drupal",
    (3, 0): "80_linux_drupal",
    (3, 1): "80_linux_phpwiki",
    (3, 2): "9200_windows_elasticsearch",
}

PRIVESC_LOCAL = 9

# KC flat local offset → exploit name (YAML ordering, not NASimEmu alphabetical)
_EXPLOIT_NAME = {
    ((1, 0), 4): "e_proftpd",
    ((2, 0), 6): "e_phpwiki",
    ((2, 1), 8): "e_wp_ninja",
    ((2, 2), 5): "e_drupal",
    ((3, 0), 5): "e_drupal",
    ((3, 1), 6): "e_phpwiki",
    ((3, 2), 7): "e_elasticsearch",
}


def discover_host_layout(address_space, slot_order=None):
    """Map slot addresses → flat-action block indices."""
    from envs.kill_chain import SLOT_ORDER
    if slot_order is None:
        slot_order = SLOT_ORDER
    layout = {}
    for slot, addr in enumerate(slot_order):
        if addr is not None and addr in address_space:
            layout[slot] = address_space.index(addr)
    return layout


def flat_action(slot_host_idx: int, local_offset: int,
                actions_per_host: int = ACTIONS_PER_HOST) -> int:
    return slot_host_idx * actions_per_host + local_offset


def flat_to_slot(flat: int, layout: dict[int, int]) -> int:
    """Map flat NaSim action index → KC slot via host block index."""
    from envs.kill_chain import MAX_SLOTS
    host_idx = flat // ACTIONS_PER_HOST
    for slot, idx in layout.items():
        if idx == host_idx:
            return slot
    return MAX_SLOTS


def _flat_addr(flat: int, layout: dict[int, int]) -> tuple[int, int]:
    host_idx = flat // ACTIONS_PER_HOST
    for slot, idx in layout.items():
        if idx == host_idx:
            return SLOT_ORDER[slot]
    return ENTRY_ADDR


def flat_to_emu_action(flat: int, layout: dict[int, int],
                       action_list) -> tuple[tuple[int, int], int]:
    """Flat NaSim KC action → NASimEmuEnv ((subnet, host), action_id).

    NASimEmuEnv sorts exploits alphabetically (e_proftpd=7), unlike KC flat
    offsets (e_proftpd=4). This helper resolves the correct action_id.
    """
    local = flat % ACTIONS_PER_HOST
    addr = _flat_addr(flat, layout)
    if local <= 3:
        return (addr, local)
    if local == PRIVESC_LOCAL:
        pe_id = next(
            i for i, (_, p) in enumerate(action_list)
            if p.get("name") == "pe_kernel"
        )
        return (addr, pe_id)
    name = _EXPLOIT_NAME.get((addr, local))
    if name:
        ex_id = next(
            i for i, (_, p) in enumerate(action_list)
            if p.get("name") == name
        )
        return (addr, ex_id)
    return (addr, local)


def flat_to_pe_action(flat: int, layout: dict[int, int]) -> tuple[tuple[int, int], int]:
    """Deprecated alias — prefer flat_to_emu_action with env.action_list."""
    host_idx = flat // ACTIONS_PER_HOST
    action_id = flat % ACTIONS_PER_HOST
    for slot, idx in layout.items():
        if idx == host_idx:
            return (SLOT_ORDER[slot], action_id)
    return (SLOT_ORDER[0], action_id)


def entry_on_host(kc_obs) -> bool:
    """True if the agent is on the entry host (slot 0, on_host feature)."""
    from envs.kill_chain import ENTRY_SLOT, KC_FEATS
    return float(kc_obs[ENTRY_SLOT * KC_FEATS + 2]) > 0.5
