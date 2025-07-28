from typing import Any
from tests._sim_params_tests import SimParams as sparams_module
from tests._user_config_tests import UserConfig as cfg_module

import math
import random

from src.utils.transmission import get_rssi_dbm
from src.utils.mcs_table import get_highest_mcs_index, calculate_data_rate_bps


def _calculate_distance(
    pos1: tuple[float, float, float], pos2: tuple[float, float, float]
) -> float:
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def get_unique_position(
    bounds: tuple, used_positions: set, rng: random.Random
) -> tuple:
    """Generate a unique random position within bounds."""
    while True:
        x_lim, y_lim, z_lim = bounds
        pos = (
            round(rng.uniform(0, x_lim), 2),
            round(rng.uniform(0, y_lim), 2),
            round(rng.uniform(0, z_lim), 2),
        )
        if pos not in used_positions:
            used_positions.add(pos)
            return pos


def generate_random_flows(
    ap_pos: tuple[float, float, float],
    sta_pos: tuple[float, float, float],
    seed: int,
    high_load_range: tuple[int, int] = (0.4, 0.8),
    low_load_range: tuple[int, int] = (0, 0.2),
    low_load_intervals: set[int] = set(),
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sparams = sparams_module
    traffic_models = ["Poisson", "Bursty", "VR"]
    intervals_us = [
        (0, 40_000_000),
        (40_000_000, 80_000_000),
        (80_000_000, 120_000_000),
    ]

    # Estimate max capacity
    distance_m = round(_calculate_distance(ap_pos, sta_pos), 2)
    rssi_dbm = get_rssi_dbm(sparams, distance_m, rng)
    mcs_index = get_highest_mcs_index(rssi_dbm, 4 * 20)  # 4 channels max bandwidth
    data_rate_bps = calculate_data_rate_bps(
        mcs_index,
        20,
        sparams.SPATIAL_STREAMS,
        sparams.GUARD_INTERVAL_us,
    )
    max_load_kbps = data_rate_bps / 1000

    flows = []
    for i, (start_us, end_us) in enumerate(intervals_us):
        model_name = rng.choice(traffic_models)
        model = {
            "name": model_name,
            "start_time_us": start_us,
            "end_time_us": end_us,
        }

        if i in low_load_intervals:
            load = rng.uniform(low_load_range[0], low_load_range[1]) * max_load_kbps
        else:
            load = rng.uniform(high_load_range[0], high_load_range[1]) * max_load_kbps
        model["traffic_load_kbps"] = int(load)

        if model_name == "Bursty":
            model["burst_size_pkts"] = rng.randint(10, 40)
            model["avg_inter_packet_time_us"] = rng.randint(2, 10)
        elif model_name == "VR":
            model["fps"] = rng.choice([60, 90, 120])
            model["avg_inter_packet_time_us"] = rng.randint(5, 10)

        flows.append({"destination": None, "model": model})
    return flows


# ---------- Scenario A ----------
def generate_scenario_a() -> list[dict[str, Any]]:
    scenario = []

    bss_configs = [
        {
            "ap_id": 1,
            "ap_pos": (3, 6, 0.5),
            "sta_id": 101,
            "sta_pos": (4, 8, 0.5),
            "channels": [2],
        },
        {
            "ap_id": 2,
            "ap_pos": (7, 5, 1),
            "sta_id": 102,
            "sta_pos": (6, 7, 0.5),
            "channels": [3, 4],
        },
        {
            "ap_id": 3,
            "ap_pos": (3, 3, 1),
            "sta_id": 103,
            "sta_pos": (5, 2, 1),
            "channels": [1],
        },
    ]

    for idx, cfg in enumerate(bss_configs, start=1):
        ap_pos = cfg["ap_pos"]
        sta_pos = cfg["sta_pos"]

        traffic_flows = [
            {
                "destination": cfg["sta_id"],
                "model": {"name": "Full"},
            }
        ]

        bss = {
            "id": idx,
            "ap": {
                "id": cfg["ap_id"],
                "pos": ap_pos,
                "channels": cfg["channels"],
                "primary_channel": cfg["channels"][0],
            },
            "stas": [{"id": cfg["sta_id"], "pos": sta_pos}],
            "traffic_flows": traffic_flows,
        }

        scenario.append(bss)

    return scenario


# ---------- Scenario B ----------
def generate_scenario_b(seed: int) -> list[dict[str, Any]]:
    if seed is None:
        seed = random.randint(0, 2**24)
    rng = random.Random(seed)

    scenario = []

    used_positions = set()
    bounds = cfg_module.NETWORK_BOUNDS_m

    bss_configs = [
        {
            "ap_id": 1,
            "ap_pos": None,
            "sta_id": 101,
            "sta_pos": None,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 2,
            "ap_pos": None,
            "sta_id": 102,
            "sta_pos": None,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 3,
            "ap_pos": None,
            "sta_id": 103,
            "sta_pos": None,
            "channels": [1, 2],
        },
        {
            "ap_id": 4,
            "ap_pos": None,
            "sta_id": 104,
            "sta_pos": None,
            "channels": [3, 4],
        },
    ]

    for idx, cfg in enumerate(bss_configs, start=1):
        ap_pos = (
            cfg["ap_pos"]
            if cfg["ap_pos"] is not None
            else get_unique_position(bounds, used_positions, rng)
        )
        sta_pos = (
            cfg["sta_pos"]
            if cfg["sta_pos"] is not None
            else get_unique_position(bounds, used_positions, rng)
        )

        traffic_flows = generate_random_flows(ap_pos, sta_pos, seed + idx)
        for flow in traffic_flows:
            flow["destination"] = cfg["sta_id"]

        bss = {
            "id": idx,
            "ap": {
                "id": cfg["ap_id"],
                "pos": ap_pos,
                "channels": cfg["channels"],
                "primary_channel": cfg["channels"][0],
            },
            "stas": [{"id": cfg["sta_id"], "pos": sta_pos}],
            "traffic_flows": traffic_flows,
        }

        scenario.append(bss)

    return scenario


# ---------- Scenario C ----------
def generate_scenario_c(seed: int) -> list[dict[str, Any]]:
    if seed is None:
        seed = random.randint(0, 2**24)
    rng = random.Random(seed)
    scenario = []

    used_positions = set()
    bounds = cfg_module.NETWORK_BOUNDS_m

    bss_configs = [
        {
            "ap_id": 1,
            "ap_pos": None,
            "sta_id": 101,
            "sta_pos": None,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 2,
            "ap_pos": None,
            "sta_id": 102,
            "sta_pos": None,
            "channels": [1],
        },
        {
            "ap_id": 3,
            "ap_pos": None,
            "sta_id": 103,
            "sta_pos": None,
            "channels": [2],
        },
        {
            "ap_id": 4,
            "ap_pos": None,
            "sta_id": 104,
            "sta_pos": None,
            "channels": [3],
        },
        {
            "ap_id": 5,
            "ap_pos": None,
            "sta_id": 105,
            "sta_pos": None,
            "channels": [4],
        },
    ]

    # Choose 1 random BSS to be low load in each of the 3 intervals (distinct)
    low_load_indices = rng.sample(
        range(1, len(bss_configs)), 3
    )  # Exclude BSS 1 (index 0)

    intervals_us = [
        (0, 40_000_000),
        (40_000_000, 80_000_000),
        (80_000_000, 120_000_000),
    ]

    for idx, cfg in enumerate(bss_configs, start=1):
        ap_pos = (
            cfg["ap_pos"]
            if cfg["ap_pos"] is not None
            else get_unique_position(bounds, used_positions, rng)
        )
        sta_pos = (
            cfg["sta_pos"]
            if cfg["sta_pos"] is not None
            else get_unique_position(bounds, used_positions, rng)
        )

        # Determine which intervals this BSS should have low load
        low_load_intervals = {
            i for i, low_idx in enumerate(low_load_indices) if idx - 1 == low_idx
        }

        traffic_flows = generate_random_flows(
            ap_pos, sta_pos, seed + idx, (0.6, 0.9), (0, 0.2), low_load_intervals
        )
        for flow in traffic_flows:
            flow["destination"] = cfg["sta_id"]

        bss = {
            "id": idx,
            "ap": {
                "id": cfg["ap_id"],
                "pos": ap_pos,
                "channels": cfg["channels"],
                "primary_channel": cfg["channels"][0],
            },
            "stas": [{"id": cfg["sta_id"], "pos": sta_pos}],
            "traffic_flows": traffic_flows,
        }

        scenario.append(bss)

    return scenario


def get_scenario(
    name: str,
    all_rl_driven: int = 0,
    ap1_channels: list[int] | None = None,
    seed: int | None = None,
) -> list[dict]:
    if name == "A":
        scenario = generate_scenario_a()
    elif name == "B":
        scenario = generate_scenario_b(seed)
    elif name == "C":
        scenario = generate_scenario_c(seed)
    else:
        raise ValueError(f"Unknown scenario: {name}")

    # Manually override AP 1's channels and primary channel
    if ap1_channels:
        scenario[0]["ap"]["channels"] = ap1_channels
        scenario[0]["ap"]["primary_channel"] = ap1_channels[0]
        scenario[0]["ap"]["rl_driven"] = False
    else:
        scenario[0]["ap"]["rl_driven"] = True

    # All aps have rl_driven set
    if all_rl_driven == 1:
        for bss in scenario:
            bss["ap"]["rl_driven"] = True

    return scenario
