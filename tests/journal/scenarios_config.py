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


def _create_flow(
    rng: random.Random,
    dst_id: int,
    max_load_kbps: float,
    start_us: int,
    end_us: int,
    load_range: tuple[float, float],
):
    traffic_models = ["Poisson", "Bursty", "VR"]
    model_name = rng.choice(traffic_models)
    load = rng.uniform(*load_range) * max_load_kbps

    model = {
        "name": model_name,
        "start_time_us": start_us,
        "end_time_us": end_us,
        "traffic_load_kbps": int(load),
    }

    if model_name == "Bursty":
        model.update(
            {
                "burst_size_pkts": rng.randint(10, 40),
                "avg_inter_packet_time_us": rng.randint(2, 10),
            }
        )
    elif model_name == "VR":
        model.update(
            {
                "fps": rng.choice([60, 90, 120]),
                "avg_inter_packet_time_us": rng.randint(5, 10),
            }
        )

    return {"destination": dst_id, "model": model}


def _compute_max_load(
    ap_pos: tuple[float, float, float],
    sta_pos: tuple[float, float, float],
    rng: random.Random,
):
    sparams = sparams_module()
    distance_m = round(_calculate_distance(ap_pos, sta_pos), 2)
    rssi_dbm = get_rssi_dbm(sparams, distance_m, rng)
    mcs_index = get_highest_mcs_index(rssi_dbm, 4 * 20)
    return (
        calculate_data_rate_bps(
            mcs_index, 20, sparams.SPATIAL_STREAMS, sparams.GUARD_INTERVAL_us
        )
        / 1000
    )


def generate_random_flows_basic(
    ap_pos: tuple[float, float, float],
    sta_pos: tuple[float, float, float],
    dst_id: int,
    seed: int,
    load_range=(0.8, 0.9),
):
    rng = random.Random(seed)
    max_load_kbps = _compute_max_load(ap_pos, sta_pos, rng)
    return [_create_flow(rng, dst_id, max_load_kbps, 0, None, load_range)]


def generate_random_flows_intervals(
    ap_pos: tuple[float, float, float],
    sta_pos: tuple[float, float, float],
    dst_id: int,
    seed: int,
    high_load_range=(0.8, 0.9),
    low_load_range=(0.1, 0.2),
    low_load_intervals=None,
):
    rng = random.Random(seed)
    max_load_kbps = _compute_max_load(ap_pos, sta_pos, rng)
    low_load_intervals = low_load_intervals or set()
    intervals_us = [
        (0, 15_000_000),
        (15_000_000, 30_000_000),
        (30_000_000, 45_000_000),
        (45_000_000, 60_000_000),
    ]

    flows = []
    for i, (start_us, end_us) in enumerate(intervals_us):
        load_range = low_load_range if i in low_load_intervals else high_load_range
        flows.append(
            _create_flow(rng, dst_id, max_load_kbps, start_us, end_us, load_range)
        )
    return flows


def build_bss(
    idx: int,
    cfg: dict[str, Any],
    traffic_flows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a BSS entry for the scenario."""
    return {
        "id": idx,
        "ap": {
            "id": cfg["ap_id"],
            "pos": cfg["ap_pos"],
            "channels": cfg["channels"],
            "primary_channel": cfg["channels"][0] if cfg["channels"] else None,
        },
        "stas": [{"id": cfg["sta_id"], "pos": cfg["sta_pos"]}],
        "traffic_flows": traffic_flows,
    }


def build_scenario(
    bss_configs: list[dict[str, Any]],
    seed: int = None,
    use_intervals: bool = False,
    full_buffer_indices: set[int] = None,
    low_load_selector: callable = None,
    load_range: tuple[float, float] = (0.8, 0.9),
) -> list[dict[str, Any]]:
    seed = seed or random.randint(0, 2**24)
    rng = random.Random(seed)
    full_buffer_indices = full_buffer_indices or set()
    scenario = []

    for idx, cfg in enumerate(bss_configs, start=1):
        ap_pos, sta_pos = cfg["ap_pos"], cfg["sta_pos"]

        # Generate flows
        if use_intervals:
            low_intervals = (
                low_load_selector(rng, idx, len(bss_configs))
                if low_load_selector
                else None
            )
            flows = generate_random_flows_intervals(
                ap_pos,
                sta_pos,
                cfg["sta_id"],
                seed * 10 + idx,
                low_load_intervals=low_intervals,
            )
        else:
            flows = generate_random_flows_basic(
                ap_pos, sta_pos, cfg["sta_id"], seed * 10 + idx, load_range
            )

        # Force full-buffer APs
        if idx in full_buffer_indices:
            for flow in flows:
                model = flow["model"]
                flow["model"] = {
                    "name": "Full",
                    "start_time_us": model.get("start_time_us", 0),
                    "end_time_us": model.get("end_time_us", None),
                    "traffic_load_kbps": None,
                }

        scenario.append(build_bss(idx, cfg, flows))

    return scenario


# ---------- Scenario A ----------
def generate_scenario_a() -> list[dict[str, Any]]:
    bss_configs = [
        {
            "ap_id": 1,
            "ap_pos": (3, 6, 0.5),
            "sta_id": 101,
            "sta_pos": (4, 8, 0.5),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
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

    return build_scenario(
        bss_configs, use_intervals=False, full_buffer_indices={1, 2, 3}
    )


# ---------- Scenario B ----------
def generate_scenario_b(seed: int):
    def low_load_selector(rng, idx, num_bss):
        # 3 distinct low-load BSS (excluding AP1), plus repeat one
        if idx == 1:
            return set()
        chosen = rng.sample(range(2, num_bss + 1), 3)
        chosen.append(rng.choice(chosen))
        return {i for i, low_idx in enumerate(chosen) if idx == low_idx}

    bss_configs = [
        {
            "ap_id": 1,
            "ap_pos": (3, 8, 1.5),
            "sta_id": 101,
            "sta_pos": (4, 6, 0.5),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 2,
            "ap_pos": (6, 5, 1.5),
            "sta_id": 102,
            "sta_pos": (5, 6, 1.5),
            "channels": [1],
        },
        {
            "ap_id": 3,
            "ap_pos": (6, 3, 1),
            "sta_id": 103,
            "sta_pos": (6, 4, 1.5),
            "channels": [2],
        },
        {
            "ap_id": 4,
            "ap_pos": (2, 2, 0.5),
            "sta_id": 104,
            "sta_pos": (4, 2, 1.5),
            "channels": [3],
        },
        {
            "ap_id": 5,
            "ap_pos": (7, 6, 1),
            "sta_id": 105,
            "sta_pos": (6, 7, 0.5),
            "channels": [4],
        },
    ]

    return build_scenario(
        bss_configs,
        seed=seed,
        use_intervals=True,
        full_buffer_indices={1},
        low_load_selector=low_load_selector,
    )


# ---------- Scenario 1 ----------
def generate_scenario_1() -> list[dict[str, Any]]:
    bss_configs = [
        {
            "ap_id": 1,
            "ap_pos": (2, 7, 1),
            "sta_id": 101,
            "sta_pos": (1, 6, 1),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 2,
            "ap_pos": (3, 5, 1),
            "sta_id": 102,
            "sta_pos": (5, 5, 0.5),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 3,
            "ap_pos": (7, 8, 1),
            "sta_id": 103,
            "sta_pos": (7, 9, 1),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
    ]

    return build_scenario(
        bss_configs, use_intervals=False, full_buffer_indices={1, 2, 3}
    )


# ---------- Scenario 2 ----------
def generate_scenario_2(seed: int) -> list[dict[str, Any]]:
    bss_configs = [
        {
            "ap_id": 1,
            "ap_pos": (6, 9, 1),
            "sta_id": 101,
            "sta_pos": (7, 8, 0.5),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 2,
            "ap_pos": (4, 5, 1),
            "sta_id": 102,
            "sta_pos": (4, 5, 1),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 3,
            "ap_pos": (3, 9, 0),
            "sta_id": 103,
            "sta_pos": (2, 8, 0),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 4,
            "ap_pos": (4, 4, 0.5),
            "sta_id": 104,
            "sta_pos": (5, 4, 1),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
    ]

    return build_scenario(
        bss_configs,
        seed=seed,
        use_intervals=False,
        full_buffer_indices={1, 2},
        load_range=(0.2, 0.4),
    )


# ---------- Scenario 3 ----------
def generate_scenario_3(seed: int) -> list[dict[str, Any]]:
    bss_configs = [
        {
            "ap_id": 1,
            "ap_pos": (2, 7, 1),
            "sta_id": 101,
            "sta_pos": (1, 8, 1),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 2,
            "ap_pos": (9, 5, 1),
            "sta_id": 102,
            "sta_pos": (8, 4, 0.5),
            "rl_driven": True,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 3,
            "ap_pos": (1, 2, 0),
            "sta_id": 103,
            "sta_pos": (2, 1, 0),
            "rl_driven": False,
            "channels": [1, 2, 3, 4],
        },
        {
            "ap_id": 4,
            "ap_pos": (7, 6, 0.5),
            "sta_id": 104,
            "sta_pos": (5, 6, 1),
            "rl_driven": False,
            "channels": [1, 2, 3, 4],
        },
    ]

    return build_scenario(
        bss_configs,
        seed=seed,
        use_intervals=False,
        load_range=(0.6, 0.9),
    )


def get_scenario(
    name: str,
    baseline_channels: list[int] | None = None,
    seed: int | None = None,
) -> list[dict]:
    if name == "A":
        scenario = generate_scenario_a()
    elif name == "B":
        scenario = generate_scenario_b(seed)
    elif name == "1":
        scenario = generate_scenario_1()
    elif name == "2":
        scenario = generate_scenario_2(seed)
    elif name == "3":
        scenario = generate_scenario_3(seed)
    else:
        raise ValueError(f"Unknown scenario: {name}")

    # Manually override the AP channels and primary channel
    if baseline_channels:
        if name in ["A", "B"]:  # Override AP 1's channels and primary channel
            scenario[0]["ap"]["channels"] = baseline_channels
            scenario[0]["ap"]["primary_channel"] = baseline_channels[0]
            scenario[0]["ap"]["rl_driven"] = False
        else:
            for bss in scenario:  # Override all AP's channels and primary channel
                bss["ap"]["channels"] = baseline_channels
                bss["ap"]["primary_channel"] = baseline_channels[0]
                bss["ap"]["rl_driven"] = False

    return scenario


# for seed in range(1,21):
#     print("SEED", seed)
#     scenario = get_scenario("3", seed=seed)

#     for bss in scenario:
#         print("BSS", bss["id"])
#         for flow in bss["traffic_flows"]:
#             print(flow)
