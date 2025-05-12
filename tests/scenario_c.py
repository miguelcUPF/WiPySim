from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module

from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, validate_settings, wandb_init
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
    SECTION_DIVIDER_MSG,
)

from codecarbon import EmissionsTracker

import simpy
import json


STRATEGY = "sw_linucb"
cfg_module.RL_MODE = 1
cfg_module.SIMULATION_TIME_us = 30e6

# SW-LinUCB RL MODE 1 (MARL) TODO
cfg_module.AGENTS_SETTINGS = {
    "strategy": STRATEGY,
    "channel_frequency": 1,
    "primary_frequency": 1,
    "cw_frequency": 1,
    "alpha": 1,
    "window_size": 0,
}

sparams_module.CW_MIN = 16
sparams_module.CW_MAX = 2**6 * sparams_module.CW_MIN

sparams_module.NUM_CHANNELS = 4

cfg_module.SEED = 1
cfg_module.ENABLE_RL = True

cfg_module.NETWORK_BOUNDS_m = (20, 20, 2)

cfg_module.FIRST_AS_PRIMARY = True

cfg_module.ENABLE_CONSOLE_LOGGING = True
cfg_module.DISABLE_SIMULTANEOUS_ACTION_SELECTION = False
cfg_module.ENABLE_REWARD_DECOMPOSITION = False

cfg_module.ENABLE_ADVANCED_NETWORK_CONFIG = True

cfg_module.ENABLE_STATS_COMPUTATION = True

cfg_module.USE_WANDB = True
cfg_module.WANDB_RUN_NAME = f"{cfg_module.RL_MODE}_{STRATEGY}"
cfg_module.USE_CODECARBON = True


cfg_module.BSSs_Advanced = [
    {
        "id": 1,  # A BSS
        "ap": {"id": 1, "pos": (4, 18, 1), "rl_driven": True},
        "stas": [{"id": 2, "pos": (7, 10, 1)}],
        "traffic_flows": [
            {
                "destination": 2,
                "model": {"name": "Full"},
            },
        ],
    },
    {
        "id": 2,  # Another BSS TODO: implement changing in mid
        "ap": {"id": 3, "pos": (10, 10, 0.5), "channels": [3, 4], "primary_channel": 3},
        "stas": [{"id": 4, "pos": (12, 17, 1)}],
        "traffic_flows": [
            {
                "destination": 4,
                "model": {"name": "Full"},
            }
        ],
    },
    {
        "id": 3,  # Another BSS
        "ap": {"id": 5, "pos": (14, 14, 1), "channels": [2], "primary_channel": 2},
        "stas": [{"id": 6, "pos": (18, 8, 0.8)}],
        "traffic_flows": [
            {
                "destination": 6,
                "model": {"name": "Full"},
            }
        ],
    },
    {
        "id": 4,  # Another BSS
        "ap": {"id": 7, "pos": (14, 8, 1.5), "channels": [1], "primary_channel": 1},
        "stas": [{"id": 8, "pos": (10, 2, 1.5)}],
        "traffic_flows": [
            {
                "destination": 8,
                "model": {"name": "Full"},
            }
        ],
    },
    {
        "id": 5,  # Another BSS TODO: implement changing in mid
        "ap": {"id": 9, "pos": (2, 4, 1), "channels": [4], "primary_channel": 4},
        "stas": [{"id": 10, "pos": (10, 6, 1.2)}],
        "traffic_flows": [
            {
                "destination": 10,
                "model": {"name": "Full"},
            }
        ],
    },
]

DISPLAY_AGENTS_EMISSIONS = True
DISPLAY_SIMULATION_EMISSIONS = True

emissions_tracker = (
    EmissionsTracker(project_name="simulation") if cfg_module.USE_CODECARBON else None
)

if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg_module, sparams_module)

    validate_settings(cfg_module, sparams_module, logger)
    wandb_init(cfg_module)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = initialize_network(cfg_module, sparams_module, env)

    emissions_tracker.start() if cfg_module.USE_CODECARBON else None

    env.run(until=cfg_module.SIMULATION_TIME_us)

    emissions_tracker.stop() if cfg_module.USE_CODECARBON else None

    network.stats.collect_stats()

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> Tx attempts: {ap.tx_stats.tx_attempts}, Tx Failures: {ap.tx_stats.tx_failures}, Tx Pkts: {ap.tx_stats.pkts_tx}, Pkts Success: {ap.tx_stats.pkts_success}, Dropped Pkts: {ap.tx_stats.pkts_dropped_queue_lim + ap.tx_stats.pkts_dropped_retry_lim}, Channels: [{', '.join(map(str, ap.phy_layer.channels_ids))}], Sensing Channels: {', '.join(map(str, ap.phy_layer.sensing_channels_ids))}"
        )
        if cfg_module.USE_CODECARBON:
            (
                ap.mac_layer.rl_controller.log_emissions_data()
                if ap.mac_layer.rl_controller
                else None
            )

    if cfg_module.USE_CODECARBON:
        if DISPLAY_AGENTS_EMISSIONS:
            print(SECTION_DIVIDER_MSG)
            for ap in network.get_aps():
                if not ap.mac_layer.rl_controller:
                    continue
                emissions_data = ap.mac_layer.rl_controller.get_emissions_data()
                logger.info(f"AP {ap.id}:\n{json.dumps(emissions_data, indent=6)}")
                logger.info(
                    f"AP {ap.id}: \n{len(ap.mac_layer.rl_controller.results)} results"
                )

        if DISPLAY_SIMULATION_EMISSIONS:
            print(SECTION_DIVIDER_MSG)
            logger.info(
                f"SIMULATION:\n{json.dumps(emissions_tracker.final_emissions_data.__dict__, indent=6)}"
            )

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
