from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module

from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, validate_settings
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

import simpy

sparams_module.NUM_CHANNELS = 1

cfg_module.SIMULATION_TIME_us = 2e5

cfg_module.EXCLUDED_LOGS = {
    "NETWORK": ["ALL"],
    "NODE": ["ALL"],
    "GEN": ["ALL"],
    "LOAD": ["ALL"],
    "APP": ["ALL"],
    "MAC": ["ALL"],
    "PHY": ["ALL"],
    "MEDIUM": ["ALL"],
    "CHANNEL": ["ALL"],
    "STATS": [],
}

cfg_module.ENABLE_STATS_COLLECTION = True
cfg_module.STATS_SAVE_PATH = "data/statistics"

cfg_module.ENABLE_ADVANCED_NETWORK_CONFIG = True

cfg_module.BSSs_Advanced = [
    {
        "id": 1,  # A BSS
        "ap": {"id": 1, "pos": (0, 0, 0)},  # BSS Access Point (AP)
        "stas": [{"id": 2, "pos": (3, 4, 0)}],
        "traffic_flows": [
            {
                "destination": 2,
                "model": {"name": "Poisson"},
            },
        ],
    },
    {
        "id": 2,  # Another BSS
        "ap": {"id": 4, "pos": (5, 5, 1)},  # BSS Access Point (AP)
        "stas": [{"id": 5, "pos": (1, 2, 1)}],
        "traffic_flows": [
            {
                "destination": 5,
                "model": {"name": "Poisson"},
            },
        ],
    },
]


if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg_module, sparams_module)

    validate_settings(cfg_module, sparams_module, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = initialize_network(cfg_module, sparams_module, env)

    env.run(until=cfg_module.SIMULATION_TIME_us)

    network.stats.collect_stats()
    network.stats.display_stats()

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
