from src.user_config import UserConfig as cfg
from src.sim_params import SimParams as sparams

from src.components.network import Network
from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, validate_settings
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)


import simpy


sparams.MAX_TX_QUEUE_SIZE_pkts = 100
sparams.ENABLE_RTS_CTS = True
sparams.MPDU_ERROR_PROBABILITY = 0.1

sparams.BONDING_MODE = 0

sparams.NUM_CHANNELS = 1

cfg.SIMULATION_TIME_us = 2e4
cfg.SEED = 1

cfg.ENABLE_CONSOLE_LOGGING = True
cfg.USE_COLORS_IN_LOGS = True
cfg.ENABLE_LOGS_RECORDING = False
cfg.EXCLUDED_LOGS = {
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

cfg.ENABLE_TRAFFIC_GEN_RECORDING = False

cfg.ENABLE_STATS_COLLECTION = True
cfg.STATS_SAVE_PATH = "data/statistics"

cfg.NETWORK_BOUNDS_m = (10, 10, 2)

cfg.ENABLE_ADVANCED_NETWORK_CONFIG = True

cfg.BSSs_Advanced = [
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

    logger = get_logger("TEST", cfg, sparams)

    validate_settings(cfg, sparams, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = initialize_network(cfg, sparams, env)

    env.run(until=cfg.SIMULATION_TIME_us)

    network.stats.collect_stats()
    network.stats.display_stats()

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
