from src.components.network import Network
from src.utils.event_logger import get_logger
from src.utils.support import initialize_network
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)


import simpy
import importlib
import random

import src.user_config as cfg
import src.sim_params as sparams
import src.utils.event_logger
import src.utils.plotters
import src.components.mac
import src.components.phy
import src.components.medium

cfg.ENABLE_CONSOLE_LOGGING = True
cfg.USE_COLORS_IN_LOGS = True
cfg.ENABLE_LOGS_RECORDING = False
cfg.EXCLUDED_LOGS = {"GEN": ["ALL"]}

cfg.ENABLE_TRAFFIC_GEN_RECORDING = False

cfg.NETWORK_BOUNDS_m = (10, 10, 2)

sparams.MAX_TX_QUEUE_SIZE_pkts = 100  # Test: 10, 50, 100
sparams.ENABLE_RTS_CTS = True  # Test: False and True
sparams.MPDU_ERROR_PROBABILITY = 0.1  # Test: 0, 0.1, 0.5

sparams.NUM_CHANNELS = 1

importlib.reload(src.utils.event_logger)
importlib.reload(src.components.mac)
importlib.reload(src.components.phy)
importlib.reload(src.components.medium)


SIMULATION_TIME_us = 2e4

SEED = 1
random.seed(SEED)

BSSs = [
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
    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    logger = get_logger("TEST", env)

    network = Network(env)

    initialize_network(env, BSSs, cfg.NETWORK_BOUNDS_m, network)

    env.run(until=SIMULATION_TIME_us)

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> Tx attempts:{ap.tx_stats.tx_attempts}, Tx Failures: {ap.tx_stats.tx_failures}, Tx Pkts: {ap.tx_stats.pkts_tx}, Dropped Pkts: {ap.tx_stats.pkts_dropped_queue_lim + ap.tx_stats.pkts_dropped_retry_lim}"
        )

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
