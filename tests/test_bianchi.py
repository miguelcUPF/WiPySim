from src.components.network import Network
from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, add_bss_automatically
from src.utils.theoretical import compute_collision_probability
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
import src.utils.statistics

def reload_libs():
    importlib.reload(src.utils.event_logger)
    importlib.reload(src.components.mac)
    importlib.reload(src.components.phy)
    importlib.reload(src.components.medium)
    importlib.reload(src.utils.statistics)

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
    "STATS": ["ALL"],
}

cfg.ENABLE_TRAFFIC_GEN_RECORDING = False

cfg.ENABLE_STATS_COLLECTION = False

cfg.NETWORK_BOUNDS_m = (10, 10, 2)

sparams.MAX_TX_QUEUE_SIZE_pkts = 100
sparams.ENABLE_RTS_CTS = True
sparams.MPDU_ERROR_PROBABILITY = 0.1

sparams.NUM_CHANNELS = 1

reload_libs()


SIMULATION_TIME_us = 2e5

SEED = 1
random.seed(SEED)

N = [2, 5, 10, 15, 20]
M = [0, 1, 3, 7]
CW_MIN = [16, 32, 64, 128, 256, 512, 1024]


if __name__ == "__main__":
    print(STARTING_TEST_MSG)
    print(STARTING_SIMULATION_MSG)

    for n in N:
        BSSs = []
        j = 0
        for i in range(n):
            BSSs = add_bss_automatically(BSSs, i-1, i+j)
            j += 1

        for m in M:
            sparams.COMMON_RETRY_LIMIT = m
            for cw_min in CW_MIN:
                sparams.CW_MIN = cw_min
                reload_libs()

                env = simpy.Environment()

                logger = get_logger("TEST", env)

                network = Network(env)

                initialize_network(env, BSSs, cfg.NETWORK_BOUNDS_m, network)

                env.run(until=SIMULATION_TIME_us)

                network.stats.collect_stats()

                print(f"n: {n}, m: {m}, cw_min: {cw_min}")
                print(f"Simulation Collision Probability: {network.stats.total_tx_failures / network.stats.total_tx_attempts * 100}%")

                p = compute_collision_probability(n, m, cw_min)
                print(f"Theoretical Collision Probability: {p * 100}%")

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
