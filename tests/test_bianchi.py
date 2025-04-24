from src.user_config import UserConfig as cfg
from src.sim_params import SimParams as sparams

from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, validate_settings
from src.utils.bianchis import compute_collision_probability
from src.utils.plotters import CollisionProbPlotter
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
    PRESS_TO_EXIT_MSG,
)

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import simpy
import os


sparams.MAX_TX_QUEUE_SIZE_pkts = 100
sparams.ENABLE_RTS_CTS = True
sparams.MPDU_ERROR_PROBABILITY = 0.1

sparams.BONDING_MODE = 0

sparams.NUM_CHANNELS = 1

cfg.SIMULATION_TIME_us = 2e6
cfg.SEED = 1
cfg.ENABLE_RL = False
cfg.USE_WANDB = False

cfg.ENABLE_CONSOLE_LOGGING = False
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
cfg.EXCLUDED_IDS = []

cfg.ENABLE_TRAFFIC_GEN_RECORDING = False

cfg.ENABLE_STATS_COLLECTION = False

cfg.ENABLE_FIGS_DISPLAY = True
cfg.ENABLE_FIGS_SAVING = True
cfg.FIGS_SAVE_PATH = "figs/tests"


cfg.NETWORK_BOUNDS_m = (10, 10, 2)
cfg.TRAFFIC_MODEL = "Poisson"
cfg.TRAFFIC_LOAD_kbps = 300e3

cfg.ENABLE_ADVANCED_NETWORK_CONFIG = False


N = range(1, 21)
M = [0, 1, 2, 3, 4]
CW_MIN = [4]  # test values: 4, 8, 16, 32, 64


def run_simulation(cfg: cfg, sparams: sparams, n: int, m: int, cw_min: int) -> tuple:
    sparams.CW_MIN = cw_min
    sparams.CW_MAX = 2**m * cw_min

    cfg.NUMBER_OF_BSSS = n

    env = simpy.Environment()

    network = initialize_network(cfg, sparams, env)

    env.run(until=cfg.SIMULATION_TIME_us)

    network.stats.collect_stats()

    tx_attempts = np.array(
        [s["tx"]["tx_attempts"] for s in network.stats.per_node_stats.values()]
    )
    tx_failures = np.array(
        [s["tx"]["tx_failures"] for s in network.stats.per_node_stats.values()]
    )
    valid_mask = tx_attempts > 0
    simulated_p = (
        np.mean(tx_failures[valid_mask] / tx_attempts[valid_mask])
        if np.any(valid_mask)
        else 0
    )
    theoretical_p = compute_collision_probability(n, m, cw_min)

    return n, m, cw_min, simulated_p, theoretical_p


if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg, sparams)

    validate_settings(cfg, sparams, logger)

    print(STARTING_SIMULATION_MSG)

    col_prob_results = {cw_min: {m: {} for m in M} for cw_min in CW_MIN}

    params_list = [
        (cfg(), sparams(), n, m, cw_min) for n in N for m in M for cw_min in CW_MIN
    ]

    batch_size = 5  # Adjust batch size based on available CPU cores
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count() // 2
    ) as executor:
        for i in tqdm(range(0, len(params_list), batch_size)):
            batch = params_list[i : i + batch_size]
            futures = [executor.submit(run_simulation, *params) for params in batch]

            for future in concurrent.futures.as_completed(futures):
                n, m, cw_min, simulated_p, theoretical_p = future.result()
                col_prob_results[cw_min][m][n] = {
                    "simulated": simulated_p,
                    "theoretical": theoretical_p,
                }

                logger.info(f"n: {n}, m: {m}, cw_min: {cw_min}")
                logger.info(
                    f"Simulation Collision Probability: {simulated_p * 100:.2f}%"
                )
                logger.info(
                    f"Theoretical Collision Probability: {theoretical_p * 100:.2f}%"
                )

    save_name = (
        f"collision_prob_cw_min_{CW_MIN[0]}" if len(CW_MIN) == 1 else "collision_prob"
    )
    CollisionProbPlotter(cfg, sparams).plot_prob(
        col_prob_results, M, CW_MIN, save_name=save_name
    )

    print(SIMULATION_TERMINATED_MSG)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(TEST_COMPLETED_MSG)
