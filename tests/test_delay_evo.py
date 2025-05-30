from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module

from src.utils.event_logger import get_logger
from src.utils.plotters import DelayPerLoadPlotter
from src.utils.support import initialize_network, validate_settings
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
    PRESS_TO_EXIT_MSG,
)

from tqdm import tqdm

import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd
import simpy
import os

sparams_module.MAX_TX_QUEUE_SIZE_pkts = 50

sparams_module.NUM_CHANNELS = 1

cfg_module.SIMULATION_TIME_us = 2e6

cfg_module.ENABLE_CONSOLE_LOGGING = False

cfg_module.ENABLE_FIGS_DISPLAY = True
cfg_module.ENABLE_FIGS_SAVING = True

cfg_module.ENABLE_ADVANCED_NETWORK_CONFIG = True

LOADS_kbps = [
    2e3,
    10e3,
    20e3,
    40e3,
    60e3,
    80e3,
    100e3,
    120e3,
    140e3,
    160e3,
    180e3,
    200e3,
    220e3,
    240e3,
    260e3,
    280e3,
    300e3,
]


def run_simulation(cfg: cfg_module, sparams: sparams_module, load_kbps: float) -> tuple:
    cfg.BSSs_Advanced = [
        {
            "id": 1,  # A BSS
            "ap": {"id": 1, "pos": (0, 0, 0)},
            "stas": [{"id": 2, "pos": (3, 4, 0)}],
            "traffic_flows": [
                {
                    "destination": 2,
                    "model": {"name": "Poisson", "traffic_load_kbps": load_kbps},
                },
            ],
        },
    ]

    env = simpy.Environment()

    network = initialize_network(cfg, sparams, env)

    env.run(until=cfg.SIMULATION_TIME_us)

    network.stats.collect_stats()

    stas_rx_packets_history = {}

    for ap in network.get_aps():
        print(ap.tx_stats.pkts_dropped_queue_lim)

    for sta in network.get_stas():
        df = sta.rx_stats.rx_packets_history.copy()
        df["delay_us"] = df["reception_time_us"] - df["creation_time_us"]
        stas_rx_packets_history[sta.id] = df
    return load_kbps, stas_rx_packets_history


if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg_module, sparams_module)

    validate_settings(cfg_module, sparams_module, logger)

    print(STARTING_SIMULATION_MSG)

    results_flat = {load_kbps: {} for load_kbps in LOADS_kbps}

    params_list = [(cfg_module, sparams_module, load_kbps) for load_kbps in LOADS_kbps]

    batch_size = 5  # Adjust batch size based on available CPU cores
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count() // 2
    ) as executor:
        for i in tqdm(range(0, len(params_list), batch_size)):
            batch = params_list[i : i + batch_size]
            futures = [executor.submit(run_simulation, *params) for params in batch]

            for future in concurrent.futures.as_completed(futures):
                load_kbps, stas_rx_packets_history = future.result()
                results_flat[load_kbps] = pd.concat(
                    [df for df in stas_rx_packets_history.values()],
                    axis=0,
                    ignore_index=True,
                )

                logger.info(f"Load: {load_kbps/1e3} Mbps")
                for sta_id, df in stas_rx_packets_history.items():
                    if len(df) == 0:
                        logger.error(f"\tSTA {sta_id} -> No packets received!")
                        continue
                    elif len(df) == 1:
                        logger.warning(
                            f"\tSTA {sta_id} -> Only one packet received! Cannot compute STD!"
                        )
                    logger.info(
                        f"\tSTA {sta_id} -> Delay (mean): {df['delay_us'].mean():.2f} μs"
                    )
                    logger.info(
                        f"\tSTA {sta_id} -> Delay (max): {df['delay_us'].max():.2f} μs"
                    )
                    logger.info(
                        f"\tSTA {sta_id} -> Delay (min): {df['delay_us'].min():.2f} μs"
                    )
                    logger.info(
                        f"\tSTA {sta_id} -> Delay (std): {df['delay_us'].std():.2f} μs"
                    )
                    logger.info(
                        f"\tSTA {sta_id} -> Delay (95%): {df['delay_us'].quantile(0.95):.2f} μs"
                    )
                    logger.info(
                        f"\tSTA {sta_id} -> Delay (99%): {df['delay_us'].quantile(0.99):.2f} μs"
                    )

    save_name = f"mean_delay_vs_load"
    DelayPerLoadPlotter(cfg_module, sparams_module).plot_mean_delay(
        results_flat, LOADS_kbps, save_name
    )

    print(SIMULATION_TERMINATED_MSG)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(TEST_COMPLETED_MSG)
