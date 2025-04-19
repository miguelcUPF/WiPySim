from src.user_config import UserConfig as cfg
from src.sim_params import SimParams as sparams

from src.utils.plotters import NetworkPlotter
from src.utils.support import initialize_network, validate_settings
from src.utils.event_logger import get_logger
from src.utils.messages import (
    STARTING_EXECUTION_MSG,
    EXECUTION_TERMINATED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
    PRESS_TO_EXIT_MSG,
)

import simpy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print(STARTING_EXECUTION_MSG)

    logger = get_logger("MAIN", cfg, sparams)

    validate_settings(cfg, sparams, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()
    network = initialize_network(cfg, sparams, env)

    env.run(until=cfg.SIMULATION_TIME_us)

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> Tx attempts: {ap.tx_stats.tx_attempts}, Tx Failures: {ap.tx_stats.tx_failures}, Tx Pkts: {ap.tx_stats.pkts_tx}, Pkts Success: {ap.tx_stats.pkts_success}, Dropped Pkts: {ap.tx_stats.pkts_dropped_queue_lim + ap.tx_stats.pkts_dropped_retry_lim}, Channels: [{', '.join(map(str, ap.phy_layer.channels_ids))}], Sensing Channels: {', '.join(map(str, ap.phy_layer.sensing_channels_ids))}"
        )

    print(SIMULATION_TERMINATED_MSG)

    plotter = NetworkPlotter(cfg, sparams, env)
    plotter.plot_network(network)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(EXECUTION_TERMINATED_MSG)
