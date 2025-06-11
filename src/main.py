from src.user_config import UserConfig as cfg_module
from src.sim_params import SimParams as sparams_module

from src.utils.plotters import NetworkPlotter
from src.utils.support import initialize_network, validate_settings, wandb_init
from src.utils.event_logger import get_logger
from src.utils.messages import (
    STARTING_EXECUTION_MSG,
    EXECUTION_TERMINATED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
    PRESS_TO_EXIT_MSG,
    SECTION_DIVIDER_MSG,
)

from codecarbon import EmissionsTracker

import simpy
import matplotlib.pyplot as plt
import json

emissions_tracker = (
    EmissionsTracker(project_name="simulation") if cfg_module.USE_CODECARBON else None
)

if __name__ == "__main__":
    print(STARTING_EXECUTION_MSG)

    logger = get_logger("MAIN", cfg_module, sparams_module)

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
        print(SECTION_DIVIDER_MSG)
        logger.info(
            f"SIMULATION:\n{json.dumps(emissions_tracker.final_emissions_data.__dict__, indent=6)}"
        )

    print(SIMULATION_TERMINATED_MSG)

    plotter = NetworkPlotter(cfg_module, sparams_module, env)
    plotter.plot_network(network)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(EXECUTION_TERMINATED_MSG)
