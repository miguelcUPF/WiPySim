from src.user_config import UserConfig as cfg
from src.sim_params import SimParams as sparams

from src.utils.plotters import NetworkPlotter
from src.utils.support import initialize_network, validate_params, validate_config, warn_overwriting_enabled_paths
from src.utils.event_logger import get_logger
from src.utils.messages import (
    STARTING_EXECUTION_MSG,
    EXECUTION_TERMINATED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
    PRESS_TO_EXIT_MSG
)

import simpy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print(STARTING_EXECUTION_MSG)
    
    logger = get_logger("MAIN", cfg, sparams)

    validate_params(sparams, logger)
    validate_config(cfg, logger)
    warn_overwriting_enabled_paths(cfg, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()
    network = initialize_network(cfg, sparams, env)

    env.run(until=cfg.SIMULATION_TIME_us)

    print(SIMULATION_TERMINATED_MSG)

    plotter = NetworkPlotter(cfg, sparams, env)
    plotter.plot_network(network)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(EXECUTION_TERMINATED_MSG)
