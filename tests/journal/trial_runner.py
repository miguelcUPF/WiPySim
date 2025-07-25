from codecarbon import EmissionsTracker

from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module

from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, validate_settings, wandb_init

from tests.journal.scenarios_config import get_scenario

import os
import simpy

# Load env vars
STRATEGY = os.environ.get("STRATEGY", "sw_linucb")
RL_MODE = int(os.environ.get("RL_MODE", 1))
SEED = int(os.environ.get("SEED", 1))
SCENARIO = os.environ.get("SCENARIO", "A")
ALL_RL_DRIVEN = int(os.environ.get("ALL_RL_DRIVEN", 0))
AP1_CHANNELS_ENV = os.environ.get("AP1_CHANNELS", None)
WANDB_RUN_NAME = os.environ.get(
    "WANDB_RUN_NAME", f"{SCENARIO}_all{ALL_RL_DRIVEN}_{STRATEGY}_rl{RL_MODE}_seed{SEED}"
)

# Config
cfg_module.RL_MODE = RL_MODE
cfg_module.SEED = SEED
cfg_module.WANDB_RUN_NAME = WANDB_RUN_NAME
cfg_module.SIMULATION_TIME_us = 120e6
cfg_module.ENABLE_RL = True
cfg_module.FIRST_AS_PRIMARY = True
cfg_module.ENABLE_CONSOLE_LOGGING = False
cfg_module.DISABLE_SIMULTANEOUS_ACTION_SELECTION = False
cfg_module.ENABLE_REWARD_DECOMPOSITION = False
cfg_module.ENABLE_ADVANCED_NETWORK_CONFIG = True
cfg_module.ENABLE_STATS_COMPUTATION = False
cfg_module.USE_WANDB = True
cfg_module.USE_CODECARBON = False
cfg_module.WANDB_PROJECT_NAME = "journal-wipysim"

# CW & Channels
sparams_module.CW_MIN = 16
sparams_module.CW_MAX = 2**6 * sparams_module.CW_MIN
sparams_module.NUM_CHANNELS = 4

# Agent settings
settings_mapping = {
    0: {
        "ucb": {"strategy": STRATEGY, "alpha": 1.096},
        "sw_linucb": {"strategy": STRATEGY, "alpha": 0.52, "window_size": 0},
        "epsilon_greedy": {
            "strategy": STRATEGY,
            "epsilon": 0.020,
            "eta": 0.086,
            "gamma": 0.87,
            "alpha_ema": 0.22,
        },
    },
    1: {
        "ucb": {"strategy": STRATEGY, "alpha": 1.14},
        "osub": {"strategy": STRATEGY},
        "sw_osub": {"strategy": STRATEGY, "window_size": 71},
        "sw_linucb": {"strategy": STRATEGY, "alpha": 0.361, "window_size": 30},
        "epsilon_greedy": {
            "strategy": STRATEGY,
            "epsilon": 0.038,
            "eta": 0.069,
            "gamma": 0.788,
            "alpha_ema": 0.245,
        },
    },
}

if STRATEGY in settings_mapping[RL_MODE]:
    cfg_module.AGENTS_SETTINGS = settings_mapping[RL_MODE][STRATEGY]

ap1_channels = None
if AP1_CHANNELS_ENV:
    ap1_channels = [int(ch) for ch in AP1_CHANNELS_ENV.split(",")]

# Scenario
cfg_module.BSSs_Advanced = get_scenario(
    SCENARIO, all_rl_driven=ALL_RL_DRIVEN, ap1_channels=ap1_channels, seed=SEED
)

DISPLAY_AGENTS_EMISSIONS = True
DISPLAY_SIMULATION_EMISSIONS = True

emissions_tracker = (
    EmissionsTracker(project_name="simulation") if cfg_module.USE_CODECARBON else None
)

if __name__ == "__main__":
    logger = get_logger("TEST", cfg_module, sparams_module)
    logger.disabled = True

    validate_settings(cfg_module, sparams_module, logger, skip_warnings=True)
    wandb_init(cfg_module)

    env = simpy.Environment()
    network = initialize_network(cfg_module, sparams_module, env)

    if emissions_tracker:
        emissions_tracker.start()

    env.run(until=cfg_module.SIMULATION_TIME_us)

    if emissions_tracker:
        emissions_tracker.stop()

    network.stats.collect_stats()

    for ap in network.get_aps():
        if cfg_module.USE_CODECARBON:
            if ap.mac_layer.rl_controller:
                ap.mac_layer.rl_controller.log_emissions_data()
        if STRATEGY in ["epsilon_greedy", "decay_epsilon_greedy"]:
            if ap.mac_layer.rl_controller:
                ap.mac_layer.rl_controller.log_eps_weight_matrix()
