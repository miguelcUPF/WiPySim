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
BASELINE_CHANNELS = os.environ.get("BASELINE_CHANNELS", None)
ENABLE_DCB = os.environ.get("ENABLE_DCB", "False") == "True"
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", f"{STRATEGY}_rl{RL_MODE}_seed{SEED}")

# Config
cfg = cfg_module()
sparams = sparams_module()

cfg.RL_MODE = RL_MODE
cfg.SEED = SEED
cfg.WANDB_RUN_NAME = WANDB_RUN_NAME
cfg.SIMULATION_TIME_us = 60e6
cfg.ENABLE_RL = True
cfg.FIRST_AS_PRIMARY = True
cfg.ENABLE_CONSOLE_LOGGING = False
cfg.DISABLE_SIMULTANEOUS_ACTION_SELECTION = False
cfg.ENABLE_REWARD_DECOMPOSITION = False
cfg.ENABLE_ADVANCED_NETWORK_CONFIG = True
cfg.ENABLE_STATS_COMPUTATION = False
cfg.USE_WANDB = True
cfg.USE_CODECARBON = False
cfg.WANDB_PROJECT_NAME = f"UCBjournal-runs-wipysim{SCENARIO}{'-DCB' if ENABLE_DCB else ''}"

# CW & Channels
sparams.CW_MIN = 16
sparams.CW_MAX = 2**6 * sparams.CW_MIN
sparams.NUM_CHANNELS = 4
sparams.BONDING_MODE = 1 if ENABLE_DCB else 0

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
        "sw_linucb": {"strategy": STRATEGY, "alpha": 0.50, "window_size": 0},
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
    cfg.AGENTS_SETTINGS = settings_mapping[RL_MODE][STRATEGY]

baseline_channels = None
if BASELINE_CHANNELS:
    baseline_channels = [int(ch) for ch in BASELINE_CHANNELS.split(",")]

# Scenario
cfg.BSSs_Advanced = get_scenario(
    SCENARIO, baseline_channels=baseline_channels, seed=SEED
)

DISPLAY_AGENTS_EMISSIONS = True
DISPLAY_SIMULATION_EMISSIONS = True

emissions_tracker = (
    EmissionsTracker(project_name="simulation") if cfg.USE_CODECARBON else None
)

if __name__ == "__main__":
    logger = get_logger("TEST", cfg, sparams)
    logger.disabled = True

    validate_settings(cfg, sparams, logger, skip_warnings=True)
    wandb_init(cfg)

    env = simpy.Environment()
    network = initialize_network(cfg, sparams, env)

    if emissions_tracker:
        emissions_tracker.start()

    env.run(until=cfg.SIMULATION_TIME_us)

    if emissions_tracker:
        emissions_tracker.stop()

    network.stats.collect_stats()

    for ap in network.get_aps():
        if cfg.USE_CODECARBON:
            if ap.mac_layer.rl_controller:
                ap.mac_layer.rl_controller.log_emissions_data()
        if STRATEGY in ["epsilon_greedy", "decay_epsilon_greedy"]:
            if ap.mac_layer.rl_controller:
                ap.mac_layer.rl_controller.log_eps_weight_matrix()
