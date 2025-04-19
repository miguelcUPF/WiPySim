from src.user_config import UserConfig as cfg
from src.sim_params import SimParams as sparams

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

sparams.CW_MIN = 4
sparams.CW_MAX = 2**0 * sparams.CW_MIN

sparams.BONDING_MODE = 0  # Test: 0 and 1

sparams.NUM_CHANNELS = 2

cfg.SIMULATION_TIME_us = 2e5
cfg.SEED = 2

cfg.ENABLE_CONSOLE_LOGGING = False
cfg.USE_COLORS_IN_LOGS = True
cfg.ENABLE_LOGS_RECORDING = False
cfg.EXCLUDED_LOGS = {
    "GEN": ["ALL"],
}

cfg.ENABLE_TRAFFIC_GEN_RECORDING = False

cfg.NETWORK_BOUNDS_m = (10, 10, 2)
cfg.NUMBER_OF_BSSS = 3
cfg.TRAFFIC_MODEL = "Poisson"
cfg.TRAFFIC_LOAD_kbps = 100e3

cfg.ENABLE_ADVANCED_NETWORK_CONFIG = False


if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg, sparams)

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
    print(TEST_COMPLETED_MSG)
