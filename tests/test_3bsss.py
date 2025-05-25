from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module

from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, validate_settings
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

import simpy

sparams_module.CW_MIN = 4
sparams_module.CW_MAX = 2**0 * sparams_module.CW_MIN

sparams_module.NUM_CHANNELS = 1

cfg_module.SIMULATION_TIME_us = 2e5

cfg_module.NUMBER_OF_BSSS = 3
cfg_module.TRAFFIC_LOAD_kbps = 200e3


if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg_module, sparams_module)

    validate_settings(cfg_module, sparams_module, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = initialize_network(cfg_module, sparams_module, env)

    env.run(until=cfg_module.SIMULATION_TIME_us)

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> Tx attempts: {ap.tx_stats.tx_attempts}, Tx Failures: {ap.tx_stats.tx_failures}, Tx Pkts: {ap.tx_stats.pkts_tx}, Pkts Success: {ap.tx_stats.pkts_success}, Dropped Pkts: {ap.tx_stats.pkts_dropped_queue_lim + ap.tx_stats.pkts_dropped_retry_lim}, Channels: [{', '.join(map(str, ap.phy_layer.channels_ids))}], Sensing Channels: {', '.join(map(str, ap.phy_layer.sensing_channels_ids))}"
        )

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
