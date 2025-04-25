from tests._user_config_tests import UserConfig as cfg
from tests._sim_params_tests import SimParams as sparams

from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, validate_settings
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

import simpy


sparams.CW_MIN = 16
sparams.CW_MAX = 2**0 * sparams.CW_MIN

sparams.BONDING_MODE = 1  # Test: 0 and 1

sparams.NUM_CHANNELS = 2

cfg.SIMULATION_TIME_us = 2e5

cfg.ENABLE_CONSOLE_LOGGING = False

cfg.ENABLE_ADVANCED_NETWORK_CONFIG = True

cfg.BSSs_Advanced = [
    {
        "id": 1,  # A BSS
        "ap": {"id": 1, "pos": (0, 0, 0), "channels": [1, 2], "primary_channel": 1},
        "stas": [{"id": 2, "pos": (2, 1, 0)}],
        "traffic_flows": [
            {
                "destination": 2,
                "model": {"name": "Full"},
            },
        ],
    },
    {
        "id": 2,  # Another BSS
        "ap": {"id": 3, "pos": (1, 3, 1), "channels": [1], "primary_channel": 1},
        "stas": [{"id": 4, "pos": (1, 2, 1)}],
        "traffic_flows": [
            {
                "destination": 4,
                "model": {"name": "Full"},
            }
        ],
    },
    {
        "id": 3,  # Another BSS
        "ap": {"id": 5, "pos": (2, 3, 1), "channels": [2], "primary_channel": 2},
        "stas": [{"id": 6, "pos": (1, 0, 1)}],
        "traffic_flows": [
            {
                "destination": 6,
                "model": {"name": "Full"},
            }
        ],
    },
]

if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg, sparams)

    validate_settings(cfg, sparams, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = initialize_network(cfg, sparams, env)

    env.run(until=cfg.SIMULATION_TIME_us)

    network.stats.collect_stats()

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> Tx attempts: {ap.tx_stats.tx_attempts}, Tx Failures: {ap.tx_stats.tx_failures}, Tx Pkts: {ap.tx_stats.pkts_tx}, Pkts Success: {ap.tx_stats.pkts_success}, Dropped Pkts: {ap.tx_stats.pkts_dropped_queue_lim + ap.tx_stats.pkts_dropped_retry_lim}, Channels: [{', '.join(map(str, ap.phy_layer.channels_ids))}], Sensing Channels: {', '.join(map(str, ap.phy_layer.sensing_channels_ids))}, MCS: {ap.phy_layer.mcs_indexes}"
        )

    for node_id, stats in network.stats.per_node_stats.items():
        if node_id in [ap.id for ap in network.get_aps()]:
            logger.info(
                f"AP {node_id} -> Tx Rate (Mbps): {stats['tx']['tx_rate_Mbits_per_sec']:.6f}"
            )
        if node_id in [sta.id for sta in network.get_stas()]:
            logger.info(
                f"STA {node_id} -> RX Rate (Mbps): {stats['rx']['rx_rate_Mbits_per_sec']:.6f}, Rx Effective Throughput (Mbps): {stats['rx']['app_effective_throughput_Mbits_per_sec']:.6f}"
            )

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
