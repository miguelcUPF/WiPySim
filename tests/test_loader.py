from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module

from src.utils.data_units import MPDU, Packet
from src.utils.plotters import TrafficPlotter
from src.utils.event_logger import get_logger
from src.utils.support import initialize_network, validate_settings
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    PRESS_TO_EXIT_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

from typing import cast

import matplotlib.pyplot as plt
import simpy


cfg_module.SIMULATION_TIME_us = 2e5

cfg_module.ENABLE_FIGS_DISPLAY = True

cfg_module.ENABLE_ADVANCED_NETWORK_CONFIG = True

cfg_module.BSSs_Advanced = [
    {
        "id": 1,  # A BSS
        "ap": {"id": 1, "pos": (0, 0, 0)},  # BSS Access Point (AP)
        "stas": [{"id": 2, "pos": (3, 4, 0)}],
        "traffic_flows": [
            {
                "destination": 2,
                "file": {
                    "path": "tests/sim_traces/traffic_trace_node_1_to_node_2.csv",
                    "start_time_us": 1e6,
                },
            },
        ],
    },
    {
        "id": 2,  # Another BSS
        "ap": {"id": 3, "pos": (5, 5, 1)},
        "stas": [{"id": 4, "pos": (1, 2, 1)}],
        "traffic_flows": [
            {
                "destination": 4,
                "file": {
                    "path": "tests/ws_traces/tshark_processed_traffic.tsv",
                    "end_time_us": 1e6,
                },
            }
        ],
    },
]


class DummyMAC:
    def __init__(self, cfg: cfg_module, sparams: sparams_module, env: simpy.Environment):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.load_bits = 0

        self.plotter = TrafficPlotter(cfg, sparams, env)

    def tx_enqueue(self, packet: Packet):
        mpdu = MPDU(packet, self.env.now)

        self.load_bits += packet.size_bytes * 8

        self.plotter.add_data_unit(mpdu)


if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg_module, sparams_module)

    validate_settings(cfg_module, sparams_module, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = initialize_network(cfg_module, sparams_module, env)

    for node in network.get_nodes():
        node.mac_layer = DummyMAC(cfg_module, sparams_module, env)

    env.run(until=cfg_module.SIMULATION_TIME_us)

    for ap in network.get_aps():
        mac_layer = cast(DummyMAC, ap.mac_layer)
        logger.info(
            f"AP {ap.id} -> DL Load: {mac_layer.load_bits*1e-6:.2f} Mbits \t DL Rate: {(mac_layer.load_bits*1e-6) / (cfg_module.SIMULATION_TIME_us*1e-6):.2f} Mbps"
        )

        mac_layer.plotter.plot_generation(title=f"Traffic AP {ap.id}")

    print(SIMULATION_TERMINATED_MSG)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(TEST_COMPLETED_MSG)
