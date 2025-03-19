from src.utils.data_units import MPDU, Packet
from src.utils.plotters import TrafficPlotter
from src.utils.event_logger import get_logger
from src.components.network import Network
from src.utils.support import initialize_network
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    PRESS_TO_EXIT_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

import simpy
import importlib
import matplotlib.pyplot as plt

import src.user_config as cfg
import src.utils.event_logger
import src.utils.plotters

cfg.ENABLE_CONSOLE_LOGGING = True
cfg.USE_COLORS_IN_LOGS = True
cfg.ENABLE_LOGS_RECORDING = False
cfg.EXCLUDED_LOGS = {"LOAD": ["ALL"]}

cfg.ENABLE_FIGS_DISPLAY = True
cfg.ENABLE_FIGS_SAVING = False

cfg.ENABLE_TRAFFIC_GEN_RECORDING = False

importlib.reload(src.utils.event_logger)
importlib.reload(src.utils.plotters)

SIMULATION_TIME_us = 2e6

BSSs = [
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
        "ap": {"id": 3, "pos": (5, 5, 5)},
        "stas": [{"id": 4, "pos": (1, 2, 3)}],
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
    def __init__(self, env):
        self.env = env

        self.load_bits = 0

        self.plotter = TrafficPlotter(env)

    def tx_enqueue(self, packet: Packet):
        mpdu = MPDU(packet, self.env.now)

        self.load_bits += packet.size_bytes * 8

        self.plotter.add_data_unit(mpdu)


if __name__ == "__main__":
    print(STARTING_TEST_MSG)
    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    logger = get_logger("TEST", env)

    network = Network(env)

    initialize_network(env, BSSs, network)

    for node in network.get_nodes():
        node.mac_layer = DummyMAC(env)

    env.run(until=SIMULATION_TIME_us)

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> DL Load: {ap.mac_layer.load_bits*1e-6:.2f} Mbits \t DL Rate: {(ap.mac_layer.load_bits*1e-6) / (SIMULATION_TIME_us*1e-6):.2f} Mbps"
        )

        ap.mac_layer.plotter.show_generation(title=f"Traffic AP {ap.id}")

    print(SIMULATION_TERMINATED_MSG)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(TEST_COMPLETED_MSG)
