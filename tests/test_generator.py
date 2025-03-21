from src.user_config import UserConfig as cfg
from src.sim_params import SimParams as sparams

from src.utils.data_units import MPDU, Packet
from src.utils.plotters import TrafficPlotter
from src.utils.event_logger import get_logger
from src.utils.support import (
    initialize_network,
    validate_params,
    validate_config,
    warn_overwriting_enabled_paths,
)
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    PRESS_TO_EXIT_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

import matplotlib.pyplot as plt
import simpy


cfg.SIMULATION_TIME_us = 2e6
cfg.SEED = 1

cfg.ENABLE_CONSOLE_LOGGING = True
cfg.USE_COLORS_IN_LOGS = True
cfg.ENABLE_LOGS_RECORDING = False
cfg.EXCLUDED_LOGS = {"GEN": ["ALL"]}

cfg.ENABLE_FIGS_DISPLAY = True
cfg.ENABLE_FIGS_SAVING = True
cfg.FIGS_SAVE_PATH = "figs/tests"

cfg.ENABLE_TRAFFIC_GEN_RECORDING = True
cfg.TRAFFIC_GEN_RECORDING_PATH = "tests/sim_traces"

cfg.NETWORK_BOUNDS_m = (10, 10, 2)

cfg.BSSs = [
    {
        "id": 1,  # A BSS
        "ap": {"id": 1, "pos": (0, 0, 0)},  # BSS Access Point (AP)
        "stas": [{"id": 2, "pos": (3, 4, 0)}],
        "traffic_flows": [
            {
                "destination": 2,
                "model": {
                    "name": "Poisson",
                    "traffic_load_kbps": 50e3,
                    "max_packet_size_bytes": 1240,
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
                "model": {
                    "name": "Bursty",
                    "end_time_us": 1e6,
                    "traffic_load_kbps": 50e3,
                    "max_packet_size_bytes": 1240,
                    "burst_size_pkts": 30,
                    "avg_inter_packet_time_us": 5,
                },
            }
        ],
    },
    {
        "id": 3,  # Another BSS
        "ap": {"id": 5, "pos": (5, 6, 1)},
        "stas": [{"id": 6, "pos": (1, 2, 2)}],
        "traffic_flows": [
            {
                "destination": 6,
                "model": {"name": "VR", "start_time_us": 2000, "fps": 60},
            }
        ],
    },
]

SAVE_NAMES = {1: "Poisson_traffic", 3: "Bursty_traffic", 5: "VR_traffic"}


class DummyMAC:
    def __init__(self, cfg: cfg, sparams: sparams, env: simpy.Environment):
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

    logger = get_logger("TEST", cfg, sparams)

    validate_params(sparams, logger)
    validate_config(cfg, logger)
    warn_overwriting_enabled_paths(cfg, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = initialize_network(cfg, sparams, env)

    for node in network.get_nodes():
        node.mac_layer = DummyMAC(cfg, sparams, env)

    env.run(until=cfg.SIMULATION_TIME_us)

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> DL Load: {ap.mac_layer.load_bits*1e-6:.2f} Mbits \t DL Rate: {(ap.mac_layer.load_bits*1e-6) / (cfg.SIMULATION_TIME_us*1e-6):.2f} Mbps"
        )

        ap.mac_layer.plotter.plot_generation(
            title=f"Traffic AP {ap.id}", save_name=SAVE_NAMES[ap.id]
        )

    print(SIMULATION_TERMINATED_MSG)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(TEST_COMPLETED_MSG)
