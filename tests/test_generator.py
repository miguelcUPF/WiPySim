from src.utils.data_units import MPDU, Packet
from src.utils.plotters import TrafficPlotter
from src.utils.event_logger import get_logger, update_logger_env
from src.traffic.generator import TrafficGenerator
from src.components.app import APP
import simpy
import importlib
import matplotlib.pyplot as plt
import src.sim_config as sim_config
import src.utils.event_logger
import src.utils.plotters
import src.traffic.recorder

sim_config.ENABLE_CONSOLE_LOGGING = True
sim_config.USE_COLORS_IN_EVENT_LOGS = True

sim_config.EXCLUDED_CONSOLE_LEVELS = ["DEBUG"]
sim_config.EXCLUDED_CONSOLE_MODULES = []

sim_config.ENABLE_EVENT_RECORDING = False

sim_config.ENABLE_FIGS_DISPLAY = True
sim_config.ENABLE_FIGS_SAVING = True
sim_config.FIGS_SAVE_PATH = "figs/tests"
sim_config.ENABLE_FIGS_OVERWRITE = True

sim_config.ENABLE_TRAFFIC_GEN_RECORDING = True
sim_config.TRAFFIC_GEN_RECORD_PATH = "tests/sim_traces"

importlib.reload(src.utils.event_logger)
importlib.reload(src.traffic.recorder)
importlib.reload(src.utils.plotters)


logger = get_logger("MAIN")

TRAFFIC_GEN_CONFIG = [
    {
        "source": 1,  # Source node 1
        "destinations": [  # List of destinations for source node 1
            {
                "destination": 2,  # Destination node 2
                "models": [  # List of traffic models to generate from node 1 to node 2
                    {
                        "model": "Poisson",  # Poisson model
                        "traffic_load_kbps": 50e3,
                        "packet_size_bytes": 1280
                    }
                ]
            },
        ]
    },
    {
        "source": 2,  # Source node 2
        "destinations": [  # List of destinations for source node 2
            {
                "destination": 3,  # Destination node 3
                "models": [  # List of traffic models to generate from node 2 to node 3
                    {
                        "model": "Bursty",  # Bursty model
                        "traffic_load_kbps": 50e3,
                        "packet_size_bytes": 1024,
                        "burst_size_pkts": 30  # Number of packets in each burst
                    }
                ]
            }
        ]
    },
    {
        "source": 3,  # Source node 3
        "destinations": [  # List of destinations for source node 3
            {
                "destination": 1,  # Destination node 1
                "models": [  # List of traffic models to generate from node 3 to node 1
                    {
                        "model": "VR",  # VR model
                        "start_time_us": 2000,  # Start time in microseconds for VR model
                        "fps": 60  # Frames per second for VR model
                    }
                ]
            }
        ]
    },
    {
        "source": 4,  # Source node 4
        "destinations": [  # List of destinations for source node 4
            {
                "destination": 2,  # Destination node 2
                "models": [  # List of traffic models to generate from node 4 to node 2
                    {
                        "model": "Poisson"
                    },
                    {
                        "model": "Bursty"
                    },
                    {
                        "model": "VR"
                    },
                    {
                        "model": "nonexistingmodel"
                    }
                ]
            },
        ]
    },
]

SAVE_NAMES = ["Poisson_traffic", "Bursty_traffic", "VR_traffic", None]

SIMULATION_TIME_us = 2e6


class DummyMac:
    def __init__(self, env):
        self.env = env
        self.load_bits = 0
        self.plotter = TrafficPlotter(env)

    def tx_enqueue(self, packet: Packet):
        mpdu = MPDU(packet, self.env.now)
        self.load_bits += packet.size_bytes * 8
        self.plotter.add_data_unit(mpdu)

    def reset_load(self):
        self.load_bits = 0


if __name__ == "__main__":
    print("\033[93m\n" + "="*24 + "  TEST STARTED  " + "="*24 + "\033[0m")

    logger.header(f"Starting Traffic Generator...")

    env = simpy.Environment()
    update_logger_env(env)

    nodes_app_layer = {}
    nodes_mac_layer = {}

    def initialize_layers(source):
        app_layer = nodes_app_layer.get(source) or APP(env, source)
        mac_layer = nodes_mac_layer.get(source) or DummyMac(env)

        nodes_app_layer[source] = app_layer
        nodes_mac_layer[source] = mac_layer

        app_layer.mac_layer = mac_layer
        return app_layer, mac_layer

    for config in TRAFFIC_GEN_CONFIG:
        source = config.get("source")

        if not source:
            continue

        app_layer, mac_layer = initialize_layers(source)

        for dest_config in config["destinations"]:
            destination = dest_config.get("destination")

            if not destination:
                logger.warning(
                    f"Destination not specified for node {source}!")
                continue

            for model_config in dest_config["models"]:
                TrafficGenerator(env, source, destination,
                                 app_layer, model_config["model"], **model_config)

    env.run(until=SIMULATION_TIME_us)

    for i, (node_id, mac_layer) in enumerate(nodes_mac_layer.items()):
        logger.header(f"Traffic Node {node_id}")
        logger.info(
            f"Node {node_id} -> Load: {mac_layer.load_bits*1e-6:.2f} Mbits \t Rate: {(mac_layer.load_bits*1e-6) / (SIMULATION_TIME_us*1e-6):.2f} Mbps")

        if SAVE_NAMES[i]:
            mac_layer.plotter.show_generation(
                title=f"Traffic Node {node_id}", save_name=SAVE_NAMES[i])
            
    print("\033[93m\n" + "="*23 + "  TEST COMPLETED  " + "="*23)
    if len(plt.get_fignums()) > 0:
        input(f"{'='*10}  Press Enter to exit and close all plots   {'='*10}")
    print("="*20 + "  EXECUTION TERMINATED  " + "="*20 + "\033[0m")
