from src.utils.data_units import MPDU, Packet
from src.utils.plotters import TrafficPlotter
from src.utils.event_logger import get_logger, update_logger_env
from src.traffic.loader import TrafficLoader
from src.components.app import APP
import simpy
import importlib
import matplotlib.pyplot as plt
import src.sim_config as sim_config
import src.utils.event_logger
import src.utils.plotters

sim_config.ENABLE_CONSOLE_LOGGING = True
sim_config.USE_COLORS_IN_EVENT_LOGS = True

sim_config.EXCLUDED_CONSOLE_LEVELS = ["DEBUG"]
sim_config.EXCLUDED_CONSOLE_MODULES = []

sim_config.ENABLE_EVENT_RECORDING = False

sim_config.ENABLE_FIGS_DISPLAY = True
sim_config.ENABLE_FIGS_SAVING = False


sim_config.ENABLE_TRAFFIC_GEN_RECORDING = False

importlib.reload(src.utils.event_logger)
importlib.reload(src.utils.plotters)


logger = get_logger("MAIN")

TRAFFIC_LOAD_CONFIG = [
    {
        "source": 1,
        "destinations": [
            {
                "destination": 2,
                "traffic_files": [
                    {"file": "tests/sim_traces/traffic_trace_node_1_to_node_2.csv",
                        "start_time_us": 5000},  # Start after 5000 us
                    {"file": "tests/sim_traces/traffic_trace_node_3_to_node_1.csv"}
                ]
            }
        ]
    },
    {
        "source": 2,
        "destinations": [
            {"destination": 1,
                "traffic_files": [
                    {"start_time_us": 1000},
                    {"file": "nonexistingfile.csv"},
                ]
             },
            {
                "destination": 3,
                "traffic_files": [
                    {"file": "tests/ws_traces/tshark_processed_traffic.tsv",
                        "start_time_us": 2000}
                ]
            }
        ]
    }
]

SIMULATION_TIME_us = 1e6


class DummyMac:
    def __init__(self, env):
        self.env = env
        self.load_bits = 0
        self.plotter = TrafficPlotter()

    def tx_enqueue(self, packet: Packet):
        mpdu = MPDU(packet, self.env.now)
        self.load_bits += packet.size_bytes * 8
        self.plotter.add_data_unit(mpdu)

    def reset_load(self):
        self.load_bits = 0


if __name__ == "__main__":
    print("\033[93m\n" + "="*24 + "  TEST STARTED  " + "="*24 + "\033[0m")
    logger.header(f"Starting Traffic Loader...")

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

    for config in TRAFFIC_LOAD_CONFIG:
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

            for tf_config in dest_config["traffic_files"]:
                filepath = tf_config.get("file")
                start_time_us = tf_config.get("start_time_us", 0)

                if not filepath:
                    logger.warning(
                        f"File not specified for node {source} to node {destination}")
                    continue

                TrafficLoader(env, source, destination,
                              app_layer, filepath, start_time_us)

    env.run(until=SIMULATION_TIME_us)

    for node_id, mac_layer in nodes_mac_layer.items():
        logger.header(f"Traffic Node {node_id}")
        logger.info(
            f"Load: {mac_layer.load_bits*1e-6:.2f} Mbits \t Rate: {(mac_layer.load_bits*1e-6) / (SIMULATION_TIME_us*1e-6):.2f} Mbps")

        mac_layer.plotter.show_generation(
            title=f"Traffic Node {node_id}", save_name=f"loaded_traffic_node_{node_id}")

    print("\033[93m\n" + "="*23 + "  TEST COMPLETED  " + "="*23)
    if len(plt.get_fignums()) > 0:
        input(f"{'='*10}  Press Enter to exit and close all plots   {'='*10}")
    print("="*20 + "  EXECUTION TERMINATED  " + "="*20 + "\033[0m")
