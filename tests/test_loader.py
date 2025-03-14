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

sim_config.EXCLUDED_CONSOLE_LEVELS = []
sim_config.EXCLUDED_CONSOLE_MODULES = ["LOAD"]

sim_config.ENABLE_EVENT_RECORDING = False

sim_config.ENABLE_FIGS_DISPLAY = True
sim_config.ENABLE_FIGS_SAVING = False


sim_config.ENABLE_TRAFFIC_GEN_RECORDING = False

importlib.reload(src.utils.event_logger)
importlib.reload(src.utils.plotters)


logger = get_logger("MAIN")

ENABLE_TRAFFIC_LOADING = True
TRAFFIC_SOURCES_LOADING = {
    1: {  # Node 1 as a source
        "destinations": [2, 3],  # Destination nodes
        "traffic_paths": [  # Traffic trace files
            "tests/sim_traces/traffic_trace_node_1_to_node_2_VR.csv",
            "tests/sim_traces/traffic_trace_node_1_to_node_2_Poisson.csv"
        ]
    },
    2: {  # Node 2 as a source
        "destinations": [1],  # Destination nodes
        "traffic_paths": [  # Traffic trace files
            "tests/ws_traces/tshark_processed_traffic.tsv",
        ]
    }
}

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
    if ENABLE_TRAFFIC_LOADING:
        logger.header(f"Starting Traffic Loader...")

        env = simpy.Environment()
        update_logger_env(env)

        nodes_app_layer = {}
        nodes_mac_layer = {}
        for source, config in TRAFFIC_SOURCES_LOADING.items():
            if source not in nodes_app_layer:
                app_layer = APP(env, source)
                nodes_app_layer[source] = app_layer
            else:
                app_layer = nodes_app_layer[source]

            if source not in nodes_mac_layer:
                mac_layer = DummyMac(env)
                nodes_mac_layer[source] = mac_layer
            else:
                mac_layer = nodes_mac_layer[source]

            app_layer.mac_layer = mac_layer

            for destination, file_path in zip(config["destinations"], config["traffic_paths"]):
                traffic_loader = TrafficLoader(
                    env, source, destination, file_path, app_layer
                )

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
