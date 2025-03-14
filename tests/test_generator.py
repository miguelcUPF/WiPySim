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
import src.utils.traffic_recorder

sim_config.ENABLE_CONSOLE_LOGGING = True
sim_config.USE_COLORS_IN_EVENT_LOGS = True

sim_config.EXCLUDED_CONSOLE_LEVELS = []
sim_config.EXCLUDED_CONSOLE_MODULES = ["GEN"]

sim_config.ENABLE_EVENT_RECORDING = False

sim_config.ENABLE_FIGS_DISPLAY = True
sim_config.ENABLE_FIGS_SAVING = True
sim_config.FIGS_SAVE_PATH = "figs/tests"
sim_config.ENABLE_FIGS_OVERWRITE = True

sim_config.ENABLE_TRAFFIC_GEN_RECORDING = True
sim_config.TRAFFIC_GEN_RECORD_PATH = "tests/sim_traces"

importlib.reload(src.utils.event_logger)
importlib.reload(src.utils.traffic_recorder)
importlib.reload(src.utils.plotters)


logger = get_logger("MAIN")

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


def run_tests(env, traffic_model, traffic_generator, mac_layer):
    mac_layer.reset_load()
    match traffic_model:
        case "Poisson":
            env.process(
                traffic_generator.generate_poisson_traffic())
        case "Bursty":
            env.process(
                traffic_generator.generate_bursty_traffic())
        case 'VR':
            env.process(traffic_generator.generate_vr_traffic())
        case _:
            raise ValueError(
                "Invalid traffic model specified.")

    env.run(until=SIMULATION_TIME_us)


if __name__ == "__main__":
    print("\033[93m\n" + "="*24 + "  TEST STARTED  " + "="*24 + "\033[0m")

    traffic_models = ["Poisson", "Bursty", "VR"]

    for traffic_model in traffic_models:

        logger.header(f"Starting {traffic_model} Traffic Generator...")

        env = simpy.Environment()
        update_logger_env(env)

        app_layer = APP(env, 1)
        mac_layer = DummyMac(env)

        traffic_generator = TrafficGenerator(env, 1, 2, app_layer)
        app_layer.mac_layer = mac_layer

        run_tests(env, traffic_model, traffic_generator, mac_layer)
        logger.info(
            f"Load: {mac_layer.load_bits*1e-6:.2f} Mbits \t Rate: {(mac_layer.load_bits*1e-6) / (SIMULATION_TIME_us*1e-6):.2f} Mbps")
        mac_layer.plotter.show_generation(
            title=f"{traffic_model} Traffic Generation", save_name=f"{traffic_model}_traffic")

    print("\033[93m\n" + "="*23 + "  TEST COMPLETED  " + "="*23)
    if len(plt.get_fignums()) > 0:
        input(f"{'='*10}  Press Enter to exit and close all plots   {'='*10}")
    print("="*20 + "  EXECUTION TERMINATED  " + "="*20 + "\033[0m")
