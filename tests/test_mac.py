from src.components.network import Network, Node
from src.utils.event_logger import get_logger
from src.utils.data_units import DataUnit, PPDU
from src.utils.support import initialize_network
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)


import simpy
import importlib
import random
import matplotlib.pyplot as plt

import src.user_config as cfg
import src.sim_params as sparams
import src.utils.event_logger
import src.utils.plotters
import src.components.mac

cfg.ENABLE_CONSOLE_LOGGING = True
cfg.USE_COLORS_IN_LOGS = True
cfg.ENABLE_LOGS_RECORDING = False
cfg.EXCLUDED_LOGS = {"LOAD": ["ALL"]}

cfg.ENABLE_TRAFFIC_GEN_RECORDING = False

sparams.MAX_TX_QUEUE_SIZE_pkts = 100  # Test: 10, 50, 100
sparams.ENABLE_RTS_CTS = True  # Test: False and True

importlib.reload(src.utils.event_logger)
importlib.reload(src.utils.plotters)
importlib.reload(src.components.mac)


SIMULATION_TIME_us = 2e3

IDLE_CHANNEL_PROBABILITY = 0.95  # Test: 1, 0.95, 0.9
MPDU_ERROR_PROBABILITY = 0.5  # Test: 0, 0.1, 0.5, 0.9

SEED = 1
random.seed(SEED)

BSSs = [
    {
        "id": 1,  # A BSS
        "ap": {"id": 1, "pos": (0, 0, 0)},  # BSS Access Point (AP)
        "stas": [{"id": 2, "pos": (3, 4, 0)}],
        "traffic_flows": [
            {
                "destination": 2,
                "file": {"path": "tests/sim_traces/traffic_trace_node_1_to_node_2.csv"},
            },
        ],
    }
]


class DummyPHY:
    def __init__(self, env: simpy.Environment, node: Node):
        self.env = env

        self.node = node

        self.channels_ids = [1]
        self.primary_channel = 1

        self.mcs_index = 11

        self.name = "DummyPHY"
        self.logger = get_logger(self.name, env)

    def transmit(self, data_unit: DataUnit):
        ppdu = PPDU(data_unit, self.env.now)

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Sending {ppdu.type} to node {ppdu.dst_id} over channel(s) {', '.join(map(str, self.channels_ids))}..."
        )

        yield self.env.process(node.medium.transmit(ppdu))

    def receive(self, ppdu: PPDU):
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Received {ppdu.type} from node {ppdu.src_id}"
        )

        ppdu.reception_time_us = self.env.now

        data_unit = ppdu.data_unit

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Forwarding received {data_unit.type} from PHY to MAC..."
        )

        self.node.mac_layer.receive(data_unit)

    def are_channels_idle(self):
        return self.node.medium.are_channels_idle(self.channels_ids)

    def is_primary_channel_idle(self):
        return self.node.medium.are_channels_idle([self.primary_channel])

    def occupy_channels(self):
        pass

    def release_channels(self):
        pass

    def end_nav(self):
        pass

class DummyChannel20MHz:
    def __init__(self, env: simpy.Environment, id: int):
        self.env = env
        self.id = id

    def add_node(self, node: Node):
        pass

class DummyMedium:
    def __init__(self, env, network: Network):
        self.env = env
        self.network = network

        self.channels = {1: DummyChannel20MHz(env, 1)}

    def are_channels_idle(self, channels_ids: list[int]):
        return True if random.random() < IDLE_CHANNEL_PROBABILITY else False
    
    def get_valid_channels(self):
        return [(1,)]

    def transmit(self, ppdu: PPDU):
        # Simulate transmission over the medium
        if (
            ppdu.data_unit.type == "RTS"
            or ppdu.data_unit.type == "CTS"
            or ppdu.data_unit.type == "BACK"
        ):
            yield self.env.timeout(random.randint(1, 10))
        else:
            yield self.env.timeout(random.randint(50, 150))

        if ppdu.data_unit.type == "AMPDU":
            for mpdu in ppdu.data_unit.mpdus:
                mpdu.is_corrupted = (
                    True if MPDU_ERROR_PROBABILITY > random.random() else False
                )

        self.network.get_node(ppdu.dst_id).phy_layer.receive(ppdu)


if __name__ == "__main__":
    print(STARTING_TEST_MSG)
    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    logger = get_logger("TEST", env)

    network = Network(env)
    network.medium = DummyMedium(env, network)

    initialize_network(env, BSSs, network)

    for node in network.get_nodes():
        node.phy_layer = DummyPHY(env, node)

    env.run(until=SIMULATION_TIME_us)

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> Tx attempts:{ap.mac_layer.tx_attempts}, Tx Failures: {ap.mac_layer.tx_failures}, Tx Pkts: {ap.mac_layer.pkts_tx}, Dropped Pkts: {ap.mac_layer.pkts_dropped}"
        )

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
