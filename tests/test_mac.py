from src.user_config import UserConfig as cfg
from src.sim_params import SimParams as sparams


from src.components.network import Network, Node
from src.utils.data_units import DataUnit, PPDU
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
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

import simpy
import random


IDLE_CHANNEL_PROBABILITY = 0.95  # Test: 1, 0.95, 0.9
MPDU_ERROR_PROBABILITY = 0.5  # Test: 0, 0.1, 0.5, 0.9

sparams.MAX_TX_QUEUE_SIZE_pkts = 100  # Test: 10, 50, 100
sparams.ENABLE_RTS_CTS = True  # Test: False and True

cfg.SIMULATION_TIME_us = 2e3
cfg.SEED = 1

cfg.ENABLE_CONSOLE_LOGGING = True
cfg.USE_COLORS_IN_LOGS = True
cfg.ENABLE_LOGS_RECORDING = False
cfg.EXCLUDED_LOGS = {"LOAD": ["ALL"]}

cfg.ENABLE_TRAFFIC_GEN_RECORDING = False

cfg.NETWORK_BOUNDS_m = (10, 10, 2)

cfg.ENABLE_ADVANCED_NETWORK_CONFIG = True

cfg.BSSs_Advanced = [
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
    def __init__(self, cfg: cfg, sparams: sparams, env: simpy.Environment, node: Node):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.node = node

        self.channels_ids = [1]
        self.primary_channel_id = 1

        self.mcs_index = 11

        self.name = "DummyPHY"
        self.logger = get_logger(self.name, cfg, sparams, env)

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

    def receive_channel_info(self, channels_ids: list[int], primary_channel_id: int):
        pass

    def receive_mcs_info(self, mcs_index: int):
        pass

    def is_primary_channel_idle(self):
        return self.node.medium.are_channels_idle([self.primary_channel_id])

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

    def assign_as_primary_channel(self, node: Node, channel_id: int):
        pass

    def release_as_primary_channel(self, node: Node, channel_id: int):
        pass

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

    def broadcast_channel_info(self, src_id, dst_id, channels_ids, primary_channel):
        pass

    def broadcast_mcs_info(self, src_id, dst_id, mcs_index):
        pass


if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg, sparams)

    validate_params(sparams, logger)
    validate_config(cfg, logger)
    warn_overwriting_enabled_paths(cfg, logger)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = Network(cfg, sparams, env)
    network.medium = DummyMedium(env, network)

    network = initialize_network(cfg, sparams, env, network)

    for node in network.get_nodes():
        node.phy_layer = DummyPHY(cfg, sparams, env, node)

    env.run(until=cfg.SIMULATION_TIME_us)

    for ap in network.get_aps():
        logger.info(
            f"AP {ap.id} -> Tx attempts: {ap.tx_stats.tx_attempts}, Tx Failures: {ap.tx_stats.tx_failures}, Tx Pkts: {ap.tx_stats.pkts_tx}, Pkts Success: {ap.tx_stats.pkts_success}, Dropped Pkts: {ap.tx_stats.pkts_dropped_queue_lim + ap.tx_stats.pkts_dropped_retry_lim}"
        )

    print(SIMULATION_TERMINATED_MSG)
    print(TEST_COMPLETED_MSG)
