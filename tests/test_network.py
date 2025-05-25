from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module

from src.utils.event_logger import get_logger
from src.components.network import Network, Node
from src.utils.plotters import NetworkPlotter
from src.utils.support import initialize_network
from src.utils.messages import (
    STARTING_TEST_MSG,
    TEST_COMPLETED_MSG,
    PRESS_TO_EXIT_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

import simpy
import matplotlib.pyplot as plt

cfg_module.ENABLE_FIGS_DISPLAY = True
cfg_module.ENABLE_FIGS_SAVING = True

cfg_module.ENABLE_ADVANCED_NETWORK_CONFIG = True

cfg_module.BSSs_Advanced = [
    {
        "id": 1,  # A BSS
        "ap": {"id": 1, "pos": (0, 0, 0)},  # BSS Access Point (AP)
        "stas": [
            {"id": 2, "pos": (3, 4, 0)},  # Associated Stations (STAs)
            {"id": 3, "pos": (6, 8, 1)},
        ],
    },
    {
        "id": 2,  # Another BSS
        "ap": {"id": 4, "pos": (5, 5, 1)},
        "stas": [
            {"id": 5, "pos": (1, 2, 1)},
            {"id": 6, "pos": (5, 0, 1)},
            {"id": 7, "pos": (0, 5, 1)},
        ],
    },
    {
        "id": 3,  # Another BSS
        "ap": {"id": 8, "pos": (9, 10, 1)},
        "stas": [{"id": 9, "pos": (1, 2, 2)}],
    },
]


class DummyTrafficSource:
    def __init__(
        self,
        cfg: cfg_module,
        sparams: sparams_module,
        env: simpy.Environment,
        node: Node,
        src_id: int,
        dst_id: int,
    ):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.node = node

        self.src_id = src_id
        self.dst_id = dst_id

    def stop(self):
        return


if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg_module, sparams_module)

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()

    network = Network(cfg_module, sparams_module, env)
    # Test creating BSSs
    logger.header(f"Creating BSSs...")
    logger.info(f"APs in Network (before): {list(ap.id for ap in network.get_aps())}")
    logger.info(
        f"STAs in Network (before): {list(sta.id for sta in network.get_stas())}"
    )

    network = initialize_network(cfg_module, sparams_module, env)

    logger.info(f"APs in Network (after): {list(ap.id for ap in network.get_aps())}")
    logger.info(
        f"STAs in Network (after): {list(sta.id for sta in network.get_stas())}"
    )

    assert len(network.get_nodes()) == 9, logger.error(
        f"Expected 9 Nodes, got {len(network.get_nodes())}"
    )

    # Test updating nodes positions
    logger.header(f"Updating Node 1 and 2 Positions...")
    logger.info(
        f"Node 1 and 2 positions (before): Node 1 -> {network.get_node(1).position}, Node 2 -> {network.get_node(2).position}"
    )

    network.update_node_position(1, (1, 0, 1))
    network.update_node_position(2, (3, 3, 1))

    logger.info(
        f"Node 1 and 2 positions (after): Node 1 -> {network.get_node(1).position}, Node 2 -> {network.get_node(2).position}"
    )

    assert network.get_node(1).position == (1, 0, 1), logger.error(
        f"Expected node 1 position to be (1, 0, 1), got {network.get_node(1).position}"
    )
    assert network.get_node(2).position == (3, 3, 1), logger.error(
        f"Expected node 2 position to be (3, 3, 1), got {network.get_node(2).position}"
    )

    # Test removing nodes
    logger.header(f"Removing Node 7 and 8...")
    logger.info(
        f"Nodes in Network (before): {list(node.id for node in network.get_nodes())}"
    )

    network.remove_node(7)
    network.remove_node(8)

    logger.info(
        f"Nodes in Network (after): {list(node.id for node in network.get_nodes())}"
    )

    assert len(network.get_nodes()) == 6, logger.error(
        f"Expected 6 Nodes, got {len(network.get_nodes())}"
    )

    # Print network details
    logger.header(f"Network details:")
    logger.default(f"{network}")

    # Adding a dummy traffic source from AP 1 to STAs 2 and 3, and from AP 4 to STA 5
    logger.header(f"Adding Traffic Sources...")

    ap1 = network.get_node(1)
    ap1.add_traffic_flow(DummyTrafficSource(cfg_module, sparams_module, env, ap1, 1, 2))
    ap1.add_traffic_flow(DummyTrafficSource(cfg_module, sparams_module, env, ap1, 1, 3))

    ap4 = network.get_node(4)
    ap4.add_traffic_flow(DummyTrafficSource(cfg_module, sparams_module, env, ap4, 4, 5))

    # Test network plot
    logger.header(f"Plotting Network...")
    plotter = NetworkPlotter(cfg_module, sparams_module, env)
    plotter.plot_network(network)

    # Test clearing network
    logger.header(f"Clearing Network...")
    logger.info(
        f"Nodes in Network (before): {list(node.id for node in network.get_nodes())}"
    )

    network.clear()

    logger.info(
        f"Nodes in Network (after): {list(node.id for node in network.get_nodes())}"
    )

    assert len(network.get_nodes()) == 0, logger.error(
        f"Expected 0 nodes, got {len(network.get_nodes())}"
    )

    print(SIMULATION_TERMINATED_MSG)

    if len(plt.get_fignums()) > 0:
        input(PRESS_TO_EXIT_MSG)

    print(TEST_COMPLETED_MSG)
