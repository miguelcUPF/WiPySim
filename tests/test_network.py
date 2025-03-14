
import simpy

import importlib
import matplotlib.pyplot as plt
import src.sim_config as sim_config
import src.utils.event_logger
import src.utils.plotters

from src.utils.event_logger import get_logger, update_logger_env
from src.components.network import Network, Node
from src.utils.plotters import NetworkPlotter

sim_config.ENABLE_CONSOLE_LOGGING = True
sim_config.USE_COLORS_IN_EVENT_LOGS = True

sim_config.EXCLUDED_CONSOLE_LEVELS = []
sim_config.EXCLUDED_CONSOLE_MODULES = []

sim_config.ENABLE_EVENT_RECORDING = False

sim_config.ENABLE_FIGS_DISPLAY = True
sim_config.ENABLE_FIGS_SAVING = True
sim_config.FIGS_SAVE_PATH = "figs/tests"
sim_config.ENABLE_FIGS_OVERWRITE = True

importlib.reload(src.utils.event_logger)
importlib.reload(src.utils.plotters)

logger = get_logger("MAIN")

if __name__ == "__main__":
    print("\033[93m\n" + "="*24 + "  TEST STARTED  " + "="*24 + "\033[0m")

    env = simpy.Environment()
    network = Network(env)
    update_logger_env(env)

    logger.header(f"Starting Network Test...")

    # Create nodes
    nodes = [
        Node(env, 1, (0, 0, 0)),
        Node(env, 2, (3, 4, 0)),
        Node(env, 3, (6, 8, 2)),
        Node(env, 4, (5, 5, 5)),
        Node(env, 5, (1, 2, 3)),
        Node(env, 6, (7, 6, 4)),
        Node(env, 7, (9, 1, 8)),
        Node(env, 8, (2, 9, 7)),
    ]

    logger.header(f"Creating Nodes...")
    logger.info(f"Nodes in Network (before):")
    logger.default(f"{network.get_nodes()}")
    logger.debug(f"Adding {len(nodes)} nodes to Network...")
    network.add_nodes(nodes)
    logger.info(f"Nodes in Network (after):")
    logger.default(f"{network.get_nodes()}")
    assert len(network.get_nodes(
    )) == 8, logger.error(f"Expected 8 nodes, got {len(network.get_nodes())}")

    # Create bidirectional links
    bidirectional_links = [
        (nodes[0], nodes[1], 1),  # 1 <-> 2
        (nodes[2], nodes[3], 2),  # 3 <-> 4
        (nodes[4], nodes[5], 1),  # 5 <-> 6
        (nodes[6], nodes[7], 3)   # 7 <-> 8
    ]

    logger.header(f"Creating Bidirectional Links...")
    logger.info(f"Links in Network (before):")
    logger.default(f"{network.get_links()}")
    logger.debug(
        f"Adding {len(bidirectional_links)} bidirectional links to Network...")
    network.add_links(bidirectional_links)
    logger.info(f"Links in Network (after):")
    logger.default(f"{network.get_links()}")
    assert len(network.get_links(
    )) == 4, logger.error(f"Expected 4 links, got {len(network.get_links())}")

    # Ensure no node has depth greater than 2
    for node in nodes:
        assert network.get_node_degree(node.id
                                       ) <= 2, logger.error(f"Node {node.id} has depth greater than 2")

    # Test updating nodes positions
    logger.header(f"Updating Node Positions...")
    logger.info(f"Node 1 and 2 positions (before):")
    logger.default(f"{network.get_nodes_by_id([1, 2])}")
    logger.debug(f"Updating node 1 and node 2 positions...")
    network.update_nodes_positions(nodes[:2], [(1, 0, 1), (3, 3, 3)])
    logger.info(f"Node 1 and 2 positions (after):")
    logger.default(f"{network.get_nodes_by_id([1, 2])}")
    assert network.get_node_by_id(1).position == (1, 0, 1), logger.error(
        f"Expected node 1 position to be (1, 0, 1), got {network.get_node_by_id(1).position}")
    assert network.get_node_by_id(2).position == (3, 3, 3), logger.error(
        f"Expected node 2 position to be (3, 3, 3), got {network.get_node_by_id(2).position}")

    # Test removing links
    logger.header(f"Removing Link...")
    logger.info(f"Links in Network (before):")
    logger.default(f"{network.get_links()}")
    logger.debug(f"Removing bidirectional link between nodes 7 and 8...")
    network.remove_link(
        (network.get_node_by_id(8), network.get_node_by_id(7)))
    logger.info(f"Links in Network (after):")
    logger.default(f"{network.get_links()}")
    assert len(network.get_links(
    )) == 3, logger.error(f"Expected 3 links, got {len(network.get_links())}")

    # Test removing nodes
    logger.header(f"Removing Node...")
    logger.info(f"Nodes in Network (before):")
    logger.default(f"{network.get_nodes()}")
    logger.debug(f"Removing nodes 7 and 8...")
    network.remove_nodes(
        [network.get_node_by_id(7), network.get_node_by_id(8)])
    logger.info(f"Nodes in Network (after):")
    logger.default(f"{network.get_nodes()}")
    assert len(network.get_nodes(
    )) == 6, logger.error(f"Expected 6 nodes, got {len(network.get_nodes())}")

    # Test updating link channel
    logger.header(f"Updating Link Channel...")
    logger.info(f"Links using channel 1 (before):")
    logger.default(f"{network.get_links_by_channel(1)}")
    logger.info(f"Links using channel 3 (before):")
    logger.default(f"{network.get_links_by_channel(3)}")
    logger.debug(
        f"Updating edge between node 1 and 2 from channel 1 to channel 3...")
    network.update_links_channels(
        [(network.get_node_by_id(1), network.get_node_by_id(2))], [3])
    logger.info(f"Links using channel 1 (after):")
    logger.default(f"{network.get_links_by_channel(1)}")
    logger.info(f"Links using channel 3 (after):")
    logger.default(f"{network.get_links_by_channel(3)}")
    assert len(network.get_links_by_channel(1)) == 1, logger.error(
        f"Expected 1 link using channel 1, got {len(network.get_links_by_channel(1))}")
    assert len(network.get_links_by_channel(3)) == 1, logger.error(
        f"Expected 1 link using channel 3, got {len(network.get_links_by_channel(3))}")

    # Test is bidirectional link
    logger.header(f"Is there a bidirectional link between nodes 1 and 2?")
    logger.default(
        f"{network.is_bidirectional_link((network.get_node_by_id(1), network.get_node_by_id(2)))}")
    assert network.is_bidirectional_link((network.get_node_by_id(1), network.get_node_by_id(
        2))), logger.error(f"Expected True, got False")

    # Print network details
    logger.header(f"Network details:")
    logger.default(f"{network}")

    # Plot the network
    logger.header(f"Plotting Network...")
    plotter = NetworkPlotter()
    plotter.plot_network(network)

    # Test clearing network
    logger.header(f"Clearing Network...")
    logger.info(f"Nodes in Network (before):")
    logger.default(f"{network.get_nodes()}")
    logger.info(f"Links in Network (before):")
    logger.default(f"{network.get_links()}")
    logger.debug(f"Clearing network nodes and links...")
    network.clear()
    logger.info(f"Nodes in Network (after):")
    logger.default(f"{network.get_nodes()}")
    logger.info(f"Links in Network (after):")
    logger.default(f"{network.get_links()}")
    assert len(network.get_nodes(
    )) == 0, logger.error(f"Expected 0 nodes, got {len(network.get_nodes())}")
    assert len(network.get_links(
    )) == 0, logger.error(f"Expected 0 links, got {len(network.get_links())}")

    print("\033[93m\n" + "="*23 + "  TEST COMPLETED  " + "="*23)
    if len(plt.get_fignums()) > 0:
        input(f"{'='*10}  Press Enter to exit and close all plots   {'='*10}")
    print("="*20 + "  EXECUTION TERMINATED  " + "="*20 + "\033[0m")
