import networkx as nx
import math
import simpy

from src.components.mac import MAC
from src.components.phy import PHY
from src.components.app import APP
from src.components.medium import Medium

from src.utils.event_logger import get_logger


class Node:
    def __init__(self, env: simpy.Environment, id: int, position: tuple[float, float, float]):
        """Initializes an individual network node object."""
        self.id = id
        self.position = position  # x, y, z
        self.app_layer = APP(env, id)
        self.mac_layer = MAC(env, id)
        self.phy_layer = PHY(env, id)

        self.app_layer.mac_layer = self.mac_layer

        self.mac_layer.app_layer = self.app_layer
        self.mac_layer.phy_layer = self.phy_layer

        self.phy_layer.mac_layer = self.mac_layer

    def set_position(self, position: tuple[float, float, float]):
        """Updates the node's position."""
        self.position = position

    def __repr__(self):
        return f"Node({self.id}, pos={self.position})"


class Network:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.name = "NETWORK"
        self.graph = nx.Graph()
        self.nodes = {}
        self.medium = Medium(env)

        self.logger = get_logger(self.name)

    @staticmethod
    def _calculate_distance(pos1: tuple[float, float, float], pos2: tuple[float, float, float]) -> float:
        """Returns the Euclidean distance between two 3D points."""
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def add_node(self, node: Node):
        """Adds a node to the network and stores its position."""
        self.logger.debug(
            f"Adding node {node.id} to the network at position {node.position}")

        x, y, z = node.position
        self.graph.add_node(node.id, pos=(x, y, z))
        self.nodes[node.id] = node

    def add_nodes(self, nodes: list[Node]):
        """Adds multiple nodes to the network."""
        for node in nodes:
            self.add_node(node)

    def get_nodes(self) -> list[Node]:
        """Returns a list of all nodes in the network."""
        return [self.get_node_by_id(id) for id in self.graph.nodes]

    def get_node_by_id(self, id: int) -> Node:
        return self.nodes[id]

    def get_nodes_by_id(self, ids: list[int]) -> list[Node]:
        return [self.get_node_by_id(id) for id in ids]

    def get_node_positions(self) -> list[tuple[float, float, float]]:
        """Returns the positions of all nodes in the network."""
        return [self.graph.nodes[id]["pos"] for id in self.graph.nodes]

    def get_node_degree(self, id: int) -> int:
        return self.graph.degree(id)

    def get_nodes_degrees(self) -> list[int]:
        return [self.graph.degree(id) for id in self.graph.nodes]

    def remove_node(self, node: Node):
        """Removes a node from the network."""
        self.logger.debug(f"Removing node {node.id} from the network")

        id = node.id
        self.graph.remove_node(id)
        del self.nodes[id]

    def remove_nodes(self, nodes: list[Node]):
        for node in nodes:
            self.remove_node(node)

    def update_node_position(self, node: Node, new_position: tuple[float, float, float]):
        """Updates the position of a node in the network."""
        self.logger.debug(
            f"Updating node {node.id} position to {new_position}")

        node.set_position(new_position)
        id = node.id
        self.graph.nodes[id]["pos"] = new_position

    def update_nodes_positions(self, nodes: list[Node], new_positions: list[tuple[float, float, float]]):
        for node, new_position in zip(nodes, new_positions):
            self.update_node_position(node, new_position)

    def add_link(self, link: tuple[Node, Node, int]):
        """Creates a link between two nodes and stores the distance."""
        node1, node2, channel_id = link

        self.logger.debug(
            f"Adding link between node {node1.id} and node {node2.id}")

        distance = round(self._calculate_distance(node1.position, node2.position), 4)
        self.graph.add_edge(node1.id, node2.id,
                            distance=distance, channel=channel_id)

    def add_links(self, links: list[tuple[Node, Node, int]]):
        for link in links:
            self.add_link(link)

    def remove_link(self, link: tuple[Node, Node]):
        """Removes a link between two nodes."""
        node1, node2 = link

        self.logger.debug(
            f"Removing link between node {node1.id} and node {node2.id}")

        if self.graph.has_edge(node1.id, node2.id):
            self.graph.remove_edge(node1.id, node2.id)

    def remove_links(self, links: list[tuple[Node, Node]]):
        for link in links:
            self.remove_link(link)

    def update_link_channel(self, link: tuple[Node, Node], channel: int = None) -> None:
        """Updates the channel of a link."""
        node1, node2 = link

        self.logger.debug(
            f"Updating channel of link between node {node1.id} to node {node2.id}")

        if self.graph.has_edge(node1.id, node2.id):
            self.graph[node1.id][node2.id]["channel"] = channel

    def update_links_channels(self, links: list[tuple[Node, Node]], channels: list[int]):
        for link, channel in zip(links, channels):
            self.update_link_channel(link, channel)

    def get_distance_between_nodes(self, node1: Node, node2: Node) -> float:
        """Returns the distance between two connected nodes."""
        if self.graph.has_edge(node1.id, node2.id):
            return self.graph[node1.id][node2.id]["distance"]

    def get_node_links(self, id: int) -> list[tuple[int, int]]:
        """Returns a list of links for a node."""
        return [(id, neighbor) for neighbor in self.graph[id]]

    def get_links(self) -> list[tuple[int, int, dict]]:
        """Returns a list of links in the network."""
        return list(self.graph.edges(data=True))

    def get_links_by_channel(self, channel: int) -> list[tuple[int, int]]:
        """Returns a list of links with the specified channel."""
        return [(u, v) for u, v, data in self.graph.edges(data=True) if data["channel"] == channel]

    def get_nodes_by_channel(self, channel: int) -> list[int]:
        """Returns a list of nodes that use the specified channel."""
        nodes = set()
        for u, v in self.get_links_by_channel(channel):
            nodes.add(self.get_node_by_id(u))
            nodes.add(self.get_node_by_id(v))
        return list(set(nodes))

    def is_bidirectional_link(self, nodes: tuple[Node, Node]) -> bool:
        """Returns True if the link is bidirectional, False otherwise."""
        return self.graph.has_edge(nodes[0].id, nodes[1].id) and self.graph.has_edge(nodes[1].id, nodes[0].id)

    def clear(self):
        self.graph.clear()
        self.nodes.clear()

    def __repr__(self):
        return f"Network(nodes={list(self.graph.nodes)}, edges={list(self.graph.edges)})"
