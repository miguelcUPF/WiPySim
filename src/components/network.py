from src.user_config import UserConfig as cfg_module
from src.sim_params import SimParams as sparams_module

from src.utils.event_logger import get_logger
from src.utils.statistics import TransmissionStats, ReceptionStats, NetworkStats

from typing import cast

import networkx as nx
import math
import simpy


class Node:
    def __init__(
        self,
        cfg: cfg_module,
        sparams: sparams_module,
        env: simpy.Environment,
        id: int,
        position: tuple[float, float, float],
        channels: set,
        sensing_channels: set,
        type: str,
        medium,
        network,
        rl_driven: bool = False
    ):
        """Initializes an individual network node object."""
        from src.components.medium import Medium

        from src.components.mac import MAC
        from src.components.phy import PHY
        from src.components.app import APP

        self.id = id
        self.position = position  # x, y, z
        self.type = type

        self.bss_id = None

        self.network: Network = network

        self.medium: Medium = medium

        self.app_layer = APP(cfg, sparams, env, self)
        self.mac_layer = MAC(cfg, sparams, env, self, rl_driven)
        self.phy_layer = PHY(cfg, sparams, env, self, channels, sensing_channels)

        self.tx_stats = TransmissionStats()
        self.rx_stats = ReceptionStats()

        self.traffic_flows = []

        self.name = "NODE"
        self.logger = get_logger(self.name, cfg, sparams, env)

    def add_traffic_flow(self, traffic_flow):
        self.traffic_flows.append(traffic_flow)
        self.logger.debug(
            f"{self.type} {self.id} -> Added traffic source: {traffic_flow.__class__.__name__}"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id}, pos={self.position})"


class STA(Node):
    def __init__(
        self,
        cfg: cfg_module,
        sparams: sparams_module,
        env: simpy.Environment,
        id: int,
        position: tuple[float, float, float],
        channels: set,
        sensing_channels: set,
        bss_id: int,
        ap,
        medium,
        network,
    ):
        """Station (STA) node, associated with an AP and BSS."""
        self.bss_id = bss_id
        self.ap: AP = ap

        super().__init__(
            cfg,
            sparams,
            env,
            id,
            position,
            channels,
            sensing_channels,
            "STA",
            medium,
            network,
        )

        self.ap.add_sta(self)  # Automatically associate with the AP

    def __repr__(self):
        return (
            f"STA({self.id}, pos={self.position}, BSS={self.bss_id}, AP={self.ap.id})"
        )


class AP(Node):
    def __init__(
        self,
        cfg: cfg_module,
        sparams: sparams_module,
        env: simpy.Environment,
        id: int,
        position: tuple[float, float, float],
        channels: set,
        sensing_channels: set,
        bss_id: int,
        medium,
        network,
        rl_driven: bool = False
    ):
        """Access Point (AP) node."""
        self.bss_id = bss_id
        self.associated_stas = []

        super().__init__(
            cfg,
            sparams,
            env,
            id,
            position,
            channels,
            sensing_channels,
            "AP",
            medium,
            network,
            rl_driven=rl_driven
        )

    def add_sta(self, sta: STA):
        """Associates a STA with this AP."""
        self.associated_stas.append(sta)

    def get_stas(self) -> list[STA]:
        return self.associated_stas

    def __repr__(self):
        return f"AP({self.id}, pos={self.position}, BSS={self.bss_id}, STAs={[sta.id for sta in self.associated_stas]})"


class Network:
    def __init__(self, cfg: cfg_module, sparams: sparams_module, env: simpy.Environment):
        from src.components.medium import Medium

        self.env = env

        self.cfg = cfg
        self.sparams = sparams

        self.graph = nx.Graph()

        self.nodes = {}

        self.medium = Medium(cfg, sparams, env, self)

        self.stats = NetworkStats(cfg, sparams, self)

        self.name = "NETWORK"
        self.logger = get_logger(self.name, cfg, sparams, env)

    @staticmethod
    def _calculate_distance(
        position_1: tuple[float, float, float], position_2: tuple[float, float, float]
    ) -> float:
        """Returns the Euclidean distance between two 3D points."""
        x1, y1, z1 = position_1
        x2, y2, z2 = position_2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def add_ap(
        self,
        ap_id: int,
        position: tuple[float, float, float],
        bss_id: int,
        channels: set,
        sensing_channels: set,
        rl_driven: bool = False,
    ) -> AP:
        """Adds an Access Point (AP) to the network."""
        if ap_id in self.nodes:
            existing_node = self.nodes[ap_id]
            if isinstance(existing_node, AP):
                self.logger.warning(
                    f"AP {ap_id} already exists in the network... Returning existing AP."
                )
                return existing_node
            else:
                self.logger.error(
                    f"Node {ap_id} already exists as a {existing_node.__class__.__name__}, cannot add as AP!"
                )
                return None

        self.logger.debug(
            f"Adding AP {ap_id} to BSS {bss_id} at position {position}. Channels: {channels}. Sensing Channels: {sensing_channels}"
        )
        ap = AP(
            self.cfg,
            self.sparams,
            self.env,
            ap_id,
            position,
            channels,
            sensing_channels,
            bss_id,
            self.medium,
            self,
            rl_driven=rl_driven
        )
        self.nodes[ap_id] = ap
        self.graph.add_node(ap_id, pos=position, type="AP", bss_id=bss_id)
        return ap

    def add_sta(
        self, sta_id: int, position: tuple[float, float, float], bss_id: int, ap: AP
    ) -> STA:
        """Adds a Station (STA) to the network and associates it with an AP."""
        if sta_id in self.nodes:
            existing_node = self.nodes[sta_id]
            if isinstance(existing_node, STA):
                self.logger.warning(
                    f"STA {sta_id} already exists in the network... Returning existing STA."
                )
                return existing_node
            else:
                self.logger.error(
                    f"Node {sta_id} already exists as a {existing_node.__class__.__name__}, cannot add as STA!"
                )
                return None

        self.logger.debug(
            f"Adding STA {sta_id} to BSS {bss_id} at position {position}, connected to AP {ap.id}"
        )
        sta = STA(
            self.cfg,
            self.sparams,
            self.env,
            sta_id,
            position,
            ap.phy_layer.channels_ids,
            ap.phy_layer.sensing_channels_ids,
            bss_id,
            ap,
            self.medium,
            self,
        )
        self.nodes[sta_id] = sta
        self.graph.add_node(sta_id, pos=position, type="STA", bss_id=bss_id)
        self.graph.add_edge(ap.id, sta_id)
        return sta

    def get_aps(self) -> list[AP]:
        return [node for node in self.nodes.values() if isinstance(node, AP)]

    def get_stas(self) -> list[STA]:
        return [node for node in self.nodes.values() if isinstance(node, STA)]

    def get_node(self, node_id: int) -> Node:
        if node_id not in self.nodes:
            self.logger.error(f"Node {node_id} not found")
            return None
        return self.nodes[node_id]

    def get_nodes(self) -> list[Node]:
        return list(self.nodes.values())

    def get_node_pos(self, node_id: int) -> tuple[float, float, float]:
        return self.graph.nodes[node_id]["pos"]

    def remove_node(self, node_id: int):
        if node_id not in self.nodes:
            self.logger.error(f"Node {node_id} not found... Cannot remove.")
            return

        node = self.nodes[node_id]

        if isinstance(node, AP):
            # If the node is an AP, remove all associated STAs
            self.logger.debug(f"Removing AP {node_id} and all associated STAs")
            for sta in node.get_stas():
                self.remove_node(sta.id)
        elif isinstance(node, STA):
            self.logger.debug(f"Removing STA {node_id}")

        self.graph.remove_node(node_id)
        del self.nodes[node_id]

    def update_node_position(
        self, node_id: int, new_position: tuple[float, float, float]
    ):
        node = self.get_node(node_id)
        if node:
            self.logger.debug(
                f"Updating {node.type} {node_id} position to {new_position}"
            )
            node.position = new_position
            self.graph.nodes[node_id]["pos"] = new_position

            if isinstance(node, AP):
                ap = cast(AP, node)
                ap.phy_layer.select_ap_mcs_indexs()
            elif isinstance(node, STA):
                sta = cast(STA, node)
                sta.ap.phy_layer.select_mcs_index(sta.id)

        else:
            self.logger.error(f"Node {node_id} not found. Cannot update position.")

    def get_distance_between_nodes(
        self, node1_id: int, node2_id: int, digits=0
    ) -> float:
        node1 = self.get_node(node1_id)
        node2 = self.get_node(node2_id)

        if not node1 or not node2:
            self.logger.error(f"One or both nodes not found: {node1_id}, {node2_id}")
            return -1

        return round(self._calculate_distance(node1.position, node2.position), digits)

    def clear(self):
        self.graph.clear()
        self.nodes.clear()

    def __repr__(self):
        network_repr = f"Network("

        for node in self.nodes.values():
            if isinstance(node, AP):
                associated_stas = [sta.id for sta in node.get_stas()]
                network_repr += f"AP {node.id} with STAs: {associated_stas}, "

        network_repr += ")"
        return network_repr
