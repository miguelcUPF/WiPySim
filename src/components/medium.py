from src.utils.event_logger import get_logger
from src.utils.data_units import PPDU
from src.utils.mcs_table import calculate_data_rate_bps, get_min_sensitivity
from src.components.network import Network, Node
from src.sim_params import (
    MPDU_ERROR_PROBABILITY,
    NUM_CHANNELS,
    SPATIAL_STREAMS,
    GUARD_INTERVAL_us,
    TX_POWER_dBm,
    TX_GAIN_dB,
    RX_GAIN_dB,
    PATH_LOSS_EXPONENT,
    ENABLE_SHADOWING,
    SHADOWING_STD_dB,
    FREQUENCY_MHz,
    ENABLE_RTS_CTS,
)

import simpy
import random
import math

VALID_20MHZ_BONDS = [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]
VALID_40MHZ_BONDS = [(1, 2), (3, 4), (5, 6), (7, 8)]
VALID_80MHZ_BONDS = [(1, 2, 3, 4), (5, 6, 7, 8)]
VALID_160MHZ_BOND = [(1, 2, 3, 4, 5, 6, 7, 8)]

VALID_BONDS = {
    20: VALID_20MHZ_BONDS,
    40: VALID_40MHZ_BONDS,
    80: VALID_80MHZ_BONDS,
    160: VALID_160MHZ_BOND,
}


class Channel20MHz:
    """Represents a single 20 MHz wireless channel."""

    def __init__(self, env: simpy.Environment, id: int):
        self.env = env
        self.id = id

        self.nodes = {}  # Nodes using the channel
        self.nodes_transmitting = {}  # Nodes currently transmitting

        self.nav_occupied = (
            False  # Flag to indicate if the channel is occupied due to NAV
        )
        self.nav_master_id = None
        self.nav_nodes_ids = set()

        self.collision_detected = False  # Flag to indicate if collision is detected

        self.name = "CHANNEL"
        self.logger = get_logger(self.name, env)

    def add_node(self, node: Node):
        self.logger.debug(f"Channel {self.id} -> Allocated to {node.type} {node.id}")
        self.nodes[node.id] = node

    def is_idle(self, node: Node):
        """Returns True if the channel is idle, False if busy."""
        if node.id in self.nav_nodes_ids:
            return True
        return len(self.nodes_transmitting) == 0 and not self.nav_occupied

    def occupy(self, node: Node):
        """Marks the channel as busy by a node and checks for collisions."""
        if node.id == self.nav_master_id:
            return

        self.logger.debug(f"Channel {self.id} -> Occupied by {node.type} {node.id}")

        if node.id in self.nav_nodes_ids:
            return

        self.nodes_transmitting[node.id] = node

        # If more than one node is transmitting, a collision occurs
        if len(self.nodes_transmitting) > 1:
            self.handle_collision()

    def start_nav(self, src_id: int, dst_id: int):
        self.logger.debug(
            f"Channel {self.id} -> Channel reservation started (NAV) by Node {src_id} for transmission to Node {dst_id}"
        )
        self.nav_occupied = True
        self.nav_master_id = src_id
        self.nav_nodes_ids = set([src_id, dst_id])

    def end_nav(self, dst_id: int):
        if dst_id == self.nav_master_id:
            self.logger.debug(
                f"Channel {self.id} -> Channel reservation ended (NAV) by Node {dst_id}"
            )
            self.nav_occupied = False
            self.nav_master_id = None
            self.nav_nodes_ids = set()

    def release(self, node: Node):
        """Removes a node from the channel."""
        self.logger.debug(f"Channel {self.id} -> Released by {node.type} {node.id}")
        self.nodes_transmitting.pop(node.id, None)
        if len(self.nodes_transmitting) == 0:
            self.collision_detected = False

    def handle_collision(self):
        """Handles collision."""
        self.logger.warning(f"Channel {self.id} -> Collision detected!")
        self.collision_detected = True


class MEDIUM:
    def __init__(self, env, network: Network):
        self.env = env

        self.network = network

        self.channels = {ch: Channel20MHz(env, ch) for ch in range(1, NUM_CHANNELS + 1)}

        self.name = "MEDIUM"
        self.logger = get_logger(self.name, env)

    def get_valid_channels(self) -> list:
        max_bond_size = NUM_CHANNELS * 20

        if max_bond_size not in VALID_BONDS:
            valid_values = {1, 2, 4, 8}
            self.logger.critical(
                f"Invalid NUM_CHANNELS: {NUM_CHANNELS}. It must be one of {valid_values}."
            )
            return None

        valid_channel_bonds = []
        for bw, bond_list in VALID_BONDS.items():
            if bw <= max_bond_size:
                valid_channel_bonds.extend(bond_list)

        valid_channels = list(tuple(channel) for channel in valid_channel_bonds)
        return valid_channels

    def are_channels_idle(self, node: Node, channels_ids: list[int]):
        """Checks if all selected channels are idle."""
        return all(self.channels[ch_id].is_idle(node) for ch_id in channels_ids)

    def any_collision_detected(self, channels_ids: list[int]):
        """Checks if any of the selected channels has a collision."""
        return any(self.channels[ch_id].collision_detected for ch_id in channels_ids)

    def occupy_channels(self, node, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].occupy(node)

    def release_channels(self, node, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].release(node)

    def start_nav(self, src_id: int, dst_id: int, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].start_nav(src_id, dst_id)

    def end_nav(self, src_id, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].end_nav(src_id)

    def broadcast_channel_info(self, src_id, dst_id, channels_ids, primary_channel):
        src_node = self.network.get_node(src_id)
        dst_node = self.network.get_node(dst_id)

        self.logger.header(
            f"Broadcasting channel info from {src_node.type} {src_id} to {dst_node.type} {dst_id}..."
        )

        dst_node.phy_layer.receive_channel_info(channels_ids, primary_channel)

    def broadcast_mcs_info(self, src_id, dst_id, mcs_index):
        src_node = self.network.get_node(src_id)
        dst_node = self.network.get_node(dst_id)

        self.logger.header(
            f"Broadcasting MCS info from {src_node.type} {src_id} to {dst_node.type} {dst_id}..."
        )

        dst_node.phy_layer.receive_mcs_info(mcs_index)

    def transmit(self, ppdu: PPDU, channels_ids: list[int], mcs_index: int):
        def _calculate_tx_time(mcs_index, channel_width, size_bytes):
            """Computes transmission time based on bandwidth and MCS."""

            data_rate_bps = calculate_data_rate_bps(
                mcs_index, channel_width, SPATIAL_STREAMS, GUARD_INTERVAL_us
            )

            tx_duration_us = size_bytes * 8 / data_rate_bps * 1e6

            return tx_duration_us

        self.logger.header(
            f"Transmitting {ppdu.type} from {ppdu.src_id} to {ppdu.dst_id} over channel(s) {', '.join(map(str, channels_ids))}..."
        )

        tx_duration_us = _calculate_tx_time(
            mcs_index, len(channels_ids) * 20, ppdu.size_bytes
        )

        collision_detected = False
        start_time = self.env.now
        while self.env.now - start_time < tx_duration_us:
            if self.any_collision_detected(channels_ids):
                collision_detected = False
            yield self.env.timeout(1)

        if not collision_detected:
            self.receive(ppdu, channels_ids, mcs_index)
        else:
            self.logger.warning(
                f"Collision detected while transmitting {ppdu.type} from {ppdu.src_id} to {ppdu.dst_id} over channel(s) {', '.join(map(str, channels_ids))}"
            )
        self.release_channels(self.network.get_node(ppdu.src_id), channels_ids)

    def receive(self, ppdu: PPDU, channels_ids: list[int], mcs_index: int):
        def _calculate_path_loss(distance_m: float):
            path_loss_1m_dB = (
                20 * math.log10(1) + 20 * math.log10(FREQUENCY_MHz) - 147.55
            )  # Assuming free space path loss

            path_loss = path_loss_1m_dB + 10 * PATH_LOSS_EXPONENT * math.log10(
                distance_m / 1
            )

            if ENABLE_SHADOWING:
                path_loss += random.gauss(0, SHADOWING_STD_dB)

            return path_loss

        distance_m = self.network.get_distance_between_nodes(ppdu.src_id, ppdu.dst_id)

        rssi_dbm = (
            TX_POWER_dBm + TX_GAIN_dB + RX_GAIN_dB - _calculate_path_loss(distance_m)
        )

        min_sensitivity_dbm = get_min_sensitivity(mcs_index, len(channels_ids) * 20)
        if rssi_dbm < min_sensitivity_dbm:
            self.logger.warning(
                f"Unreliable PPDU reception (from {ppdu.src_id} to {ppdu.dst_id}): RSSI ({rssi_dbm:.2f} dBm) below min. sensitivity threshold ({min_sensitivity_dbm:.2f} dBm)"
            )
            return

        if ppdu.data_unit.type == "AMPDU":
            for mpdu in ppdu.data_unit.mpdus:
                mpdu.is_corrupted = (
                    True if MPDU_ERROR_PROBABILITY > random.random() else False
                )

        if ENABLE_RTS_CTS:
            if ppdu.data_unit.type == "RTS":
                self.start_nav(ppdu.src_id, ppdu.dst_id, channels_ids)

        node_dst = self.network.get_node(ppdu.dst_id)
        node_dst.phy_layer.receive(ppdu)
