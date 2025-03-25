from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.utils.event_logger import get_logger
from src.components.network import Network, Node
from src.utils.data_units import PPDU
from src.utils.statistics import ChannelStats, MediumStats
from src.utils.mcs_table import calculate_data_rate_bps, get_min_sensitivity

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

    def __init__(self, cfg: cfg, sparams: sparams, env: simpy.Environment, id: int):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.id = id

        self.primary_channel_nodes: dict[int, Node] = (
            {}
        )  # Nodes using the channel as primary
        self.nodes: dict[int, Node] = {}  # Nodes using the channel
        self.nodes_transmitting: dict[int, Node] = {}  # Nodes currently transmitting

        self.nav_occupied = (
            False  # Flag to indicate if the channel is occupied due to NAV
        )
        self.nav_master_id = None
        self.nav_nodes_ids = set()

        self.collision_detected = False  # Flag to indicate if collision is detected
        self.last_collision_time = 0

        self.idle_start_time = self.env.now
        self.busy_start_time = None

        self.stats = ChannelStats()

        self.name = "CHANNEL"
        self.logger = get_logger(self.name, cfg, sparams, env)

    def assign(self, node: Node):
        self.nodes[node.id] = node

    def release(self, node: Node):
        self.nodes.pop(node.id, None)

    def assign_as_primary_channel(self, node: Node):
        self.primary_channel_nodes[node.id] = node

    def release_as_primary_channel(self, node: Node):
        self.primary_channel_nodes.pop(node.id, None)

    def is_idle(self, node: Node = None):
        """Returns True if the channel is idle, False if busy."""
        if node is not None and node.id in self.nav_nodes_ids:
            return True
        return self.idle_start_time is not None

    def occupy(self, node: Node):
        """Marks the channel as busy by a node and checks for collisions."""
        self.logger.debug(f"Channel {self.id} -> Occupied by {node.type} {node.id}")

        self.idle_start_time = None

        if self.busy_start_time is None:
            self.busy_start_time = self.env.now

        for node_id, p_node in self.primary_channel_nodes.items():
            if node_id not in self.nav_nodes_ids:
                p_node.phy_layer.set_primary_busy()

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
            self.idle_start_time = self.env.now
            for node_id, p_node in self.primary_channel_nodes.items():
                p_node.phy_layer.set_primary_idle()

    def unoccupy(self, node: Node):
        """Removes a node from the channel."""
        if node.id not in self.nodes_transmitting:
            return
        self.logger.debug(f"Channel {self.id} -> Unoccupied by {node.type} {node.id}")
        self.nodes_transmitting.pop(node.id, None)
        if len(self.nodes_transmitting) == 0:
            self.stats.airtime_us += self.env.now - self.busy_start_time
            self.busy_start_time = None
            self.collision_detected = False
            if not self.nav_occupied:
                self.idle_start_time = self.env.now
                for node_id, p_node in self.primary_channel_nodes.items():
                    p_node.phy_layer.set_primary_idle()

    def handle_collision(self):
        """Handles collision."""
        self.collision_detected = True
        self.last_collision_time = self.env.now

    def rts_collision_detected(self):
        for node_id, node in self.nodes.items():
            node.phy_layer.rts_collision_detected()

    def ampdu_collision_detected(self):
        for node_id, node in self.nodes.items():
            node.phy_layer.ampdu_collision_detected()

    def successful_transmission_detected(self):
        for node_id, node in self.nodes.items():
            node.phy_layer.successful_transmission_detected()


class MEDIUM:
    def __init__(
        self, cfg: cfg, sparams: sparams, env: simpy.Environment, network: Network
    ):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.network = network

        self.channels = {
            ch: Channel20MHz(cfg, sparams, env, ch)
            for ch in range(1, self.sparams.NUM_CHANNELS + 1)
        }

        self.stats = MediumStats()

        self.busy_start_time = None

        self.name = "MEDIUM"
        self.logger = get_logger(self.name, cfg, sparams, env)

    def get_valid_channels(self) -> list:
        available_channels = set(self.channels.keys())  # Extract available channel IDs

        valid_channel_bonds = []
        for bw, bond_list in VALID_BONDS.items():
            # Filter bonds that are fully contained in the available channels
            valid_channel_bonds.extend(
                [bond for bond in bond_list if set(bond).issubset(available_channels)]
            )

        return valid_channel_bonds

    def are_all_channels_idle(self):
        return all(self.channels[ch_id].is_idle() for ch_id in self.channels)

    def are_channels_idle(self, node: Node, channels_ids: list[int]):
        """Checks if all selected channels are idle."""
        return all(self.channels[ch_id].is_idle(node) for ch_id in channels_ids)

    def any_collision_detected(self, channels_ids: list[int], start_time_us: int):
        """Checks if any of the selected channels has a collision."""
        return any(
            self.channels[ch_id].last_collision_time >= start_time_us
            for ch_id in channels_ids
        )

    def assign_channels(self, node: Node, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].assign(node)

    def release_channels(self, node: Node, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].release(node)

    def assign_as_primary_channel(self, node: Node, channel_id: int):
        self.channels[channel_id].assign_as_primary_channel(node)

    def release_as_primary_channel(self, node: Node, channel_id: int):
        self.channels[channel_id].release_as_primary_channel(node)

    def occupy_channels(self, node, channels_ids: list[int]):
        if self.busy_start_time is None:
            self.busy_start_time = self.env.now

        for ch_id in channels_ids:
            self.channels[ch_id].occupy(node)

    def unoccupy_channels(self, node, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].unoccupy(node)

        if self.are_all_channels_idle():
            self.stats.airtime_us += self.env.now - self.busy_start_time
            self.busy_start_time = None

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

    def rts_collision_detected(self, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].rts_collision_detected()

    def ampdu_collision_detected(self, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].ampdu_collision_detected()

    def successful_transmission_detected(self, channels_ids: list[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].successful_transmission_detected()

    def transmit(self, ppdu: PPDU, channels_ids: list[int], mcs_index: int):
        def _calculate_tx_time(
            mcs_index, channel_width, size_bytes, is_mgmt_ctrl_frame=False
        ):
            """Computes transmission time based on bandwidth and MCS."""

            data_rate_bps = calculate_data_rate_bps(
                mcs_index,
                channel_width,
                self.sparams.SPATIAL_STREAMS if not is_mgmt_ctrl_frame else 1,
                self.sparams.GUARD_INTERVAL_us,
            )

            tx_duration_us = size_bytes * 8 / data_rate_bps * 1e6

            return round(tx_duration_us)

        self.logger.header(
            f"Transmitting {ppdu.data_unit.type} from node {ppdu.src_id} to node {ppdu.dst_id} over channel(s) {', '.join(map(str, channels_ids))}..."
        )

        self.occupy_channels(self.network.get_node(ppdu.src_id), channels_ids)

        tx_duration_us = _calculate_tx_time(
            mcs_index,
            len(channels_ids) * 20,
            ppdu.size_bytes,
            ppdu.data_unit.is_mgmt_ctrl_frame,
        )

        tx_start = self.env.now

        yield self.env.timeout(tx_duration_us)

        collision_detected = self.any_collision_detected(channels_ids, tx_start)

        self.stats.ppdus_tx += 1

        if not collision_detected:
            self.successful_transmission_detected(channels_ids)
            self.receive(ppdu, channels_ids, mcs_index)
            self.stats.ppdus_success += 1
        else:
            self.logger.warning(
                f"{ppdu.type} ({ppdu.data_unit.type}) from {ppdu.src_id} to {ppdu.dst_id} over channel(s) {', '.join(map(str, channels_ids))} collided!"
            )

            self.stats.ppdus_fail += 1

            if ppdu.data_unit.type == "RTS":
                self.rts_collision_detected(channels_ids)
            else:
                self.ampdu_collision_detected(channels_ids)

        self.unoccupy_channels(self.network.get_node(ppdu.src_id), channels_ids)

    def receive(self, ppdu: PPDU, channels_ids: list[int], mcs_index: int):
        def _calculate_path_loss(distance_m: float):
            path_loss_1m_dB = (
                20 * math.log10(1)
                + 20 * math.log10(self.sparams.FREQUENCY_MHz)
                - 147.55
            )  # Assuming free space path loss

            path_loss = (
                path_loss_1m_dB
                + 10
                * self.sparams.PATH_LOSS_EXPONENT
                * math.log10(max(distance_m, 0.1) / 1)
            )

            if self.sparams.ENABLE_SHADOWING:
                path_loss += random.gauss(0, self.sparams.SHADOWING_STD_dB)

            return path_loss

        distance_m = self.network.get_distance_between_nodes(ppdu.src_id, ppdu.dst_id)

        rssi_dbm = (
            self.sparams.TX_POWER_dBm
            + self.sparams.TX_GAIN_dB
            + self.sparams.RX_GAIN_dB
            - _calculate_path_loss(distance_m)
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
                    True
                    if self.sparams.MPDU_ERROR_PROBABILITY > random.random()
                    else False
                )

        if self.sparams.ENABLE_RTS_CTS:
            if ppdu.data_unit.type == "RTS":
                self.start_nav(ppdu.src_id, ppdu.dst_id, channels_ids)

        node_dst = self.network.get_node(ppdu.dst_id)
        node_dst.phy_layer.receive(ppdu)
