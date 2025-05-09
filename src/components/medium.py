from src.sim_params import SimParams as sparams_module
from src.user_config import UserConfig as cfg_module

from src.utils.event_logger import get_logger
from src.components.network import Network, Node
from src.components.mac import MACState
from src.utils.data_units import PPDU
from src.utils.statistics import ChannelStats, MediumStats
from src.utils.mcs_table import get_min_sensitivity_dBm
from src.utils.transmission import get_tx_duration_us, get_rssi_dbm

import simpy
import random

VALID_20MHZ_BONDS = [{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}]
VALID_40MHZ_BONDS = [{1, 2}, {3, 4}, {5, 6}, {7, 8}]
VALID_80MHZ_BONDS = [{1, 2, 3, 4}, {5, 6, 7, 8}]
VALID_160MHZ_BOND = [{1, 2, 3, 4, 5, 6, 7, 8}]

VALID_BONDS = {
    20: VALID_20MHZ_BONDS,
    40: VALID_40MHZ_BONDS,
    80: VALID_80MHZ_BONDS,
    160: VALID_160MHZ_BOND,
}


class UtilizationTracker:
    def __init__(self, ch_id: int, window_duration_us: float):
        self.ch_id = ch_id
        self.window_duration_us = window_duration_us
        self.events = []  # list of (start_time, duration, node_id)

    def record_busy_start(self, time_start: float, node_id: int):
        """Record a busy period for a given channel (considering other BSSs)."""
        self.events.append((time_start, None, node_id))

    def record_busy_end(self, time_end: float, node_id: int):
        for i in reversed(range(len(self.events))):
            start, end, n_id = self.events[i]
            if n_id == node_id and end is None:
                self.events[i] = (start, time_end, node_id)
                return

    def cleanup(self, current_time: float):
        """Remove outdated events based on the window duration."""
        threshold = current_time - self.window_duration_us
        self.events = [
            (start, end, n_id)
            for (start, end, n_id) in self.events
            if end is None or end >= threshold
        ]

    def get_utilization(self, current_time: float, ignore_nodes: set[int]) -> float:
        """Compute the utilization of the channel, excluding certain nodes."""
        self.cleanup(current_time)
        window_start = max(0.0, current_time - self.window_duration_us)
        intervals = []

        for start, end, n_id in self.events:
            if n_id in ignore_nodes:
                continue
            if end is None:
                end = current_time
            overlap_start = max(start, window_start)
            overlap_end = min(end, current_time)
            if overlap_end > overlap_start:
                intervals.append((overlap_start, overlap_end))

        if not intervals:
            return 0.0

        # Merge overlapping intervals to avoid double counting
        intervals.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))

        busy_total = sum(end - start for start, end in merged)
        elapsed = min(current_time, self.window_duration_us)

        return busy_total / elapsed if elapsed > 0 else 0.0


class Channel20MHz:
    """Represents a single 20 MHz wireless channel."""

    def __init__(
        self, cfg: cfg_module, sparams: sparams_module, env: simpy.Environment, id: int
    ):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.id = id

        self.nodes: dict[int, Node] = {}  # Nodes assigned to the channel
        self.nodes_sensing: dict[int, Node] = (
            {}
        )  # Nodes using the channel for carrier sensing
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

        self.utilization_tracker = UtilizationTracker(
            self.id, self.cfg.UTILIZATION_WINDOW_DURATION_US
        )

        self.name = "CHANNEL"
        self.logger = get_logger(self.name, cfg, sparams, env)

        self.rng = random.Random(cfg.SEED)

    def assign(self, node: Node):
        self.nodes[node.id] = node

    def release(self, node: Node):
        self.nodes.pop(node.id, None)

    def add_sensing_node(self, node: Node):
        self.nodes_sensing[node.id] = node

    def remove_sensing_node(self, node: Node):
        self.nodes_sensing.pop(node.id, None)

    def is_idle(self):
        """Returns True if the channel is idle, False if busy."""
        return self.idle_start_time is not None

    def has_been_idle_during_duration(self, duration_us: float) -> bool:
        return (
            (self.env.now - self.idle_start_time >= duration_us)
            if self.idle_start_time is not None
            else False
        )

    def occupy(self, node: Node):
        """Marks the channel as busy by a node and checks for collisions."""
        self.logger.debug(f"Channel {self.id} -> Occupied by {node.type} {node.id}")

        self.utilization_tracker.record_busy_start(self.env.now, node.id)

        self.idle_start_time = None

        if self.busy_start_time is None:
            self.busy_start_time = self.env.now

        self.nodes_transmitting[node.id] = node

        for node_id, p_node in self.nodes_sensing.items():
            p_node.phy_layer.channel_is_busy(self.id)

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

        for node_id, p_node in self.nodes_sensing.items():
            if node_id not in self.nav_nodes_ids:
                # If the node sensing on the channel is currently transmitting, it can not receive the RTS and thus NAV reservation
                if p_node.mac_layer.state != MACState.TX:
                    p_node.phy_layer.channel_is_busy(self.id)

    def end_nav(self, dst_id: int):
        if dst_id == self.nav_master_id:
            self.logger.debug(
                f"Channel {self.id} -> Channel reservation ended (NAV) by Node {dst_id}"
            )
            self.nav_occupied = False
            self.nav_master_id = None
            self.nav_nodes_ids = set()

            if len(self.nodes_transmitting) == 0:
                for node_id, p_node in self.nodes_sensing.items():
                    p_node.phy_layer.channel_is_idle(self.id)

    def unoccupy(self, node: Node):
        """Removes a node from the channel."""
        if node.id not in self.nodes_transmitting:
            return
        self.logger.debug(f"Channel {self.id} -> Unoccupied by {node.type} {node.id}")
        self.nodes_transmitting.pop(node.id, None)

        self.utilization_tracker.record_busy_end(self.env.now, node.id)

        if len(self.nodes_transmitting) == 0:
            self.stats.airtime_us += self.env.now - self.busy_start_time
            self.busy_start_time = None
            self.collision_detected = False
            self.idle_start_time = self.env.now

            for node_id, p_node in self.nodes_sensing.items():
                if not self.nav_occupied or node_id in self.nav_nodes_ids:
                    p_node.phy_layer.channel_is_idle(self.id)

    def handle_collision(self):
        """Handles collision."""
        self.collision_detected = True
        self.last_collision_time = self.env.now

    def rts_collision_detected(self):
        for node_id, node in self.nodes.items():
            if node.mac_layer.state != MACState.TX:
                node.phy_layer.rts_collision_detected(self.id)

    def ampdu_collision_detected(self):
        for node_id, node in self.nodes.items():
            if node.mac_layer.state != MACState.TX:
                node.phy_layer.ampdu_collision_detected(self.id)

    def successful_transmission_detected(self):
        for node_id, node in self.nodes.items():
            if node.mac_layer.state != MACState.TX:
                node.phy_layer.successful_transmission_detected(self.id)

    def get_utilization(self, current_time: float, ignore_nodes: set[int]):
        return self.utilization_tracker.get_utilization(current_time, ignore_nodes)


class Medium:
    def __init__(
        self,
        cfg: cfg_module,
        sparams: sparams_module,
        env: simpy.Environment,
        network: Network,
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

        self.rng = random.Random(cfg.SEED)

    def get_valid_bonds(self) -> list:
        available_channels = set(self.channels.keys())  # Extract available channel IDs

        valid_channel_bonds = []
        for bw, bond_list in VALID_BONDS.items():
            # Filter bonds that are fully contained in the available channels
            valid_channel_bonds.extend(
                [bond for bond in bond_list if bond.issubset(available_channels)]
            )

        return valid_channel_bonds

    def are_all_channels_idle(self):
        return all(ch.is_idle() for ch in self.channels.values())

    def are_all_node_channels_idle(self, channels_ids: set[int]):
        """Checks if all selected channels are idle."""
        return all(self.channels[ch_id].is_idle() for ch_id in channels_ids)

    def has_been_idle_during_duration(self, ch_id: int, duration_us: float) -> bool:
        return self.channels[ch_id].has_been_idle_during_duration(duration_us)

    def any_collision_detected(self, channels_ids: set[int], start_time_us: int):
        """Checks if any of the selected channels has a collision."""
        return any(
            self.channels[ch_id].last_collision_time >= start_time_us
            for ch_id in channels_ids
        )

    def assign_channels(self, node: Node, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].assign(node)

    def release_channels(self, node: Node, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].release(node)

    def add_sensing_channels(self, node: Node, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].add_sensing_node(node)

    def remove_sensing_channels(self, node: Node, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].remove_sensing_node(node)

    def occupy_channels(self, node, channels_ids: set[int]):
        if self.busy_start_time is None:
            self.busy_start_time = self.env.now

        for ch_id in channels_ids:
            self.channels[ch_id].occupy(node)

    def unoccupy_channels(self, node, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].unoccupy(node)

        if self.are_all_channels_idle():
            self.stats.airtime_us += self.env.now - self.busy_start_time
            self.busy_start_time = None

    def start_nav(self, src_id: int, dst_id: int, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].start_nav(src_id, dst_id)

    def end_nav(self, src_id, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].end_nav(src_id)

    def broadcast_channel_info(
        self, src_id, dst_id, channels_ids, sensing_channels_ids
    ):
        src_node = self.network.get_node(src_id)
        dst_node = self.network.get_node(dst_id)

        self.logger.header(
            f"Broadcasting channel info from {src_node.type} {src_id} to {dst_node.type} {dst_id}..."
        )

        dst_node.phy_layer.receive_channel_info(channels_ids, sensing_channels_ids)

    def broadcast_tx_channels_info(self, src_id, dst_id, channels_ids):
        src_node = self.network.get_node(src_id)
        dst_node = self.network.get_node(dst_id)

        self.logger.header(
            f"Broadcasting transmitting channels info from {src_node.type} {src_id} to {dst_node.type} {dst_id}..."
        )

        dst_node.phy_layer.receive_tx_channels_info(channels_ids)

    def get_contender_count(self):
        return [len(ch.nodes) for ch in self.channels.values()]

    def get_busy_flags(self):
        return [not ch.is_idle() for ch in self.channels.values()]

    def get_channels_utilization(self, ignore_nodes: set[int]):
        return [
            ch.get_utilization(self.env.now, ignore_nodes)
            for ch in self.channels.values()
        ]

    def rts_collision_detected(self, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].rts_collision_detected()

    def ampdu_collision_detected(self, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].ampdu_collision_detected()

    def successful_transmission_detected(self, channels_ids: set[int]):
        for ch_id in channels_ids:
            self.channels[ch_id].successful_transmission_detected()

    def transmit(
        self,
        ppdu: PPDU,
        channels_ids: set[int],
        mcs_index: int,
        nav_channels_ids: set[int] = [],
    ):
        self.logger.header(
            f"Transmitting {ppdu.data_unit.type} from node {ppdu.src_id} to node {ppdu.dst_id} over channel(s) {', '.join(map(str, channels_ids))}..."
        )

        self.occupy_channels(self.network.get_node(ppdu.src_id), channels_ids)

        tx_duration_us = get_tx_duration_us(
            sparams_module,
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
            self.receive(ppdu, channels_ids, mcs_index, nav_channels_ids)
            self.stats.ppdus_success += 1
            self.stats.tx_bytes += ppdu.size_bytes
            self.stats.tx_bytes_success += ppdu.size_bytes
        else:
            self.logger.warning(
                f"{ppdu.type} ({ppdu.data_unit.type}) from {ppdu.src_id} to {ppdu.dst_id} over channel(s) {', '.join(map(str, channels_ids))} collided!"
            )

            self.stats.ppdus_fail += 1
            self.stats.tx_bytes += ppdu.size_bytes
            if ppdu.data_unit.type == "RTS":
                self.rts_collision_detected(channels_ids)
            else:
                self.ampdu_collision_detected(channels_ids)

        self.unoccupy_channels(self.network.get_node(ppdu.src_id), channels_ids)

    def receive(
        self,
        ppdu: PPDU,
        channels_ids: set[int],
        mcs_index: int,
        nav_channels_ids: set[int] = [],
    ):
        distance_m = self.network.get_distance_between_nodes(ppdu.src_id, ppdu.dst_id)

        rssi_dbm = get_rssi_dbm(self.sparams, distance_m, self.cfg.SEED)
        min_sensitivity_dbm = get_min_sensitivity_dBm(mcs_index, len(channels_ids) * 20)

        if rssi_dbm < min_sensitivity_dbm:  # this does nothing if there is no mobility
            self.logger.warning(
                f"Unreliable PPDU reception (from {ppdu.src_id} to {ppdu.dst_id}): RSSI ({rssi_dbm:.2f} dBm) below min. sensitivity threshold ({min_sensitivity_dbm:.2f} dBm)"
            )
            node_src = self.network.get_node(ppdu.src_id)
            node_src.phy_layer.select_mcs_index(ppdu.dst_id)
            return

        if ppdu.data_unit.type == "AMPDU":
            for mpdu in ppdu.data_unit.mpdus:
                mpdu.is_corrupted = (
                    True
                    if self.sparams.MPDU_ERROR_PROBABILITY > self.rng.random()
                    else False
                )

        if self.sparams.ENABLE_RTS_CTS:
            if ppdu.data_unit.type == "RTS":
                self.start_nav(ppdu.src_id, ppdu.dst_id, nav_channels_ids)

        node_dst = self.network.get_node(ppdu.dst_id)
        node_dst.phy_layer.receive(ppdu)
