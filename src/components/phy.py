from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.components.network import Node, AP
from src.utils.event_logger import get_logger
from src.utils.data_units import DataUnit, PPDU
from src.utils.mcs_table import get_highest_mcs_index
from src.utils.transmission import get_rssi_dbm


from typing import cast

import random
import simpy


class PHY:
    def __init__(self, cfg: cfg, sparams: sparams, env: simpy.Environment, node: Node):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.node = node

        self.channels_ids = set()  # Allocated channels
        self.sensing_channels_ids = (
            set()
        )  # If BONDING_MODE is 0 or 1 sensing channels includes only the primary channel. Secondary channels are thus self.channels_ids - self.sensing_channels_ids
        self.transmitting_channels_ids = set()
        # if BONDING_MODE is 0 data transmission should occur on the entire bond (primary and secondary channels)
        # if BONDING_MODE is 1 data transmission can occur in a subset of contiguous channels following IEEE 802.11 channelization
        # if BONDING_MODE is 2 data transmission can occur in a subset of non-contiguous channels

        self.busy_channels_ids = set()

        self.mcs_indexes = {}

        self.name = "PHY"
        self.logger = get_logger(self.name, cfg, sparams, env)

        self.idle_events = {}
        self.busy_events = {}

        self.rts_collision_events = {}
        self.ampdu_collision_events = {}
        self.last_collision_times = {}

        self.env.process(self.run())

    def stop(self):
        pass

    def set_channels(self, channels_ids: set[int]):
        if self.channels_ids == channels_ids:
            return
        self.node.medium.release_channels(self.node, self.channels_ids)

        self.channels_ids = channels_ids

        self.reset_events()
        self.node.medium.assign_channels(self.node, channels_ids)

        self.set_transmitting_channels(channels_ids)

    def set_sensing_channels(self, channels_ids: set[int]):
        if self.sensing_channels_ids == channels_ids:
            return
        self.node.medium.remove_sensing_channels(self.node, self.sensing_channels_ids)

        self.sensing_channels_ids = channels_ids

        self.node.medium.add_sensing_channels(self.node, self.sensing_channels_ids)

    def reset_events(self):
        for ch_id in self.channels_ids:
            self.reset_idle_event(ch_id)
            self.reset_busy_event(ch_id)
            self.reset_rts_collision_event(ch_id)
            self.reset_ampdu_collision_event(ch_id)

    def reset_collision_events(self):
        for ch_id in self.channels_ids:
            self.reset_rts_collision_event(ch_id)
            self.reset_ampdu_collision_event(ch_id)

    def set_transmitting_channels(self, channels_ids: set[int]):
        if self.transmitting_channels_ids == channels_ids:
            return
        self.transmitting_channels_ids = channels_ids

        self.broadcast_tx_channels_info()

    def get_idle_event(self, ch_id: int):
        return self.idle_events[ch_id]

    def has_been_idle_during_duration(self, ch_id: int, duration_us: float) -> bool:
        return self.node.medium.has_been_idle_during_duration(ch_id, duration_us)

    def get_busy_event(self, ch_id: int):
        return self.busy_events[ch_id]

    def reset_idle_event(self, ch_id: int):
        self.idle_events[ch_id] = self.env.event()

    def reset_busy_event(self, ch_id: int):
        self.busy_events[ch_id] = self.env.event()

    def get_rts_collision_event(self, ch_id: int):
        return self.rts_collision_events[ch_id]

    def get_ampdu_collision_event(self, ch_id: int):
        return self.ampdu_collision_events[ch_id]

    def reset_rts_collision_event(self, ch_id: int):
        self.rts_collision_events[ch_id] = self.env.event()

    def reset_ampdu_collision_event(self, ch_id: int):
        self.ampdu_collision_events[ch_id] = self.env.event()

    def get_last_collision_time(self, ch_id: int):
        return self.last_collision_times[ch_id]

    def get_ampdu_collisions_channels_ids(self):
        return [
            ch_id
            for ch_id in self.channels_ids
            if self.ampdu_collision_events[ch_id].triggered
        ]

    def get_primary_channel_id(self):
        if len(self.sensing_channels_ids) > 1:
            self.logger.critical(
                f"{self.node.type} {self.node.id} -> Sensing channels does not contain only one (primary) channel! (Channels: {', '.join(map(str, self.node.phy_layer.sensing_channels_ids))})"
            )
        return next(iter(self.sensing_channels_ids))

    def get_valid_bonds(self):
        return self.node.medium.get_valid_bonds()

    def select_channels(self):
        # TODO: implement channel selection at the MAC layer
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Allocating channels at random..."
        )

        valid_bonds = self.get_valid_bonds()
        channels_ids = random.choice(valid_bonds)

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Allocated channels: {', '.join(map(str, channels_ids))}"
        )

        self.set_channels(set(channels_ids))

        if self.sparams.BONDING_MODE in [0, 1]:
            self.set_sensing_channels({self.select_primary_channel()})
        else:
            self.set_sensing_channels(channels_ids)

        self.broadcast_channel_info()

    def select_primary_channel(self) -> int:
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Selecting primary channel..."
        )

        primary_channel_id = min(self.channels_ids)

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Selected primary channel: {primary_channel_id}"
        )

        return primary_channel_id

    def select_mcs_index(self, sta_id: int):
        self.logger.debug(f"Node {self.node.id} -> Selecting MCS for STA {sta_id}...")

        # Predict associated STA RSSI
        distance_m = self.node.network.get_distance_between_nodes(self.node.id, sta_id)
        rssi_dbm = get_rssi_dbm(self.sparams, distance_m)

        # Get the highest MCS index that can be supported
        mcs_index = get_highest_mcs_index(
            rssi_dbm, len(self.channels_ids) * 20, self.node.id, sta_id
        )

        # If no MCS index can be supported, select MCS 0
        if mcs_index == -1:
            self.logger.warning(
                f"{self.node.type} {self.node.id} -> RSSI ({rssi_dbm:.2f} dBm) too low to support any MCS index for STA {sta_id}. Selecting MCS 0. Please reallocate AP {self.node.id} and STA {sta_id} close to each other."
            )
            mcs_index = 0

        self.mcs_indexes[sta_id] = mcs_index

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Selected MCS index for STA {sta_id}: {mcs_index}"
        )

    def select_all_mcs_indexs(self):
        if not isinstance(self.node, AP):
            return

        ap = cast(AP, self.node)

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Selecting MCS indexes..."
        )

        for sta in ap.get_stas():
            self.select_mcs_index(sta.id)

    def broadcast_channel_info(self):
        """If the node is an AP, send the channel & MCS info to all associated STAs."""
        if not isinstance(self.node, AP):
            return

        ap = cast(AP, self.node)

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Broadcasting channel info to associated STAs..."
        )

        for sta in ap.get_stas():
            self.node.medium.broadcast_channel_info(
                ap.id, sta.id, self.channels_ids, self.sensing_channels_ids
            )

    def broadcast_tx_channels_info(self):
        if not isinstance(self.node, AP):
            return

        ap = cast(AP, self.node)

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Broadcasting transmitting channels info to associated STAs..."
        )

        for sta in ap.get_stas():
            self.node.medium.broadcast_tx_channels_info(
                ap.id, sta.id, self.transmitting_channels_ids
            )

    def receive_channel_info(
        self, channels_ids: set[int], sensing_channels_ids: set[int]
    ):
        self.set_channels(channels_ids)
        self.set_sensing_channels(sensing_channels_ids)

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Updated PHY settings (channels_ids: {', '.join(map(str, self.channels_ids))}; sensing_channels_ids: {', '.join(map(str, self.sensing_channels_ids))})"
        )

    def receive_tx_channels_info(self, channels_ids: set[int]):
        self.transmitting_channels_ids = channels_ids

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Updated PHY settings (transmitting_channels_ids: {', '.join(map(str, self.transmitting_channels_ids))})"
        )

    def are_all_sensing_channels_idle(self):
        return self.node.medium.are_all_node_channels_idle(self.sensing_channels_ids)

    def is_channel_idle(self, ch_id: int):
        return self.node.medium.are_all_node_channels_idle([ch_id])

    def get_busy_sensing_channels(self):
        return [
            ch_id
            for ch_id in self.sensing_channels_ids
            if ch_id in self.busy_channels_ids
        ]

    def channel_is_busy(self, ch_id: int):
        self.busy_channels_ids.add(ch_id)
        if (
            self.busy_events[ch_id] is not None
            and not self.busy_events[ch_id].triggered
        ):
            self.busy_events[ch_id].succeed()

    def channel_is_idle(self, ch_id: int):
        (
            self.busy_channels_ids.remove(ch_id)
            if ch_id in self.busy_channels_ids
            else None
        )
        if (
            self.idle_events[ch_id] is not None
            and not self.idle_events[ch_id].triggered
        ):
            self.idle_events[ch_id].succeed()

    def end_nav(self):
        self.node.medium.end_nav(self.node.id, self.sensing_channels_ids)

    def transmit(self, data_unit: DataUnit):
        if (
            not self.transmitting_channels_ids
            or self.transmitting_channels_ids not in self.get_valid_bonds()
        ):
            self.logger.error(
                f"{self.node.type} {self.node.id} -> Not associated to any valid channel, cannot transmit"
            )
            return

        if data_unit.is_mgmt_ctrl_frame:
            # For management and control frames, use the sensing channels (i.e., primary channel if BONDING_MODE is 0 or 1) and MCS index 0
            mcs_index = 0
            tx_channels_ids = self.sensing_channels_ids
        else:
            mcs_index = self.mcs_indexes[data_unit.dst_id]
            tx_channels_ids = self.transmitting_channels_ids

        nav_channels_ids = self.sensing_channels_ids if data_unit.type == "RTS" else []

        ppdu = PPDU(data_unit, self.env.now)

        self.node.tx_stats.tx_phy_bytes += ppdu.size_bytes

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Sending {ppdu.type} to node {ppdu.dst_id} over channel(s) {', '.join(map(str, tx_channels_ids))}..."
        )

        yield self.env.process(
            self.node.medium.transmit(
                ppdu, tx_channels_ids, mcs_index, nav_channels_ids
            )
        )

    def rts_collision_detected(self, ch_id: int):
        self.last_collision_times[ch_id] = self.env.now
        (
            self.rts_collision_events[ch_id].succeed()
            if not self.rts_collision_events[ch_id].triggered
            else None
        )

    def ampdu_collision_detected(self, ch_id: int):
        self.last_collision_times[ch_id] = self.env.now
        (
            self.ampdu_collision_events[ch_id].succeed()
            if not self.ampdu_collision_events[ch_id].triggered
            else None
        )

    def successful_transmission_detected(self, ch_id: int):
        self.reset_rts_collision_event(ch_id)
        self.reset_ampdu_collision_event(ch_id)

    def receive(self, ppdu: PPDU):
        self.logger.info(
            f"{self.node.type} {self.node.id} -> Received {ppdu.type} from node {ppdu.src_id}"
        )

        ppdu.reception_time_us = self.env.now

        self.node.rx_stats.rx_phy_bytes += ppdu.size_bytes

        data_unit = ppdu.data_unit

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Forwarding received {data_unit.type} from PHY to MAC..."
        )

        self.node.mac_layer.receive(data_unit)

    def run(self):
        if isinstance(self.node, AP):
            self.select_channels()
            self.select_all_mcs_indexs()
            yield self.env.timeout(0)
