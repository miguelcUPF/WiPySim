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

        self.channels_ids = set()  # Channels selected
        self.sensing_channels_ids = (
            set()
        )  # if BONDING_MODE == 0 then only the primary channel is used; otherwise all channels are used
        self.transmitting_channels_ids = (
            set()
        )  # if BONDING_MODE == 1 it can be a subset of sensing_channels_ids; otherwise all channels are used

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
        self.node.medium.release_channels(self.node, self.channels_ids)

        self.channels_ids = channels_ids

        self.reset_events()
        self.node.medium.assign_channels(self.node, channels_ids)

        self.set_transmitting_channels(channels_ids)

    def set_sensing_channels(self, channels_ids: set[int]):
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
        self.transmitting_channels_ids = channels_ids

        self.broadcast_tx_channels_info()

    def get_idle_event(self, ch_id: int):
        return self.idle_events[ch_id]

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

    def select_channels(self):
        # TODO: implement channel selection at the MAC layer
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Selecting channels at random..."
        )

        valid_channels = self.node.medium.get_valid_channels()
        channels_ids = random.choice(valid_channels)

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Selected channels: {', '.join(map(str, channels_ids))}"
        )

        self.set_channels(channels_ids)

        if self.sparams.BONDING_MODE == 0:
            primary_channel_id = self.select_primary_channel()
            self.set_sensing_channels(set([primary_channel_id]))
        else:
            self.set_sensing_channels(self.channels_ids)

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

        self.broadcast_mcs_info(sta_id)

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

    def broadcast_mcs_info(self, sta_id: int):
        if not isinstance(self.node, AP):
            return

        ap = cast(AP, self.node)

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Broadcasting MCS info to STA {sta_id}..."
        )

        self.node.medium.broadcast_mcs_info(ap.id, sta_id, self.mcs_indexes[sta_id])

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

    def receive_mcs_info(self, mcs_index: int):
        self.mcs_index = mcs_index

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Updated PHY settings (mcs_index: {self.mcs_index})"
        )

    def receive_tx_channels_info(self, channels_ids: set[int]):
        self.transmitting_channels_ids = channels_ids

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Updated PHY settings (transmitting_channels_ids: {', '.join(map(str, self.transmitting_channels_ids))})"
        )

    def are_all_sensing_channels_idle(self):
        return self.node.medium.are_all_node_channels_idle(
            self.node, self.sensing_channels_ids
        )

    def is_any_sensing_channel_idle(self):
        return self.node.medium.is_any_node_channel_idle(
            self.node, self.sensing_channels_ids
        )

    def is_channel_idle(self, ch_id: int):
        return self.node.medium.are_all_node_channels_idle(self.node, [ch_id])

    def get_busy_sensing_channels(self):
        return [
            ch_id
            for ch_id in self.sensing_channels_ids
            if ch_id in self.busy_channels_ids
        ]

    def channel_is_busy(self, ch_id: int):
        self.busy_channels_ids.add(ch_id)
        if any(ch_id in self.busy_channels_ids for ch_id in self.sensing_channels_ids):
            self.node.mac_layer.trigger_any_busy_event()
        if (
            self.busy_events[ch_id] is not None
            and not self.busy_events[ch_id].triggered
        ):
            self.busy_events[ch_id].succeed()

    def channel_is_idle(self, ch_id: int):
        self.busy_channels_ids.remove(ch_id)
        if all(
            ch_id not in self.busy_channels_ids for ch_id in self.sensing_channels_ids
        ):
            self.node.mac_layer.trigger_all_idle_event()
        if (
            self.idle_events[ch_id] is not None
            and not self.idle_events[ch_id].triggered
        ):
            self.idle_events[ch_id].succeed()

    def end_nav(self):
        self.node.medium.end_nav(self.node.id, self.channels_ids)

    def transmit(self, data_unit: DataUnit):
        if (
            not self.transmitting_channels_ids
            or self.transmitting_channels_ids
            not in self.node.medium.get_valid_channels()
        ):
            self.logger.error(
                f"{self.node.type} {self.node.id} -> Not associated to any valid channel, cannot transmit"
            )
            return

        if data_unit.is_mgmt_ctrl_frame:
            # For management and control frames, use the primary channel (if MODE 0) and MCS index 0
            mcs_index = 0
            tx_channels_ids = (
                self.sensing_channels_ids
                if self.sparams.BONDING_MODE == 0
                else self.transmitting_channels_ids
            )
        else:
            mcs_index = self.mcs_indexes[data_unit.dst_id]
            tx_channels_ids = self.transmitting_channels_ids

        nav_channels_ids = (
            self.transmitting_channels_ids if data_unit.type == "RTS" else []
        )

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
