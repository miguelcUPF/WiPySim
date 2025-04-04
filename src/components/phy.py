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

        self.channels_ids = []
        self.primary_channel_id = None

        self.mcs_indexes = {}

        self.name = "PHY"
        self.logger = get_logger(self.name, cfg, sparams, env)

        self.env.process(self.run())

    def set_channels(self, channels_ids: list[int]):
        self.node.medium.release_channels(self.node, self.channels_ids)

        self.channels_ids = channels_ids
        self.node.medium.assign_channels(self.node, channels_ids)

    def set_primary_channel(self, id: int):
        (
            self.node.medium.release_as_primary_channel(
                self.node, self.primary_channel_id
            )
            if self.primary_channel_id
            else None
        )

        self.primary_channel_id = id
        self.node.medium.assign_as_primary_channel(self.node, id)

    def stop(self):
        pass

    def select_channels(self):
        # TODO: implement channel selection
        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Selecting channels at random..."
        )
        valid_channels = self.node.medium.get_valid_channels()

        self.set_channels(random.choice(valid_channels))

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Selected channels: {', '.join(map(str, self.channels_ids))}"
        )

    def select_primary_channel_id(self):
        # TODO: implement primary channel selection
        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Selecting primary channel at random..."
        )

        self.set_primary_channel(random.choice(self.channels_ids))

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Selected primary channel: {self.primary_channel_id}"
        )

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

    def select_mcs_indexes(self):
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
                ap.id, sta.id, self.channels_ids, self.primary_channel_id
            )

    def broadcast_mcs_info(self, sta_id: int):
        if not isinstance(self.node, AP):
            return

        ap = cast(AP, self.node)

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Broadcasting MCS info to STA {sta_id}..."
        )
        
        self.node.medium.broadcast_mcs_info(ap.id, sta_id, self.mcs_indexes[sta_id])

    def receive_channel_info(self, channels_ids: list[int], primary_channel_id: int):
        self.set_channels(channels_ids)
        self.set_primary_channel(primary_channel_id)

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Updated PHY settings (channels_ids: {', '.join(map(str, self.channels_ids))}, primary_channel_id: {self.primary_channel_id})"
        )

    def receive_mcs_info(self, mcs_index: int):
        self.mcs_index = mcs_index

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Updated PHY settings (mcs_index: {self.mcs_index})"
        )

    def is_primary_channel_idle(self):
        return self.node.medium.are_channels_idle(self.node, [self.primary_channel_id])

    def set_primary_busy(self):
        self.node.mac_layer.set_primary_busy()

    def set_primary_idle(self):
        self.node.mac_layer.set_primary_idle()

    def end_nav(self):
        self.node.medium.end_nav(self.node.id, self.channels_ids)

    def transmit(self, data_unit: DataUnit):
        if (
            not self.channels_ids
            or self.channels_ids not in self.node.medium.get_valid_channels()
        ):
            self.logger.error(
                f"{self.node.type} {self.id} -> Not associated to any valid channel, cannot transmit"
            )
            return

        if data_unit.is_mgmt_ctrl_frame:
            tx_channels = [self.primary_channel_id]
            mcs_index = 0
        else:
            tx_channels = self.channels_ids
            mcs_index = self.mcs_indexes[data_unit.dst_id]

        ppdu = PPDU(data_unit, self.env.now)

        self.node.tx_stats.tx_phy_bytes += ppdu.size_bytes

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Sending {ppdu.type} to node {ppdu.dst_id} over channel(s) {', '.join(map(str, self.channels_ids))}..."
        )

        yield self.env.process(self.node.medium.transmit(ppdu, tx_channels, mcs_index))

    def rts_collision_detected(self):
        self.node.mac_layer.rts_collision_detected()

    def ampdu_collision_detected(self):
        self.node.mac_layer.ampdu_collision_detected()

    def successful_transmission_detected(self):
        self.node.mac_layer.successful_transmission_detected()

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
            self.select_primary_channel_id()
            self.select_mcs_indexes()
            self.broadcast_channel_info()
            yield self.env.timeout(0)
