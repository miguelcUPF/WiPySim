from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.components.network import Node, AP
from src.utils.event_logger import get_logger
from src.utils.data_units import DataUnit, PPDU


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

        self.mcs_index = None

        self.name = "PHY"
        self.logger = get_logger(self.name, cfg, sparams, env)

        self.env.process(self.run())

    def set_channels(self, channels_ids: list[int]):
        self.channels_ids = channels_ids

    def stop(self):
        pass

    def select_channels(self):
        # TODO: implement channels selection
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Selecting channels at random..."
        )
        valid_channels = self.node.medium.get_valid_channels()

        self.set_channels(random.choice(valid_channels))

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Selected channels: {', '.join(map(str, self.channels_ids))}"
        )

    def select_primary_channel_id(self):
        # TODO: implement primary channel selection
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Selecting primary channel at random..."
        )

        self.primary_channel_id = random.choice(self.channels_ids)

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Selected primary channel: {self.primary_channel_id}"
        )

    def select_mcs_index(self) -> int:
        # TODO: implement MCS selection
        self.logger.header(f"Node {self.node.id} -> Selecting MCS...")

        self.mcs_index = 11

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Selected MCS: {self.mcs_index}"
        )

    def broadcast_channel_info(self):
        """If the node is an AP, send the channel & MCS info to all associated STAs."""
        if self.node.type != "AP":
            return

        ap = cast(AP, self.node)

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Broadcasting channel info to associated STAs..."
        )

        for sta in ap.get_stas():
            self.node.medium.broadcast_channel_info(
                ap.id, sta.id, self.channels_ids, self.primary_channel_id
            )

    def broadcast_mcs_info(self):
        if self.node.type != "AP":
            return

        ap = cast(AP, self.node)

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Broadcasting MCS info to associated STAs..."
        )

        for sta in ap.get_stas():
            self.node.medium.broadcast_mcs_info(ap.id, sta.id, self.mcs_index)

    def receive_channel_info(self, channels_ids: list[int], primary_channel_id: int):
        self.set_channels(channels_ids)
        self.primary_channel_id = primary_channel_id

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
            mcs_index = self.mcs_index

        ppdu = PPDU(data_unit, self.env.now)

        self.node.tx_stats.tx_phy_bytes += ppdu.size_bytes

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Sending {ppdu.type} to node {ppdu.dst_id} over channel(s) {', '.join(map(str, self.channels_ids))}..."
        )

        yield self.env.process(self.node.medium.transmit(ppdu, tx_channels, mcs_index))

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
        if self.node.type == "AP":
            self.select_channels()
            self.select_primary_channel_id()
            self.select_mcs_index()
            self.broadcast_channel_info()
            self.broadcast_mcs_info()
            yield self.env.timeout(0)
