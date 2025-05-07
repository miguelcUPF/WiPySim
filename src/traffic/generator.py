from src.sim_params import SimParams as sparams_module
from src.user_config import UserConfig as cfg_module

from src.utils.data_units import Packet
from src.utils.event_logger import get_logger
from src.components.network import Node

import math
import simpy
import random

# Poisson/Bursty/VR Traffic Parameters
traffic_load_kbps = 100e3
MAX_PACKET_SIZE_bytes = 1280

# Bursty/VR Traffic Parameters
BURST_SIZE_pkts = 20
AVG_INTER_PACKET_TIME_us = 6

# VR Traffic Parameters
FPS = 90


class TrafficGenerator:
    def __init__(
        self,
        cfg: cfg_module,
        sparams: sparams_module,
        env: simpy.Environment,
        node: Node,
        dst_id: int,
        **kwargs,
    ):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.node = node

        self.src_id = node.id
        self.dst_id = dst_id

        self.traffic_model = kwargs.get("name", None)
        self.start_time_us = kwargs.get("start_time_us", 0)
        self.end_time_us = kwargs.get("end_time_us", None)
        self.traffic_load_kbps = kwargs.get("traffic_load_kbps", traffic_load_kbps)
        self.max_packet_size_bytes = kwargs.get(
            "max_packet_size_bytes", MAX_PACKET_SIZE_bytes
        )
        self.burst_size_pkts = kwargs.get("burst_size_pkts", BURST_SIZE_pkts)
        self.avg_inter_packet_time_us = kwargs.get(
            "avg_inter_packet_time_us", AVG_INTER_PACKET_TIME_us
        )
        self.fps = kwargs.get("fps", FPS)

        self.packet_id = 0

        self.name = "GEN"
        self.logger = get_logger(
            self.name,
            cfg,
            sparams,
            self.env,
            True if node.id in self.cfg.EXCLUDED_IDS else False,
        )

        self.active_processes = []

        self.env.process(self._delayed_run())

    def _delayed_run(self):
        yield self.env.timeout(self.start_time_us)

        self.run()

        if self.end_time_us:
            yield self.env.timeout(self.end_time_us - self.start_time_us)
            self.stop()

    def stop(self):
        """Stop all running traffic generation processes."""
        for process in self.active_processes:
            if process.is_alive:
                process.interrupt()

        self.logger.debug(
            f"{self.node.type} {self.src_id} -> Traffic Generator stopped."
        )
        self.active_processes.clear()

    def run(self):
        match self.traffic_model:
            case "Poisson":
                p = self.env.process(self.generate_poisson_traffic())
            case "Bursty":
                p = self.env.process(self.generate_bursty_traffic())
            case "VR":
                p = self.env.process(self.generate_vr_traffic())
            case "Full":
                p = self.env.process(self.generate_full_traffic())
            case _:
                self.logger.error(
                    f"{self.node.type} {self.src_id} -> Invalid traffic model specified (from node {self.src_id} to node {self.dst_id}): {self.traffic_model}"
                )

        self.active_processes.append(p)

    def generate_poisson_traffic(self):
        """Generates Poisson traffic"""
        avg_inter_pkt_time_us = (
            ((self.max_packet_size_bytes * 8) / (self.traffic_load_kbps * 1000) * 1e6)
            if self.traffic_load_kbps > 0
            else None
        )
        if avg_inter_pkt_time_us is None:
            return
        try:
            while True:
                inter_arrival_time_us = (
                    random.expovariate(1 / avg_inter_pkt_time_us)
                    if avg_inter_pkt_time_us > 0
                    else 1
                )
                yield self.env.timeout(int(math.ceil(inter_arrival_time_us)))
                self._create_and_send_packet(self.max_packet_size_bytes)
        except simpy.Interrupt:
            pass

    def generate_bursty_traffic(self):
        """Generates bursts of packets"""
        avg_inter_burst_time_us = (
            (
                self.burst_size_pkts
                * (self.max_packet_size_bytes * 8)
                / (self.traffic_load_kbps * 1000)
                * 1e6
            )
            if self.traffic_load_kbps > 0
            else None
        )
        if avg_inter_burst_time_us is None:
            return
        try:
            while True:
                inter_burst_time_us = (
                    random.expovariate(1 / avg_inter_burst_time_us)
                    if avg_inter_burst_time_us > 0
                    else 1
                )
                yield self.env.timeout(int(math.ceil(inter_burst_time_us)))

                self.env.process(self.send_burst(avg_inter_burst_time_us))
        except simpy.Interrupt:
            pass

    def send_burst(self, avg_inter_burst_time_us):
        try:
            for _ in range(self.burst_size_pkts):
                self._create_and_send_packet(self.max_packet_size_bytes)

                inter_pkt_time_us = min(
                    (
                        random.expovariate(1 / self.avg_inter_packet_time_us)
                        if self.avg_inter_packet_time_us > 0
                        else 1
                    ),
                    (
                        avg_inter_burst_time_us / self.burst_size_pkts
                        if self.burst_size_pkts > 0
                        else 1
                    ),
                )

                yield self.env.timeout(int(math.ceil(inter_pkt_time_us)))
        except simpy.Interrupt:
            pass

    def generate_vr_traffic(self):
        """Generates VR traffic bursts every 1/FPS"""
        avg_inter_frame_time_us = 1 / self.fps * 1e6
        bits_per_frame = (self.traffic_load_kbps * 1000) / self.fps
        packets_per_frame = (
            math.floor(bits_per_frame / (self.max_packet_size_bytes * 8))
            if self.max_packet_size_bytes > 0
            else 0
        )
        remainder_bytes = bits_per_frame % (self.max_packet_size_bytes * 8) // 8

        try:
            while True:
                yield self.env.timeout(int(math.ceil(avg_inter_frame_time_us)))

                self.env.process(
                    self.send_frame(
                        avg_inter_frame_time_us, packets_per_frame, remainder_bytes
                    )
                )
        except simpy.Interrupt:
            pass

    def send_frame(self, avg_inter_frame_time_us, packets_per_frame, remainder_bytes):
        remaining_frame_time_us = avg_inter_frame_time_us

        try:
            for i in range(packets_per_frame - 1):
                self._create_and_send_packet(self.max_packet_size_bytes)

                inter_pkt_time_us = min(
                    (
                        random.expovariate(1 / self.avg_inter_packet_time_us)
                        if self.avg_inter_packet_time_us > 0
                        else 1
                    ),
                    (
                        remaining_frame_time_us / (packets_per_frame - i)
                        if packets_per_frame - i > 0
                        else 1
                    ),
                )
                yield self.env.timeout(int(math.ceil(inter_pkt_time_us)))
                remaining_frame_time_us -= inter_pkt_time_us

            if remainder_bytes > 0:
                self._create_and_send_packet(remainder_bytes)
        except simpy.Interrupt:
            pass

    def generate_full_traffic(self):
        try:
            while True:
                for _ in range(
                    self.node.mac_layer.sparams.MAX_TX_QUEUE_SIZE_pkts
                    - len(self.node.mac_layer.tx_queue.items)
                ):
                    self._create_and_send_packet(self.max_packet_size_bytes)
                event = self.env.event()
                self.node.mac_layer.non_full_event = event
                yield event
        except simpy.Interrupt:
            pass

    def _create_and_send_packet(self, packet_size: int):
        """Creates a packet and sends it to the App Layer."""
        self.packet_id += 1
        packet = Packet(
            id=self.packet_id,
            size_bytes=packet_size,
            src_id=self.src_id,
            dst_id=self.dst_id,
            creation_time_us=self.env.now,
        )
        self.logger.debug(f"{self.node.type} {self.src_id} -> Created {packet}")
        self.node.app_layer.packet_to_mac(packet)
