import math
import simpy
import random
from src.sim_config import TRAFFIC_MODEL, MAX_PACKET_SIZE_bytes, APP_TRAFFIC_LOAD_kbps, FPS, AVG_INTER_PACKET_TIME_us, BURST_SIZE_pkts
from src.utils.data_units import Packet
from src.utils.event_logger import get_logger

class TrafficGenerator:
    def __init__(self, env: simpy.Environment, source: int, destination: int, app_layer):
        self.env = env
        self.name = "GEN"
        self.source = source
        self.destination = destination
        self.app_layer = app_layer
        self.packet_id = 0

        self.logger = get_logger(self.name, self.env)

    def run(self):
        match TRAFFIC_MODEL:
            case "Poisson":
                yield self.env.process(self.generate_poisson_traffic())
            case "Bursty":
                yield self.env.process(self.generate_bursty_traffic())
            case 'VR':
                yield self.env.process(self.generate_vr_traffic())
            case _:
                raise ValueError(
                    "Invalid traffic model specified in sim_config.")

    def generate_poisson_traffic(self):
        """Generates Poisson traffic"""
        avg_inter_pkt_time_us = (MAX_PACKET_SIZE_bytes * 8) / \
            (APP_TRAFFIC_LOAD_kbps * 1000) * 1e6
        while True:
            inter_arrival_time_us = random.expovariate(
                1 / avg_inter_pkt_time_us)
            yield self.env.timeout(int(math.ceil(inter_arrival_time_us)))
            self._create_and_send_packet(MAX_PACKET_SIZE_bytes)

    def generate_bursty_traffic(self):
        """Generates bursts of packets"""
        avg_inter_burst_time_us = (BURST_SIZE_pkts * (MAX_PACKET_SIZE_bytes * 8) /
                                   (APP_TRAFFIC_LOAD_kbps * 1000) * 1e6)

        while True:
            inter_burst_time_us = random.expovariate(
                1 / avg_inter_burst_time_us)
            yield self.env.timeout(int(math.ceil(inter_burst_time_us)))

            self.env.process(self.send_burst(avg_inter_burst_time_us))

    def send_burst(self, avg_inter_burst_time_us):
        for _ in range(BURST_SIZE_pkts):
            self._create_and_send_packet(MAX_PACKET_SIZE_bytes)

            inter_pkt_time_us = min(random.expovariate(
                1 / AVG_INTER_PACKET_TIME_us), avg_inter_burst_time_us / BURST_SIZE_pkts)

            yield self.env.timeout(int(math.ceil(inter_pkt_time_us)))

    def generate_vr_traffic(self):
        """Generates VR traffic bursts every 1/FPS"""
        avg_inter_frame_time_us = 1 / FPS * 1e6
        bits_per_frame = (APP_TRAFFIC_LOAD_kbps * 1000) / FPS
        packets_per_frame = math.floor(
            bits_per_frame / (MAX_PACKET_SIZE_bytes * 8))
        remainder_bytes = bits_per_frame % (MAX_PACKET_SIZE_bytes * 8) // 8

        while True:
            yield self.env.timeout(int(math.ceil(avg_inter_frame_time_us)))

            self.env.process(self.send_frame(
                avg_inter_frame_time_us, packets_per_frame, remainder_bytes))

    def send_frame(self, avg_inter_frame_time_us, packets_per_frame, remainder_bytes):
        remaining_frame_time_us = avg_inter_frame_time_us

        for i in range(packets_per_frame-1):
            self._create_and_send_packet(MAX_PACKET_SIZE_bytes)

            inter_pkt_time_us = min(
                random.expovariate(1 / AVG_INTER_PACKET_TIME_us),
                remaining_frame_time_us / (packets_per_frame - i)
            )
            yield self.env.timeout(int(math.ceil(inter_pkt_time_us)))
            remaining_frame_time_us -= inter_pkt_time_us

        if remainder_bytes > 0:
            self._create_and_send_packet(remainder_bytes)

    def _create_and_send_packet(self, packet_size: int):
        """Creates a packet and sends it to the App Layer."""
        self.packet_id += 1
        packet = Packet(
            id=self.packet_id,
            size_bytes=packet_size,
            source=self.source,
            destination=self.destination,
            creation_time_us=self.env.now
        )
        self.logger.debug(f"Created {packet}")
        self.app_layer.packet_to_mac(packet)
