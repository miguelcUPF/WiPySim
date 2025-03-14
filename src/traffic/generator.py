import math
import simpy
import random
from src.utils.data_units import Packet
from src.utils.event_logger import get_logger

# Poisson/Bursty/VR Traffic Parameters #
APP_TRAFFIC_LOAD_kbps = 100e3
MAX_PACKET_SIZE_bytes = 1280

# Bursty/VR Traffic Parameters #
BURST_SIZE_pkts = 20
AVG_INTER_PACKET_TIME_us = 6

# VR Traffic Parameters #
FPS = 90

class TrafficGenerator:
    def __init__(self, env: simpy.Environment, source: int, destination: int, app_layer, traffic_model: str, **kwargs):
        self.env = env
        self.name = "GEN"
        self.source = source
        self.destination = destination
        self.app_layer = app_layer

        self.traffic_model = traffic_model
        
        self.start_time_us = kwargs.get("start_time_us", 0)
        self.app_traffic_load_kbps = kwargs.get("app_traffic_load_kbps", APP_TRAFFIC_LOAD_kbps)
        self.max_packet_size_bytes = kwargs.get("max_packet_size_bytes", MAX_PACKET_SIZE_bytes)
        self.burst_size_pkts = kwargs.get("burst_size_pkts", BURST_SIZE_pkts)
        self.avg_inter_packet_time_us = kwargs.get("avg_inter_packet_time_us", AVG_INTER_PACKET_TIME_us)
        self.fps = kwargs.get("fps", FPS)

        self.packet_id = 0

        self.logger = get_logger(self.name, self.env)
        
        self.env.process(self._delayed_run())

    def _delayed_run(self):
        yield self.env.timeout(self.start_time_us)

        yield self.env.process(self.run())

    def run(self):
        match self.traffic_model:
            case "Poisson":
                yield self.env.process(self.generate_poisson_traffic())
            case "Bursty":
                yield self.env.process(self.generate_bursty_traffic())
            case 'VR':
                yield self.env.process(self.generate_vr_traffic())
            case _:
                self.logger.warning(
                    f"Invalid traffic model specified (from node {self.source} to node {self.destination}): {self.traffic_model}")

    def generate_poisson_traffic(self):
        """Generates Poisson traffic"""
        avg_inter_pkt_time_us = (self.max_packet_size_bytes * 8) / \
            (self.app_traffic_load_kbps * 1000) * 1e6
        while True:
            inter_arrival_time_us = random.expovariate(
                1 / avg_inter_pkt_time_us)
            yield self.env.timeout(int(math.ceil(inter_arrival_time_us)))
            self._create_and_send_packet(self.max_packet_size_bytes)

    def generate_bursty_traffic(self):
        """Generates bursts of packets"""
        avg_inter_burst_time_us = (self.burst_size_pkts * (self.max_packet_size_bytes * 8) /
                                   (self.app_traffic_load_kbps * 1000) * 1e6)

        while True:
            inter_burst_time_us = random.expovariate(
                1 / avg_inter_burst_time_us)
            yield self.env.timeout(int(math.ceil(inter_burst_time_us)))

            self.env.process(self.send_burst(avg_inter_burst_time_us))

    def send_burst(self, avg_inter_burst_time_us):
        for _ in range(self.burst_size_pkts):
            self._create_and_send_packet(self.max_packet_size_bytes)

            inter_pkt_time_us = min(random.expovariate(
                1 / self.avg_inter_packet_time_us), avg_inter_burst_time_us / self.burst_size_pkts)

            yield self.env.timeout(int(math.ceil(inter_pkt_time_us)))

    def generate_vr_traffic(self):
        """Generates VR traffic bursts every 1/FPS"""
        avg_inter_frame_time_us = 1 / self.fps * 1e6
        bits_per_frame = (self.app_traffic_load_kbps * 1000) / self.fps
        packets_per_frame = math.floor(
            bits_per_frame / (self.max_packet_size_bytes * 8))
        remainder_bytes = bits_per_frame % (self.max_packet_size_bytes * 8) // 8

        while True:
            yield self.env.timeout(int(math.ceil(avg_inter_frame_time_us)))

            self.env.process(self.send_frame(
                avg_inter_frame_time_us, packets_per_frame, remainder_bytes))

    def send_frame(self, avg_inter_frame_time_us, packets_per_frame, remainder_bytes):
        remaining_frame_time_us = avg_inter_frame_time_us

        for i in range(packets_per_frame-1):
            self._create_and_send_packet(self.max_packet_size_bytes)

            inter_pkt_time_us = min(
                random.expovariate(1 / self.avg_inter_packet_time_us),
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
