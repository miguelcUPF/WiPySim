from src.sim_params import SimParams as sparams_module
from src.user_config import UserConfig as cfg_module

from src.components.network import Node
from src.traffic.recorder import TrafficRecorder
from src.utils.data_units import Packet

import simpy
import random

class APP:
    def __init__(self, cfg: cfg_module, sparams: sparams_module, env: simpy.Environment, node: Node):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.node = node

        self.recorder = TrafficRecorder(cfg, sparams, self.node.id)

        self.name = "APP"

        self.rng = random.Random(cfg.SEED)

    def packet_to_mac(self, packet: Packet):
        """Receives a packet from a traffic source and forwards it to MAC."""
        self.recorder.record_packet(packet)
        self.node.mac_layer.tx_enqueue(packet)

    def packet_from_mac(self, packet: Packet):
        packet.reception_time_us = self.env.now

        self.node.rx_stats.add_packet_to_history(packet)

        
