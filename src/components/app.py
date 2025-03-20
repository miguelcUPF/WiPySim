from src.components.network import Node
from src.traffic.recorder import TrafficRecorder
from src.utils.data_units import Packet

import simpy
import pandas as pd


class APP:
    def __init__(self, env: simpy.Environment, node: Node):
        self.env = env
        self.name = "APP"
        self.node = node

        self.recorder = TrafficRecorder(self.node.id)

    def packet_to_mac(self, packet: Packet):
        """Receives a packet from a traffic source and forwards it to MAC."""
        self.recorder.record_packet(packet)
        self.node.mac_layer.tx_enqueue(packet)

    def packet_from_mac(self, packet: Packet):
        packet.reception_time_us = self.env.now

        self.node.rx_stats.add_packet_to_history(packet)

        
