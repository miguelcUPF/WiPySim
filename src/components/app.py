from src.components.network import Node
from src.traffic.recorder import TrafficRecorder

import simpy


class APP:
    def __init__(self, env: simpy.Environment, node: Node):
        self.env = env
        self.name = "APP"
        self.node = node

        self.recorder = TrafficRecorder(self.node.id)

    def packet_to_mac(self, packet):
        """Receive a packet and forwards it to MAC."""
        self.recorder.record_packet(packet)
        self.node.mac_layer.tx_enqueue(packet)  # Send packet to MAC queue
