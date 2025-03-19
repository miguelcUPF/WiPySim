import simpy
from src.traffic.recorder import TrafficRecorder

class APP:
    def __init__(self, env: simpy.Environment, node):
        self.env = env
        self.name = "APP"
        self.node = node
        
        self.recorder = TrafficRecorder(self.node.id)

    def packet_to_mac(self, packet):
        """Receive a packet and forwards it to MAC."""
        self.recorder.record_packet(packet)
        self.node.mac_layer.tx_enqueue(packet)  # Send packet to MAC queue
        # TODO: record statistics

    def stop(self):
        return

