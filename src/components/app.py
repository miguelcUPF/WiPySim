import simpy
from src.traffic.generator import TrafficGenerator
from src.utils.traffic_recorder import TrafficRecorder

class APP:
    def __init__(self, env: simpy.Environment, node_id: int):
        self.env = env
        self.name = "APP"
        self.node_id = node_id
        
        self.mac_layer = None

        self.recorder = TrafficRecorder(self.node_id)

    def packet_to_mac(self, packet):
        """Receive a packet and forwards it to MAC."""
        self.recorder.record_packet(packet)
        self.mac_layer.tx_enqueue(packet)  # Send packet to MAC queue
        # TODO: record statistics
