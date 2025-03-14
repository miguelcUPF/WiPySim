import os
import csv
from src.sim_config import ENABLE_TRAFFIC_GEN_RECORDING, TRAFFIC_GEN_RECORD_PATH
from src.utils.support import get_project_root, clean_folder


class TrafficRecorder:
    """Handles recording of network traffic events in Wireshark-style CSV format."""

    def __init__(self, node_id=None, save_name="traffic_trace", save_format="csv"):
        self.node_id = node_id
        self.save_name = save_name
        self.save_format = save_format

        self.filepath = None

    def is_enabled(self):
        return ENABLE_TRAFFIC_GEN_RECORDING

    def get_filepath(self, packet):
        if not self.filepath:
            save_folder = os.path.join(get_project_root(), TRAFFIC_GEN_RECORD_PATH)
            clean_folder(save_folder)
            self.filepath = os.path.join(save_folder, f"{self.save_name}_node_{self.node_id}_to_node_{packet.destination}.{self.save_format}")
        return self.filepath

    def record_packet(self, packet):
        if not self.is_enabled():
            return

        filepath = self.get_filepath(packet)
        write_header = not os.path.exists(filepath)

        with open(filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(
                    ["node.src", "node.dst", "frame.time_relative", "frame.len"])
            writer.writerow([
                packet.source,
                packet.destination,
                round(packet.creation_time_us * 1e-6, 6),  # Convert µs to seconds
                packet.size_bytes
            ])
