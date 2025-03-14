import os
import simpy
import pandas as pd
from src.utils.event_logger import get_logger
from src.utils.data_units import Packet
import chardet


class TrafficLoader:
    def __init__(self, env: simpy.Environment, source: int, destination: int, file_path: str, app_layer):
        self.env = env
        self.name = "LOAD"
        self.source = source
        self.destination = destination
        self.file_path = file_path
        self.app_layer = app_layer
        self.packet_id = 0

        self.logger = get_logger(self.name, self.env)

        self.traffic_data = self._load_traffic_data(file_path)

        if self.traffic_data.empty:
            self.logger.warning(f"Traffic file {file_path} is empty!")

        self.env.process(self.run())

    def _detect_encoding(self, file_path, sample_size=4096*2):
        with open(file_path, 'rb') as f:
            rawdata = f.read(sample_size)
        return chardet.detect(rawdata)["encoding"]

    def _load_traffic_data(self, file_path):
        """
        Loads a traffic file and normalizes column names.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Traffic file not found: {file_path}")
        try:
            encoding = self._detect_encoding(file_path)
            df = pd.read_csv(file_path, sep=None, engine="python", encoding=encoding, usecols=["frame.time_relative", "frame.len"], dtype={
                "frame.time_relative": float, "frame.len": int})
        except Exception as e:
            raise ValueError(f"Error reading traffic file {file_path}: {e}")

        column_mapping = {
            "frame.time_relative": "timestamp_s",
            "frame.len": "size_bytes",
        }
        df.rename(columns=column_mapping, inplace=True)

        if "timestamp_s" not in df or "size_bytes" not in df:
            raise ValueError(f"File {file_path} is missing required columns!")

        # Convert timestamps to microseconds
        df["timestamp_us"] = (df["timestamp_s"] * 1e6).astype(int)

        df["inter_arrival_us"] = df["timestamp_us"].diff().fillna(0).astype(int)

        return df.sort_values(by="timestamp_us")

    def run(self):
        for _, row in self.traffic_data.iterrows():
            yield self.env.timeout(max(0, row["inter_arrival_us"]))
            self._create_and_send_packet(row["size_bytes"])

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
        self.logger.debug(f"Created {packet} from {self.file_path}")
        self.app_layer.packet_to_mac(packet)
