import os
import simpy
import pandas as pd
from src.utils.event_logger import get_logger
from src.utils.data_units import Packet
from src.components.network import Node
import chardet


class TrafficLoader:
    def __init__(self, env: simpy.Environment, node: Node, dst_id: int, **kwargs):
        self.env = env

        self.node = node

        self.src_id = node.id
        self.dst_id = dst_id

        self.filepath = kwargs.get("path", "")
        self.start_time_us = kwargs.get("start_time_us", 0)
        self.end_time_us = kwargs.get("end_time_us", None)

        self.packet_id = 0
        self.traffic_data = self._load_traffic_data(self.filepath)

        self.name = "LOAD"
        self.logger = get_logger(self.name, self.env)

        self.active_processes = []

        self.env.process(self._delayed_run()) if self.traffic_data is not None else None

    def _delayed_run(self):
        yield self.env.timeout(self.start_time_us)

        p = self.env.process(self.run())

        self.active_processes.append(p)

        if self.end_time_us:
            yield self.env.timeout(self.end_time_us - self.start_time_us)
            self.stop()

    def stop(self):
        """Stop running traffic loading processes."""
        for process in self.active_processes:
            if process.is_alive:
                process.interrupt()

        self.logger.debug(f"{self.node.type} {self.src_id} -> Traffic Loader stopped.")
        self.active_processes.clear()

    def _detect_encoding(self, filepath, sample_size=4096*2):
        with open(filepath, 'rb') as f:
            rawdata = f.read(sample_size)
        return chardet.detect(rawdata)["encoding"]

    def _load_traffic_data(self, filepath):
        """
        Loads a traffic file and normalizes column names.
        """
        if not os.path.exists(filepath):
            self.logger.error(
                f"Traffic file not found: {filepath}")
            return None
        try:
            encoding = self._detect_encoding(filepath)
            df = pd.read_csv(filepath, sep=None, engine="python", encoding=encoding, usecols=["frame.time_relative", "frame.len"], dtype={
                "frame.time_relative": float, "frame.len": int})
        except Exception as e:
            self.logger.error(f"{self.node.type} {self.src_id} -> Error reading traffic file {filepath}: {e}")
            return None

        column_mapping = {
            "frame.time_relative": "timestamp_s",
            "frame.len": "size_bytes",
        }
        df.rename(columns=column_mapping, inplace=True)

        if "timestamp_s" not in df or "size_bytes" not in df:
           self.logger.error(f"{self.node.type} {self.src_id} -> File {filepath} is missing required columns: timestamp_s, size_bytes")
           return None

        # Convert timestamps to microseconds
        df["timestamp_us"] = (df["timestamp_s"] * 1e6).astype(int)

        df["inter_arrival_us"] = df["timestamp_us"].diff().fillna(0).astype(int)

        if df.empty:
            self.logger.error(
                f"Traffic file is empty: {filepath}")

        return df.sort_values(by="timestamp_us")

    def run(self):
        try:
            for _, row in self.traffic_data.iterrows():
                yield self.env.timeout(max(0, row["inter_arrival_us"]))
                self._create_and_send_packet(row["size_bytes"])
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
            creation_time_us=self.env.now
        )
        self.logger.debug(f"{self.node.type} {self.src_id} -> Created {packet} from {self.filepath}")
        self.node.app_layer.packet_to_mac(packet)
