from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.utils.data_units import Packet
from src.utils.event_logger import get_logger

from typing import cast

import pandas as pd
import json
import os


class TransmissionStats:
    def __init__(self):
        self.first_tx_attempt_us = None
        self.last_tx_attempt_us = None

        self.tx_attempts = 0
        self.tx_successes = 0
        self.tx_failures = 0

        self.data_units_tx = 0  # Data units transmitted
        self.cts_tx = 0  # CTSs transmitted
        self.rts_tx = 0  # RTSs transmitted
        self.backs_tx = 0  # BACKs transmitted
        self.ampdus_tx = 0  # A-MPDUs transmitted

        self.pkts_tx = 0  # Packets transmitted

        self.pkts_success = 0  # Packets successfully transmitted according to BACK
        self.pkts_fail = 0  # Packets transmission failures according to BACK

        self.pkts_dropped_queue_lim = 0
        self.pkts_dropped_retry_lim = 0

        self.tx_app_bytes = 0  # Transmitted application data bytes
        self.tx_mac_bytes = 0  # Transmitted bytes (including MAC header)
        self.tx_phy_bytes = 0  # Transmitted bytes (including MAC and PHY header)

        self.airtime_us = 0
        self.medium_utilization = 0

        self.tx_queue_len_history = pd.DataFrame(
            columns=["timestamp_us", "queue_len"], data=[[0.0, 0]]
        )

    def add_to_tx_queue_history(self, timestamp_us: int, queue_len: int):
        new_row = pd.DataFrame([{"timestamp_us": timestamp_us, "queue_len": queue_len}])
        self.tx_queue_len_history = pd.concat(
            [self.tx_queue_len_history, new_row], ignore_index=True
        )
        new_row = {"timestamp_us": timestamp_us, "queue_len": queue_len}
        self.tx_queue_len_history.loc[len(self.tx_queue_len_history)] = new_row


class MACStateStats:
    def __init__(self):
        self.prev_mac_state_us = 0
        self.mac_states_history = pd.DataFrame(
            columns=["timestamp_us", "duration_us", "state"], data=[[0.0, 0.0, "IDLE"]]
        )

    def add_to_mac_state_history(self, timestamp_us: int, state: str):
        duration_us = timestamp_us - self.prev_mac_state_us
        new_row = pd.DataFrame(
            [{"timestamp_us": timestamp_us, "duration_us": duration_us, "state": state}]
        )
        self.mac_states_history = pd.concat(
            [self.mac_states_history, new_row], ignore_index=True
        )
        self.prev_mac_state_us = timestamp_us


class ReceptionStats:
    def __init__(self):
        self.first_rx_time_us = None
        self.last_rx_time_us = None

        self.data_units_rx = 0
        self.cts_rx = 0
        self.rts_rx = 0
        self.backs_rx = 0
        self.ampdu_rx = 0

        self.pkts_rx = 0  # Packets received (either corrupted or not)

        self.pkts_success = 0
        self.pkts_fail = 0

        self.rx_phy_bytes = 0  # Received bytes (including MAC and PHY header)
        self.rx_mac_bytes = 0  # Received bytes (including MAC header)
        self.rx_app_bytes = 0  # Received application data bytes

        self.rx_packets_history = pd.DataFrame(
            columns=[
                "packet_id",
                "src_id",
                "dst_id",
                "size_bytes",
                "creation_time_us",
                "reception_time_us",
                "retries",
                "type",
            ]
        )

    def add_packet_to_history(self, packet: Packet):
        new_row = pd.DataFrame(
            [
                {
                    "packet_id": packet.id,
                    "src_id": packet.src_id,
                    "dst_id": packet.dst_id,
                    "size_bytes": packet.size_bytes,
                    "creation_time_us": packet.creation_time_us,
                    "reception_time_us": packet.reception_time_us,
                    "retries": packet.retries,
                    "type": packet.type,
                }
            ]
        )
        if self.rx_packets_history.empty:
            self.rx_packets_history = new_row
        else:
            self.rx_packets_history = pd.concat(
                [self.rx_packets_history, new_row], ignore_index=True
            )


class ChannelStats:
    def __init__(self):
        self.airtime_us = 0  # Time the channel is actively used for transmission


class MediumStats:
    def __init__(self):
        self.ppdus_tx = 0

        self.ppdus_success = 0
        self.ppdus_fail = 0

        self.tx_bytes = 0
        self.tx_bytes_success = 0

        self.airtime_us = 0  # Time the medium is actively used for transmission


class NetworkStats:
    def __init__(self, cfg: cfg, sparams: sparams, network):
        from src.components.network import Network

        self.cfg = cfg
        self.sparams = sparams

        self.network = network
        self.network = cast(Network, self.network)

        # Global Network Stats
        self.total_tx_attempts = 0
        self.total_tx_successes = 0
        self.total_tx_failures = 0

        self.total_pkts_tx = 0
        self.total_pkts_rx = 0

        self.total_bytes_tx = 0
        self.total_bytes_rx = 0

        self.pkt_loss_ratio = 0

        self.avg_queue_len = 0

        self.avg_channel_airtime_us = 0
        self.avg_channel_utilization = 0

        # Per-node stats
        self.per_node_stats = {}

        # Per-channel stats
        self.per_channel_stats = {}

        # Medium Stats
        self.medium_stats = {}

        self.name = "STATS"
        self.logger = get_logger(self.name, cfg, sparams, self.network.env)

    def collect_stats(self):
        from src.components.network import Node

        """Aggregate statistics from all nodes and the network medium."""
        total_queue_len = 0

        for node in self.network.get_nodes():
            node = cast(Node, node)
            tx_stats = node.tx_stats
            rx_stats = node.rx_stats

            mac_state_stats = node.mac_layer.mac_state_stats
            mac_state_stats.add_to_mac_state_history(
                self.network.env.now, node.mac_layer.get_state_name()
            )

            self.total_tx_attempts += tx_stats.tx_attempts
            self.total_tx_successes += tx_stats.tx_successes
            self.total_tx_failures += tx_stats.tx_failures

            self.total_pkts_tx += tx_stats.pkts_tx
            self.total_pkts_rx += rx_stats.pkts_rx

            self.total_bytes_tx += tx_stats.tx_phy_bytes
            self.total_bytes_rx += rx_stats.rx_phy_bytes

            self.pkt_loss_ratio = (
                (self.total_pkts_tx - self.total_pkts_rx) / self.total_pkts_tx
                if self.total_pkts_tx > 0
                else 0
            )

            if not tx_stats.tx_queue_len_history.empty:
                mean_queue_len = tx_stats.tx_queue_len_history["queue_len"].mean()
                total_queue_len += mean_queue_len

            self.per_node_stats[node.id] = {
                "tx": {
                    "first_tx_attempt_us": tx_stats.first_tx_attempt_us,
                    "last_tx_attempt_us": tx_stats.last_tx_attempt_us,
                    "tx_attempts": tx_stats.tx_attempts,
                    "tx_successes": tx_stats.tx_successes,
                    "tx_failures": tx_stats.tx_failures,
                    "data_units_tx": tx_stats.data_units_tx,
                    "cts_tx": tx_stats.cts_tx,
                    "rts_tx": tx_stats.rts_tx,
                    "backs_tx": tx_stats.backs_tx,
                    "ampdus_tx": tx_stats.ampdus_tx,
                    "pkts_tx": tx_stats.pkts_tx,
                    "pkts_success": tx_stats.pkts_success,
                    "pkts_fail": tx_stats.pkts_fail,
                    "pkts_dropped_queue_lim": tx_stats.pkts_dropped_queue_lim,
                    "pkts_dropped_retry_lim": tx_stats.pkts_dropped_retry_lim,
                    "tx_app_bytes": tx_stats.tx_app_bytes,
                    "tx_mac_bytes": tx_stats.tx_mac_bytes,
                    "tx_phy_bytes": tx_stats.tx_phy_bytes,
                    "avg_queue_len": (
                        mean_queue_len if not tx_stats.tx_queue_len_history.empty else 0
                    ),
                    "tx_rate_pkts_per_sec": (
                        tx_stats.pkts_tx / (self.network.env.now / 1e6)
                        if self.network.env.now > 0
                        else 0
                    ),
                    "tx_rate_Mbits_per_sec": (
                        tx_stats.tx_phy_bytes * 8 / 1e6 / (self.network.env.now / 1e6)
                        if self.network.env.now > 0
                        else 0
                    ),
                    "airtime_us": tx_stats.airtime_us,
                    "utilization": (
                        tx_stats.airtime_us / self.network.env.now * 100
                        if self.network.env.now > 0
                        else 0
                    ),
                },
                "rx": {
                    "first_rx_time_us": rx_stats.first_rx_time_us,
                    "last_rx_time_us": rx_stats.last_rx_time_us,
                    "data_units_rx": rx_stats.data_units_rx,
                    "cts_rx": rx_stats.cts_rx,
                    "rts_rx": rx_stats.rts_rx,
                    "backs_rx": rx_stats.backs_rx,
                    "ampdus_rx": rx_stats.ampdu_rx,
                    "pkts_rx": rx_stats.pkts_rx,
                    "pkts_success": rx_stats.pkts_success,
                    "pkts_fail": rx_stats.pkts_fail,
                    "rx_app_bytes": rx_stats.rx_app_bytes,
                    "rx_mac_bytes": rx_stats.rx_mac_bytes,
                    "rx_phy_bytes": rx_stats.rx_phy_bytes,
                    "rx_rate_pkts_per_sec": (
                        rx_stats.pkts_rx / (self.network.env.now / 1e6)
                        if self.network.env.now > 0
                        else 0
                    ),
                    "rx_rate_Mbits_per_sec": (
                        rx_stats.rx_phy_bytes * 8 / 1e6 / (self.network.env.now / 1e6)
                        if self.network.env.now > 0
                        else 0
                    ),
                },
                "states": {
                    "idle_time_us": mac_state_stats.mac_states_history[
                        mac_state_stats.mac_states_history["state"] == "IDLE"
                    ]["duration_us"].sum(),
                    "contend_time_us": mac_state_stats.mac_states_history[
                        mac_state_stats.mac_states_history["state"] == "CONTEND"
                    ]["duration_us"].sum(),
                    "tx_time_us": mac_state_stats.mac_states_history[
                        mac_state_stats.mac_states_history["state"] == "TX"
                    ]["duration_us"].sum(),
                    "rx_time_us": mac_state_stats.mac_states_history[
                        mac_state_stats.mac_states_history["state"] == "RX"
                    ]["duration_us"].sum(),
                },
            }

        self.avg_queue_len = (
            total_queue_len / len(self.network.get_nodes())
            if len(self.network.get_nodes()) > 0
            else 0
        )

        # Per-Channel Statistics
        total_channel_airtime_us = 0
        for ch in self.network.medium.channels.values():
            ch.stats.airtime_us += (
                self.network.env.now - ch.busy_start_time
                if ch.busy_start_time is not None
                else 0
            )
            self.per_channel_stats[ch.id] = {
                "airtime_us": ch.stats.airtime_us,
                "utilization": (
                    (ch.stats.airtime_us / self.network.env.now * 100)
                    if self.network.env.now > 0
                    else 0
                ),
            }
            total_channel_airtime_us += ch.stats.airtime_us

        self.avg_channel_airtime_us = (
            total_channel_airtime_us / len(self.network.medium.channels)
            if len(self.network.medium.channels) > 0
            else 0
        )
        self.avg_channel_utilization = (
            self.avg_channel_airtime_us / self.network.env.now * 100
            if self.network.env.now > 0
            else 0
        )

        # Medium Statistics
        self.network.medium.stats.airtime_us += (
            self.network.env.now - self.network.medium.busy_start_time
            if self.network.medium.busy_start_time is not None
            else 0
        )
        medium_stats = self.network.medium.stats
        self.medium_stats = {
            "ppdus_tx": medium_stats.ppdus_tx,
            "ppdus_success": medium_stats.ppdus_success,
            "ppdus_fail": medium_stats.ppdus_fail,
            "tx_bytes": medium_stats.tx_bytes,
            "tx_bytes_success": medium_stats.tx_bytes_success,
            "tx_rate_Mbits_per_sec": (
                medium_stats.tx_bytes * 8 / 1e6 / (self.network.env.now / 1e6)
                if self.network.env.now > 0
                else 0
            ),
            "througput_Mbits_per_sec": (
                medium_stats.tx_bytes_success * 8 / 1e6 / (self.network.env.now / 1e6)
                if self.network.env.now > 0
                else 0
            ),
            "airtime_us": medium_stats.airtime_us,
            "utilization": (
                medium_stats.airtime_us / self.network.env.now * 100
                if self.network.env.now > 0
                else 0
            ),
        }

        if self.cfg.ENABLE_STATS_COLLECTION:
            self.save_stats()

    def save_stats(self):
        """Save statistics to JSON if enabled."""
        stats_data = {
            "global_stats": {
                "total_tx_attempts": self.total_tx_attempts,
                "total_tx_successes": self.total_tx_successes,
                "total_tx_failures": self.total_tx_failures,
                "total_pkts_tx": self.total_pkts_tx,
                "total_pkts_rx": self.total_pkts_rx,
                "total_bytes_tx": self.total_bytes_tx,
                "total_bytes_rx": self.total_bytes_rx,
                "pkt_loss_ratio": self.pkt_loss_ratio,
                "avg_queue_len": self.avg_queue_len,
                "avg_channel_airtime_us": self.avg_channel_airtime_us,
                "avg_channel_utilization": self.avg_channel_utilization,
            },
            "per_node_stats": self.per_node_stats,
            "per_channel_stats": self.per_channel_stats,
            "medium_stats": self.medium_stats,
        }

        os.makedirs(self.cfg.STATS_SAVE_PATH, exist_ok=True)

        filepath = os.path.join(self.cfg.STATS_SAVE_PATH, "session_stats.json")

        with open(filepath, "w") as f:
            json.dump(stats_data, f, indent=4)

        self.logger.info(f"Statistics saved to {filepath}")

    def display_stats(self):
        """Print a summary of network statistics."""
        print("\033[93m" + "Network Statistics Summary:" + "\033[0m")
        print(
            f"Total TX Attempts: {self.total_tx_attempts} ({self.total_tx_attempts - self.total_tx_successes - self.total_tx_failures} in progress)"
        )
        print(f"Total TX Successes: {self.total_tx_successes}")
        print(f"Total TX Failures: {self.total_tx_failures}")
        print(f"Total Packets Transmitted: {self.total_pkts_tx}")
        print(f"Total Packets Received: {self.total_pkts_rx}")
        print(f"Total Bytes Transmitted: {self.total_bytes_tx}")
        print(f"Total Bytes Received: {self.total_bytes_rx}")
        print(f"Packet Loss Ratio: {self.pkt_loss_ratio:.5%}")
        print(f"Average Queue Length: {self.avg_queue_len:.2f}")
        print(f"Average Channel Airtime: {self.avg_channel_airtime_us:.2f} µs")
        print(f"Average Channel Utilization: {self.avg_channel_utilization:.2f}%")

        print("\033[93m" + "\nPer-Node Stats:" + "\033[0m")
        for node_id, stats in self.per_node_stats.items():
            print(f"  {self.network.get_node(node_id).type} {node_id}:")
            print(
                f"    TX Attempts: {stats['tx']['tx_attempts']} ({stats['tx']['tx_attempts'] - stats['tx']['tx_successes'] - stats['tx']['tx_failures']} in progress)"
            )
            print(f"    TX Successes: {stats['tx']['tx_successes']}")
            print(f"    TX Failures: {stats['tx']['tx_failures']}")
            print(
                f"    Packets TX (Total): {stats['tx']['pkts_tx']}, "
                f"Packets TX (Success): {stats['tx']['pkts_success']}, "
                f"Packets TX (Fail): {stats['tx']['pkts_fail']}"
            )
            print(
                f"    Packets Dropped: {stats['tx']['pkts_dropped_queue_lim'] + stats['tx']['pkts_dropped_retry_lim']}"
            )
            print(
                f"    Packets RX (Total): {stats['rx']['pkts_rx']}, "
                f"Packets RX (Success): {stats['rx']['pkts_success']}, "
                f"Packets RX (Fail): {stats['rx']['pkts_fail']}"
            )
            print(f"    Queue Length Avg: {stats['tx']['avg_queue_len']:.2f}")
            print(
                f"    TX Rate (pkts/s): {stats['tx']['tx_rate_pkts_per_sec']:.2f}, "
                f"TX Rate (Mbps): {stats['tx']['tx_rate_Mbits_per_sec']:.2f}"
            )
            print(
                f"    RX Rate (pkts/s): {stats['rx']['rx_rate_pkts_per_sec']:.2f}, "
                f"RX Rate (Mbps): {stats['rx']['rx_rate_Mbits_per_sec']:.2f}"
            )
            print(
                f"    Airtime: {stats['states']['tx_time_us'] + stats['states']['rx_time_us']} µs"
            )
            print(
                f"    Utilization: {(stats['states']['tx_time_us'] + stats['states']['rx_time_us']) / self.network.medium.env.now * 100:.2f}%"
            )
            print(
                f"    IDLE Time: {stats['states']['idle_time_us']} µs ({stats['states']['idle_time_us'] / self.network.medium.env.now * 100:.2f}%)"
            )
            print(
                f"    CONTEND Time: {stats['states']['contend_time_us']} µs ({stats['states']['contend_time_us'] / self.network.medium.env.now * 100:.2f}%)"
            )
            print(
                f"    TX Time: {stats['states']['tx_time_us']} µs ({stats['states']['tx_time_us'] / self.network.medium.env.now * 100:.2f}%)"
            )
            print(
                f"    RX Time: {stats['states']['rx_time_us']} µs ({stats['states']['rx_time_us'] / self.network.medium.env.now * 100:.2f}%)"
            )

        print("\033[93m" + "\nPer-Channel Stats:" + "\033[0m")
        for ch_id, stats in self.per_channel_stats.items():
            print(f"  Channel {ch_id}:")
            print(f"    Airtime: {stats['airtime_us']} µs")
            print(f"    Utilization: {stats['utilization']:.2f}%")

        print("\033[93m" + "\nMedium Stats:" + "\033[0m")
        print(
            f"  PPDUs TX (Total): {self.medium_stats['ppdus_tx']}, "
            f"PPDUs TX (Success): {self.medium_stats['ppdus_success']}, "
            f"PPDUs TX (Fail): {self.medium_stats['ppdus_fail']}"
        )
        print(f"  Total Airtime: {self.medium_stats['airtime_us']} µs")
        print(f"  Utilization: {self.medium_stats['utilization']:.2f}%")
