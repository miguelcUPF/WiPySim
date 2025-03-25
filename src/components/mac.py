from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.components.network import Node
from src.utils.data_units import Packet, AMPDU, MPDU, BACK, RTS, CTS, DataUnit
from src.utils.event_logger import get_logger
from src.utils.statistics import MACStateStats
from src.utils.mcs_table import calculate_data_rate_bps


from typing import cast
from simpy.events import AnyOf

import simpy
import random

BACK_TIMEOUT_us = 281
CTS_TX_us = round(
    (sparams.CTS_SIZE_bytes + sparams.PHY_HEADER_SIZE_bytes)
    * 8
    / calculate_data_rate_bps(0, 20, 1, sparams.GUARD_INTERVAL_us)
    * 1e6
)
CTS_TIMEOUT_us = sparams.SIFS_us + sparams.SLOT_TIME_us + CTS_TX_us


class MACState:
    IDLE = 0
    CONTEND = 1
    TX = 2
    RX = 3


class MAC:
    """
    MAC layer transmitter. Handles transmission logic including packet queuing, channel access (DCF), frame aggregation (A-MPDU) and retransmissions
    MAC layer receiver. Handles frame reception, ACK/BACK responses, and demultiplexing MPDUs.
    """

    def __init__(self, cfg: cfg, sparams: sparams, env: simpy.Environment, node: Node):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.node = node

        self.state = MACState.IDLE

        self.tx_queue: simpy.Store[MPDU] = simpy.Store(
            env, capacity=self.sparams.MAX_TX_QUEUE_SIZE_pkts
        )

        self.ampdu_counter = 0

        self.backoff_slots = 0
        self.retries = 0

        self.tx_ampdu: AMPDU = None
        self.rx_back: BACK = None

        self.tx_queue_event = None

        self.primary_busy_event = None
        self.primary_idle_event = None

        self.cts_event = None
        self.back_event = None

        self.rts_collision_event = self.env.event()
        self.ampdu_collision_event = self.env.event()

        self.last_collision_time_us = None

        self.is_first_tx = True
        self.is_first_rx = True

        self.mac_state_stats = MACStateStats()

        self.name = "MAC"
        self.logger = get_logger(self.name, cfg, sparams, env)

        self.env.process(self.run())

    def tx_enqueue(self, packet: Packet):
        """Enqueues a packet for transmission"""
        if len(self.tx_queue.items) >= self.sparams.MAX_TX_QUEUE_SIZE_pkts:
            self.node.tx_stats.pkts_dropped_queue_lim += 1

            self.logger.warning(
                f"{self.node.type} {self.node.id} -> Packet {packet.id} dropped due to full tx queue"
            )
        else:
            mpdu = MPDU(packet, self.env.now)
            self.tx_queue.put(mpdu)

            self.tx_queue_event.succeed() if self.tx_queue_event else None
            self.tx_queue_event = None

            self.node.tx_stats.add_to_tx_queue_history(
                self.env.now, len(self.tx_queue.items)
            )

            self.logger.debug(
                f"{self.node.type} {self.node.id} -> Packet {packet.id} added to tx queue (Queue length: {len(self.tx_queue.items)}, In transmission: {len(self.tx_ampdu.mpdus) if self.tx_ampdu else 0})"
            )

    def del_mpdu_from_queue(self, mpdu: MPDU):
        """Find and remove the specified MPDU from the tx_queue."""
        for i, item in enumerate(self.tx_queue.items):
            if item == mpdu:
                del self.tx_queue.items[i]
                return item
        return None

    def set_primary_busy(self):
        (
            self.primary_busy_event.succeed()
            if self.primary_busy_event and not self.primary_busy_event.triggered
            else None
        )

    def set_primary_idle(self):
        (
            self.primary_idle_event.succeed()
            if self.primary_idle_event and not self.primary_idle_event.triggered
            else None
        )

    def wait_for_idle(self, duration_us: float, waited_time_us: float = 0):
        """Check if all the used channels have been idle for the given duration."""
        self.set_state(MACState.CONTEND)
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Contending for primary channel {self.node.phy_layer.primary_channel_id} for {duration_us} μs ..."
        )

        self.primary_idle_event = self.env.event()

        if not self.node.phy_layer.is_primary_channel_idle():
            yield self.primary_idle_event
            self.primary_idle_event = None
            waited_time_us = 0

        idle_start_time = self.env.now - waited_time_us

        while True:
            remaining_time = duration_us - (self.env.now - idle_start_time)

            self.primary_busy_event = self.env.event()

            events = [self.env.timeout(remaining_time), self.primary_busy_event]

            if self.rts_collision_event is not None:
                events.append(self.rts_collision_event)
            if self.ampdu_collision_event is not None:
                events.append(self.ampdu_collision_event)

            yield AnyOf(self.env, events)

            
            if (
                self.ampdu_collision_event is not None
                and self.ampdu_collision_event.triggered
            ):
                self.ampdu_collision_event = self.env.event()
                self.rts_collision_event = self.env.event()
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> AMPDU collision detected, waiting for EIFS ({self.sparams.DIFS_us + BACK_TIMEOUT_us} us)"
                )
                yield from self.wait_for_idle(self.sparams.DIFS_us + BACK_TIMEOUT_us)
                return
            elif (
                self.rts_collision_event is not None
                and self.rts_collision_event.triggered
            ):
                self.ampdu_collision_event = self.env.event()
                self.rts_collision_event = self.env.event()
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> RTS collision detected, waiting for EIFS ({self.sparams.DIFS_us + CTS_TIMEOUT_us} us)"
                )
                yield from self.wait_for_idle(self.sparams.DIFS_us + CTS_TIMEOUT_us)
                return


            if self.primary_busy_event.triggered:
                self.primary_busy_event = None

                self.primary_idle_event = self.env.event()

                # Reset timer and wait for the channel to become idle again
                yield self.primary_idle_event

                self.primary_idle_event = None
                idle_start_time = self.env.now  # Restart idle tracking
            else:
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> Primary channel {self.node.phy_layer.primary_channel_id} has been idle for {duration_us} μs"
                )
                return

    def backoff(self):
        self.logger.header(f"{self.node.type} {self.node.id} -> Starting Backoff...")

        if self.backoff_slots > 0:
            self.logger.debug(
                f"{self.node.type} {self.node.id} -> Backoff slots already set ({self.backoff_slots})"
            )
        else:
            cw = min(self.sparams.CW_MIN * (2**self.retries), self.sparams.CW_MAX)
            self.backoff_slots = random.randint(0, max(0, cw - 1)) + 1
            self.logger.info(
                f"{self.node.type} {self.node.id} -> Backoff slots: {self.backoff_slots} (retries: {self.retries})"
            )

        while self.backoff_slots > 0:
            self.primary_busy_event = self.env.event()

            slot_start_time = self.env.now

            event_result = (
                yield self.env.timeout(self.sparams.SLOT_TIME_us)
                | self.primary_busy_event
            )

            if (
                self.primary_busy_event in event_result
                and self.env.now < slot_start_time + self.sparams.SLOT_TIME_us
            ):
                self.primary_busy_event = None
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> Channel busy, pausing backoff ({self.backoff_slots})..."
                )
                return

            self.backoff_slots -= 1
            self.logger.debug(
                f"{self.node.type} {self.node.id} -> Backoff slots reduced ({self.backoff_slots})"
            )

        if self.is_first_tx:
            self.node.tx_stats.first_tx_attempt_us = self.env.now
            self.is_first_tx = False
        else:
            self.node.tx_stats.last_tx_attempt_us = self.env.now

        self.node.tx_stats.tx_attempts += 1

        self.logger.header(f"{self.node.type} {self.node.id} -> Backoff finished")

    def ampdu_aggregation(self):
        """Aggregates MDPUS into an A-MPDU frame."""
        if self.tx_ampdu:
            if self.tx_ampdu.retries >= self.sparams.COMMON_RETRY_LIMIT:
                self.logger.info(
                    f"{self.node.type} {self.node.id} -> A-MPDU {self.tx_ampdu.id} dropped after max retries"
                )
                self.tx_ampdu = None
            elif (
                self.tx_ampdu.retries == 0
            ):  # If 0 it means that CTS timeout occurred, so no need to aggregate
                return
            else:
                self.logger.info(
                    f"{self.node.type} {self.node.id} -> No need to aggregate, retransmitting A-MPDU {self.tx_ampdu.id} (retries={self.tx_ampdu.retries})..."
                )
                return

        self.logger.header(f"{self.node.type} {self.node.id} -> Aggregating MPDUs...")

        agg_mpdus = []
        total_size = 0

        first_mpdu = cast(MPDU, self.tx_queue.items[0])
        destination = first_mpdu.dst_id

        for item in self.tx_queue.items[:]:
            mpdu = cast(MPDU, item)
            if mpdu.dst_id != destination:
                continue

            if total_size + mpdu.size_bytes > self.sparams.MAX_AMPDU_SIZE_bytes:
                break

            agg_mpdus.append(mpdu)
            total_size += mpdu.size_bytes

        if not agg_mpdus:
            return

        self.ampdu_counter += 1
        self.tx_ampdu = AMPDU(self.ampdu_counter, agg_mpdus, self.env.now)

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Created AMPDU {self.tx_ampdu.id} with {len(agg_mpdus)} MPDUs and size {total_size} bytes"
        )

    def rts_cts(self):
        """Handles RTS/CTS exchange before data transmission."""
        rts = RTS(self.tx_ampdu.src_id, self.tx_ampdu.dst_id, self.env.now)

        self.node.tx_stats.rts_tx += 1

        yield self.env.process(self.transmit(rts))
        yield self.env.process(self.wait_for_cts())

    def wait_for_cts(self):
        self.set_state(MACState.RX)

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Waiting for CTS from {self.tx_ampdu.dst_id}..."
        )

        self.cts_event = self.env.event()

        event_result = yield self.env.timeout(CTS_TIMEOUT_us) | self.cts_event

        if not self.cts_event in event_result:
            self.logger.warning(f"{self.node.type} {self.node.id} -> CTS timeout...")
            self.cts_event = None
            self.retries += 1
            if self.sparams.ENABLE_RTS_CTS:
                self.node.tx_stats.tx_failures += 1

            return
        else:
            self.retries = 0
            yield self.env.process(self.wait_for_idle(self.sparams.SIFS_us))
            yield self.env.process(self.transmit_ampdu())

    def transmit_ampdu(self):
        self.node.tx_stats.ampdus_tx += 1

        yield self.env.process(self.transmit(self.tx_ampdu))
        yield self.env.process(self.wait_for_back())

    def wait_for_back(self):
        self.set_state(MACState.RX)
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Waiting for BACK from {self.tx_ampdu.dst_id}..."
        )
        self.back_event = self.env.event()

        yield self.env.timeout(BACK_TIMEOUT_us) | self.back_event

        if self.sparams.ENABLE_RTS_CTS:
            self.node.phy_layer.end_nav()

        if not self.back_event.triggered:
            self.logger.warning(f"{self.node.type} {self.node.id} -> BACK timeout...")
            self.back_event = None

            self.tx_ampdu.retries += 1

            self.retries += 1
            if not self.sparams.ENABLE_RTS_CTS:
                self.node.tx_stats.tx_failures += 1

            return
        else:
            sent_mpdus = self.tx_ampdu.mpdus

            lost_mpdus = self.rx_back.lost_mpdus

            rx_mpdus = set(sent_mpdus) - set(lost_mpdus)

            self.node.tx_stats.pkts_tx += len(sent_mpdus)

            for mpdu in sent_mpdus:
                self.node.tx_stats.tx_app_bytes += mpdu.packet.size_bytes

            self.node.tx_stats.pkts_success += len(rx_mpdus)

            self.logger.info(
                f"{self.node.type} {self.node.id} -> {len(rx_mpdus)} Packets succesfully transmitted according to BACK"
            )

            if lost_mpdus:
                self.node.tx_stats.pkts_fail += len(lost_mpdus)

                self.logger.warning(
                    f"{self.node.type} {self.node.id} -> {len(lost_mpdus)} Packet transmission failures according to BACK"
                )

            for mpdu in lost_mpdus:
                mpdu.retries += 1
                mpdu.is_corrupted = False
                if mpdu.retries >= self.sparams.COMMON_RETRY_LIMIT:
                    self.node.tx_stats.pkts_dropped_retry_lim += 1

                    self.del_mpdu_from_queue(mpdu)

                    self.logger.warning(
                        f"{self.node.type} {self.node.id} -> MPDU {mpdu.packet.id} dropped after max retries"
                    )

            for mpdu in sent_mpdus:
                if mpdu not in lost_mpdus:
                    self.del_mpdu_from_queue(mpdu)

            self.node.tx_stats.add_to_tx_queue_history(
                self.env.now, len(self.tx_queue.items)
            )

            self.tx_ampdu = None
            self.retries = 0

    def transmit(self, data_unit: DataUnit):
        self.set_state(MACState.TX)

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Sending {data_unit.type} from MAC to PHY..."
        )

        self.node.tx_stats.data_units_tx += 1
        self.node.tx_stats.tx_mac_bytes += data_unit.size_bytes

        start_tx_time_us = self.env.now
        yield self.env.process(self.node.phy_layer.transmit(data_unit))
        self.node.tx_stats.airtime_us += self.env.now - start_tx_time_us

    def rts_collision_detected(self):
        self.last_collision_time_us = self.env.now
        (
            self.rts_collision_event.succeed()
            if self.rts_collision_event and not self.rts_collision_event.triggered
            else None
        )

    def ampdu_collision_detected(self):
        self.last_collision_time_us = self.env.now
        (
            self.ampdu_collision_event.succeed()
            if self.rts_collision_event and not self.ampdu_collision_event.triggered
            else None
        )

    def successful_transmission_detected(self):
        self.rts_collision_event = self.env.event()
        self.ampdu_collision_event = self.env.event()

    def receive(self, data_unit: DataUnit):

        if self.is_first_rx:
            self.node.rx_stats.first_rx_time_us = self.env.now
            self.is_first_rx = False
        else:
            self.node.rx_stats.last_rx_time_us = self.env.now

        self.node.rx_stats.data_units_rx += 1
        self.node.rx_stats.rx_mac_bytes += data_unit.size_bytes

        if data_unit.dst_id != self.node.id:
            self.logger.warning(
                f"{self.node.type} {self.node.id} -> Received {data_unit.type} from node {data_unit.src_id} not for me (dst={data_unit.dst_id})"
            )
            return

        data_unit.reception_time_us = self.env.now

        self.logger.success(
            f"{self.node.type} {self.node.id} -> Received {data_unit.type}{f' {data_unit.id}' if data_unit.type == 'AMPDU' else ''} from node {data_unit.src_id}"
        )

        if data_unit.type == "AMPDU":

            back = BACK(data_unit, self.node.id, data_unit.src_id, self.env.now)

            for mpdu in data_unit.mpdus:
                self.node.rx_stats.pkts_rx += 1

                if mpdu.is_corrupted:
                    self.node.rx_stats.pkts_fail += 1
                    back.add_lost_mpdus(mpdu)
                else:
                    self.node.rx_stats.pkts_success += 1
                    self.node.rx_stats.rx_app_bytes += mpdu.packet.size_bytes
                    self.node.app_layer.packet_from_mac(mpdu.packet)

            self.node.tx_stats.backs_tx += 1

            self.env.process(self.send_response(back))

        elif data_unit.type == "RTS":
            self.node.rx_stats.rts_rx += 1

            if self.state == MACState.IDLE:
                cts = CTS(data_unit.dst_id, data_unit.src_id, self.env.now)

                self.env.process(self.send_response(cts))

                self.node.tx_stats.cts_tx += 1
            else:
                self.logger.warning(
                    f"{self.node.type} {self.node.id} -> Ignoring RTS received from {data_unit.src_id} (busy) "
                )

        elif data_unit.type == "CTS":
            self.node.rx_stats.cts_rx += 1

            if self.cts_event and not self.cts_event.triggered:
                if self.sparams.ENABLE_RTS_CTS:
                    self.node.tx_stats.tx_successes += 1

                self.cts_event.succeed()  # Trigger CTS event
            else:
                self.logger.warning(
                    f"{self.node.type} {self.node.id} -> Ignoring CTS received from {data_unit.src_id} (outdated) "
                )

        elif data_unit.type == "BACK":
            self.node.rx_stats.backs_rx += 1

            if self.back_event and not self.back_event.triggered:
                self.rx_back = data_unit  # Store received BACK frame
                if not self.sparams.ENABLE_RTS_CTS:
                    self.node.tx_stats.tx_successes += 1

                self.back_event.succeed()  # Trigger BACK event

                self.set_state(MACState.IDLE)
            else:
                self.logger.warning(
                    f"{self.node.type} {self.node.id} -> Ignoring BACK received from {data_unit.src_id} (outdated) "
                )

    def send_response(self, data_unit):
        yield self.env.process(self.wait_for_idle(self.sparams.SIFS_us))
        yield self.env.process(self.transmit(data_unit))
        self.set_state(MACState.IDLE)

    def set_state(self, new_state: MACState):
        """Updates the MAC state."""
        if self.state != new_state:
            self.state = new_state

            state_name = [
                name for name, value in vars(MACState).items() if value == self.state
            ][0]

            self.mac_state_stats.add_to_mac_state_history(self.env.now, state_name)

            self.logger.info(
                f"{self.node.type} {self.node.id} -> MAC state: {self.state} ({state_name})"
            )

    def run(self):
        """Handles channel access, contention, and transmission"""
        while True:
            if len(self.tx_queue.items) > 0:
                self.rts_collision_event = self.env.event()
                if self.ampdu_collision_event and self.ampdu_collision_event.triggered:
                    self.ampdu_collision_event = self.env.event()
                    self.logger.debug(
                        f"{self.node.type} {self.node.id} -> AMPDU collision detected, waiting for EIFS ({self.sparams.DIFS_us + BACK_TIMEOUT_us} us), already waited {self.env.now - self.last_collision_time_us} us"
                    )
                    yield self.env.process(
                        self.wait_for_idle(
                            self.sparams.DIFS_us + BACK_TIMEOUT_us,
                            waited_time_us=self.env.now - self.last_collision_time_us,
                        )
                    )
                else:
                    yield self.env.process(self.wait_for_idle(self.sparams.DIFS_us))
                yield self.env.process(self.backoff())

                if self.backoff_slots > 0:
                    continue

                self.ampdu_aggregation()

                if (
                    self.sparams.ENABLE_RTS_CTS
                    and self.tx_ampdu.size_bytes > self.sparams.RTS_THRESHOLD_bytes
                ):
                    yield self.env.process(self.rts_cts())
                else:
                    if self.tx_ampdu.size_bytes <= self.sparams.RTS_THRESHOLD_bytes:
                        self.logger.info(
                            f"{self.node.type} {self.node.id} -> No need to send RTS/CTS for AMPDU {self.tx_ampdu.id} (size={self.tx_ampdu.size_bytes} bytes < {self.sparams.RTS_THRESHOLD_bytes} bytes)"
                        )
                    yield self.env.process(self.transmit_ampdu())
            else:
                self.tx_queue_event = self.env.event()
                yield self.tx_queue_event
