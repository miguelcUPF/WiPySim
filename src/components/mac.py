from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.components.network import Node
from src.utils.data_units import Packet, AMPDU, MPDU, BACK, RTS, CTS, DataUnit
from src.utils.event_logger import get_logger
from src.utils.statistics import MACStateStats
from src.utils.mcs_table import calculate_data_rate_bps


from typing import cast
from simpy.events import AnyOf, AllOf

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

        self.any_channel_busy_event = None
        self.all_channels_idle_event = None

        self.cts_event = None
        self.back_event = None

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

    def trigger_any_busy_event(self):
        (
            self.any_channel_busy_event.succeed()
            if self.any_channel_busy_event and not self.any_channel_busy_event.triggered
            else None
        )

    def trigger_all_idle_event(self):
        (
            self.all_channels_idle_event.succeed()
            if self.all_channels_idle_event
            and not self.all_channels_idle_event.triggered
            else None
        )

    def _wait_until_channel_idle(
        self, ch_id: int, duration_us: float, waited_time_us: float = 0
    ):
        wait_start_time = self.env.now - waited_time_us
        while True:
            self.node.phy_layer.reset_busy_event(ch_id)
            self.node.phy_layer.reset_idle_event(ch_id)

            # Wait for channel to become idle
            if not self.node.phy_layer.is_channel_idle(ch_id):
                channel_idle_event = self.node.phy_layer.get_idle_event(ch_id)
                yield channel_idle_event
                wait_start_time = self.env.now

            remaining_time = duration_us - (self.env.now - wait_start_time)

            # Channel is now idle, start timing
            timeout = self.env.timeout(remaining_time)
            busy_event = self.node.phy_layer.get_busy_event(ch_id)

            # Use per-channel collision events
            rts_event = self.node.phy_layer.get_rts_collision_event(ch_id)
            ampdu_event = self.node.phy_layer.get_ampdu_collision_event(ch_id)

            events = [timeout, busy_event, rts_event, ampdu_event]

            event_result = yield AnyOf(self.env, events)

            if ampdu_event in event_result:
                eifs = self.sparams.DIFS_us + BACK_TIMEOUT_us
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> AMPDU collision on Channel {ch_id}, waiting EIFS ({eifs} μs)"
                )
                self.node.phy_layer.reset_ampdu_collision_event(ch_id)
                # reset rts collision event since it might happen that both rts and ampdu collision occur on the same channel, and the ampdu collision EIFS should remain as it is longer
                self.node.phy_layer.reset_rts_collision_event(ch_id)
                yield from self._wait_until_channel_idle(ch_id, eifs)
                return

            if rts_event in event_result:
                eifs = self.sparams.DIFS_us + CTS_TIMEOUT_us
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> RTS collision on Channel {ch_id}, waiting EIFS ({eifs} μs)"
                )
                self.node.phy_layer.reset_rts_collision_event(ch_id)
                yield from self._wait_until_channel_idle(ch_id, eifs)
                return

            if timeout in event_result:
                # Successfully stayed idle for duration
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> Channel {ch_id} has been idle for {duration_us} μs"
                )
                return
            # else: it became busy before the timer finished, retry

    def wait_until_primary_idle(
        self, ch_duration_us: dict | int, ch_waited_time_us: dict | int = 0
    ):
        self.set_state(MACState.CONTEND)

        if len(self.node.phy_layer.sensing_channels_ids) > 1:
            self.logger.critical(
                f"{self.node.type} {self.node.id} -> Sensing channels does not contain only one channel when using CSMA_SENSING_MODE 0! (Channels: {', '.join(map(str, self.node.phy_layer.sensing_channels_ids))})"
            )

        primary_channel_id = next(iter(self.node.phy_layer.sensing_channels_ids))

        duration_us = (
            ch_duration_us
            if isinstance(ch_duration_us, int)
            else ch_duration_us[primary_channel_id]
        )
        waited_time_us = (
            ch_waited_time_us
            if isinstance(ch_waited_time_us, int)
            else ch_waited_time_us[primary_channel_id]
        )

        self.logger.header(
            f"{self.node.type} {self.node.id} -> Sensing if primary channel (Channel: {primary_channel_id}) is idle for {duration_us} μs..."
        )

        if not self.node.phy_layer.are_all_sensing_channels_idle():
            self.all_channels_idle_event = self.env.event()
            yield self.all_channels_idle_event
            self.all_channels_idle_event = None
            waited_time_us = 0
        yield from self._wait_until_channel_idle(
            primary_channel_id, duration_us, waited_time_us
        )

    def wait_until_any_idle(
        self, ch_duration_us: dict | int, ch_waited_time_us: dict | int = 0
    ):
        self.set_state(MACState.CONTEND)
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Sensing if any channel (Channels: {', '.join(map(str, self.node.phy_layer.sensing_channels_ids))}) are idle for {str(ch_duration_us) + 'μs' if isinstance(ch_duration_us, float) else ', '.join(map(str, ch_duration_us.values())) + 'μs respectively'}......"
        )

        sensing_channels = self.node.phy_layer.sensing_channels_ids
        idle_channel_events = [
            self.env.process(
                self._wait_until_channel_idle(
                    ch_id,
                    (
                        ch_duration_us
                        if isinstance(ch_duration_us, int)
                        else ch_duration_us[ch_id]
                    ),
                    (
                        ch_waited_time_us
                        if isinstance(ch_waited_time_us, int)
                        else ch_waited_time_us[ch_id]
                    ),
                )
            )
            for ch_id in sensing_channels
        ]

        # Wait until at least one channel completes its idle time
        event_result = yield AnyOf(self.env, idle_channel_events)

        # Get list of channels that completed the idle period
        idle_channels = [
            sensing_channels[i]
            for i, event in enumerate(idle_channel_events)
            if event in event_result
        ]

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Channels {', '.join(map(str, idle_channels))} have been idle for {str(ch_duration_us) + 'μs' if isinstance(ch_duration_us, float) else ', '.join(map(str, [ch_duration_us[ch_id] for ch_id in idle_channels])) + 'μs respectively'}"
        )

        return idle_channels

    def backoff(self, channels_ids: set[int] = None):
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Starting Backoff{'' if channels_ids is None else f' on channels'} {', '.join(map(str, self.node.phy_layer.sensing_channels_ids)) if channels_ids is not None else ''}..."
        )

        if self.backoff_slots > 0:
            self.logger.debug(
                f"{self.node.type} {self.node.id} -> Backoff slots already set ({self.backoff_slots})"
            )
        else:
            cw = min(self.sparams.CW_MIN * (2**self.retries), self.sparams.CW_MAX)
            self.backoff_slots = random.randint(0, max(0, cw - 1))
            if not self.is_first_tx:
                self.backoff_slots += 1
            self.logger.info(
                f"{self.node.type} {self.node.id} -> Backoff slots: {self.backoff_slots} (retries: {self.retries})"
            )

        bo_channels = set(channels_ids) if channels_ids else set()

        slot_remaining_time = self.sparams.SLOT_TIME_us

        while self.backoff_slots > 0:
            if self.sparams.CSMA_SENSING_MODE == 0:
                # legacy behavior
                self.any_channel_busy_event = self.env.event()
                event = self.any_channel_busy_event

                slot_start_time = self.env.now

                event_result = yield self.env.timeout(self.sparams.SLOT_TIME_us) | event

                if (
                    event in event_result
                    and self.env.now < slot_start_time + self.sparams.SLOT_TIME_us
                ):
                    self.any_channel_busy_event = None
                    self.logger.debug(
                        f"{self.node.type} {self.node.id} -> Channel busy, pausing backoff ({self.backoff_slots})..."
                    )
                    return

                self.backoff_slots -= 1
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> Backoff slots reduced ({self.backoff_slots})"
                )
            else:
                # special mode
                busy_events = {
                    ch_id: self.node.phy_layer.get_busy_event(ch_id)
                    for ch_id in bo_channels
                }
                event = self.env.any_of(list(busy_events.values()))

                slot_start_time = self.env.now
                event_result = yield self.env.timeout(slot_remaining_time) | event

                if (
                    event in event_result
                    and self.env.now < slot_start_time + slot_remaining_time
                ):
                    # Check which channels got busy
                    busy_channels = {
                        ch_id for ch_id, ev in busy_events.items() if ev.triggered
                    }
                    slot_remaining_time = self.env.now - slot_start_time
                    if busy_channels:
                        # Remove only the busy ones
                        bo_channels -= busy_channels
                        self.logger.debug(
                            f"{self.node.type} {self.node.id} -> Channels {', '.join(map(str, busy_channels))} became busy. Remaining idle: {', '.join(map(str, bo_channels))}"
                        )
                        if not bo_channels:
                            self.logger.debug(
                                f"{self.node.type} {self.node.id} -> All channels are busy, pausing backoff ({self.backoff_slots})..."
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

        if self.sparams.CSMA_SENSING_MODE == 1:
            self.node.phy_layer.set_transmitting_channels(bo_channels)

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
                self.retries = 0

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
            yield self.env.process(self.wait_until_primary_idle(self.sparams.SIFS_us))
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
        yield self.env.process(self.wait_until_primary_idle(self.sparams.SIFS_us))
        yield self.env.process(self.transmit(data_unit))
        self.set_state(MACState.IDLE)

    def get_state_name(self):
        return [name for name, value in vars(MACState).items() if value == self.state][
            0
        ]

    def set_state(self, new_state: MACState):
        """Updates the MAC state."""
        if self.state != new_state:
            self.state = new_state

            state_name = self.get_state_name()

            self.mac_state_stats.add_to_mac_state_history(self.env.now, state_name)

            self.logger.info(
                f"{self.node.type} {self.node.id} -> MAC state: {self.state} ({state_name})"
            )

    def run(self):
        """Handles channel access, contention, and transmission"""
        while True:
            if len(self.tx_queue.items) > 0:
                ch_duration_us = {}
                ch_waited_times = {}

                for ch_id in self.node.phy_layer.sensing_channels_ids:
                    ch_duration_us[ch_id] = self.sparams.DIFS_us
                    ch_waited_times[ch_id] = 0

                # If an AMDPU/RTS collision occured while not contending it should be sensed during EIFS
                # This happens when an RTS and AMPDU (sent due to size smaller than RTS_THRESHOLD_SIZE) collide
                # The RTS sender does not count the channel as idle during the time elapsed from the delivery of the AMPDU until its CTS timeout occurs.
                for ch_id in self.node.phy_layer.get_ampdu_collisions_channels_ids():
                    waited_time_us = (
                        self.env.now
                        - self.node.phy_layer.get_last_collision_time(ch_id)
                    )
                    duration_us = self.sparams.DIFS_us + BACK_TIMEOUT_us
                    ch_waited_times[ch_id] = waited_time_us
                    ch_duration_us[ch_id] = duration_us
                    self.logger.debug(
                        f"{self.node.type} {self.node.id} -> AMPDU collision on ch {ch_id}, waiting EIFS ({duration_us} us), already waited {waited_time} us"
                    )
                else:
                    waited_time = 0
                    duration_us = self.sparams.DIFS_us

                self.node.phy_layer.reset_collision_events()

                if self.sparams.CSMA_SENSING_MODE == 1:
                    idle_channels = yield self.env.process(
                        self.wait_until_any_idle(ch_duration_us, ch_waited_times)
                    )
                    yield self.env.process(self.backoff(idle_channels))
                else:
                    # legacy behavior
                    yield self.env.process(
                        self.wait_until_primary_idle(ch_duration_us, ch_waited_times)
                    )
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
