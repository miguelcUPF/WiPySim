from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.components.network import Node
from src.components.rl_agents import (
    MARLController,
    SARLController,
    CHANNEL_MAP,
    PRIMARY_CHANNEL_MAP,
    CW_MAP,
)
from src.utils.data_units import Packet, AMPDU, MPDU, BACK, RTS, CTS, DataUnit
from src.utils.event_logger import get_logger
from src.utils.statistics import MACStateStats
from src.utils.mcs_table import calculate_data_rate_bps


from typing import cast
from simpy.events import AnyOf

import simpy
import random
import numpy as np
import wandb
import math


BACK_TIMEOUT_us = 281
CTS_TX_us = round(
    (sparams.CTS_SIZE_bytes + sparams.PHY_HEADER_SIZE_bytes)
    * 8
    / calculate_data_rate_bps(0, 20, 1, 0.8)
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

    def __init__(
        self,
        cfg: cfg,
        sparams: sparams,
        env: simpy.Environment,
        node: Node,
        rl_driven: bool = False,
    ):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

        self.node = node

        self.rl_driven = rl_driven

        self.state = MACState.IDLE

        self.tx_queue: simpy.Store[MPDU] = simpy.Store(
            env, capacity=self.sparams.MAX_TX_QUEUE_SIZE_pkts
        )
        self.non_full_event = None

        self.ampdu_counter = 0

        self.backoff_slots = 0
        self.retries = 0

        self.tx_ampdu: AMPDU = None
        self.rx_back: BACK = None

        self.tx_queue_event = None

        self.cts_event = None
        self.back_event = None

        self.cts_timedout = False

        self.last_collision_time_us = None
        self.prev_rx_time_us = self.env.now
        self.ema_goodput_mbps = 0

        self.is_first_tx = True
        self.is_first_rx = True

        self.tx_counter = 0

        self.tx_attempt_time_us = None
        self.sensing_start_time_us = None
        self.bo_start_time_us = None
        self.tx_start_time_us = None
        self.sensing_duration_us = 0
        self.bo_duration_us = 0
        self.tx_duration_us = 0

        self.cw_current = 8  # lazy init

        self.mac_state_stats = MACStateStats()

        self.name = "MAC"
        self.logger = get_logger(
            self.name,
            cfg,
            sparams,
            env,
            True if node.id in self.cfg.EXCLUDED_IDS else False,
        )

        self.rl_settings = self.cfg.AGENTS_SETTINGS
        self.rl_controller = None
        self.rl_mode = self.cfg.RL_MODE

        if self.rl_driven and self.sparams.BONDING_MODE == 0:
            if self.cfg.RL_MODE == 0:
                self.rl_controller = SARLController(
                    sparams, cfg, env, self.node, self.rl_settings
                )
            else:
                self.rl_controller = MARLController(
                    sparams, cfg, env, self.node, self.rl_settings
                )

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

            self.logger.ignore(
                f"{self.node.type} {self.node.id} -> Packet {packet.id} added to tx queue (Queue length: {len(self.tx_queue.items)}, In transmission: {len(self.tx_ampdu.mpdus) if self.tx_ampdu else 0})"
            )

    def del_mpdu_from_queue(self, mpdu: MPDU):
        """Find and remove the specified MPDU from the tx_queue."""
        for i, item in enumerate(self.tx_queue.items):
            if item == mpdu:
                del self.tx_queue.items[i]
                return item
        return None

    def _wait_until_channel_idle(
        self,
        ch_id: int,
        duration_us: float,
        waited_time_us: float = 0,
        waiting_eifs=False,
    ):
        # Wait for channel to become idle
        if not self.node.phy_layer.is_channel_idle(ch_id):
            channel_idle_event = self.node.phy_layer.get_idle_event(ch_id)
            yield channel_idle_event
            waited_time_us = 0

        if (
            waiting_eifs
            and self.node.phy_layer.get_successful_tx_event(ch_id).triggered
        ):  # if sensing for EIFS but last tx was successful it should be sensed for DIFS
            yield from self._wait_until_channel_idle(
                ch_id, self.sparams.DIFS_us, waited_time_us
            )
            return

        wait_start_time = self.env.now - waited_time_us
        while True:
            self.node.phy_layer.reset_busy_event(ch_id)
            self.node.phy_layer.reset_idle_event(ch_id)

            # Wait for channel to become idle
            if not self.node.phy_layer.is_channel_idle(ch_id):
                channel_idle_event = self.node.phy_layer.get_idle_event(ch_id)
                yield channel_idle_event
                wait_start_time = self.env.now

                if (
                    waiting_eifs
                    and self.node.phy_layer.get_successful_tx_event(ch_id).triggered
                ):  # if sensing for EIFS but last tx was successful it should be sensed for DIFS
                    yield from self._wait_until_channel_idle(
                        ch_id, self.sparams.DIFS_us, 0
                    )
                    return

            remaining_time = duration_us - (self.env.now - wait_start_time)

            # Channel is now idle, start timing
            timeout = self.env.timeout(remaining_time)
            busy_event = self.node.phy_layer.get_busy_event(ch_id)

            # Use per-channel collision events
            rts_event = self.node.phy_layer.get_rts_collision_event(ch_id)
            ampdu_event = self.node.phy_layer.get_ampdu_collision_event(ch_id)

            events = [timeout, busy_event, rts_event, ampdu_event]

            yield AnyOf(self.env, events)

            if ampdu_event.triggered:
                eifs = self.sparams.DIFS_us + BACK_TIMEOUT_us
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> AMPDU collision on Channel {ch_id}, waiting EIFS ({eifs} μs)"
                )
                self.node.phy_layer.reset_ampdu_collision_event(ch_id)
                # reset rts collision event since it might happen that both rts and ampdu collision occur on the same channel, and the ampdu collision EIFS should remain as it is longer
                self.node.phy_layer.reset_rts_collision_event(ch_id)
                yield from self._wait_until_channel_idle(ch_id, eifs, waiting_eifs=True)
                return

            if rts_event.triggered:
                eifs = self.sparams.DIFS_us + CTS_TIMEOUT_us
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> RTS collision on Channel {ch_id}, waiting EIFS ({eifs} μs)"
                )
                self.node.phy_layer.reset_rts_collision_event(ch_id)
                yield from self._wait_until_channel_idle(ch_id, eifs, waiting_eifs=True)
                return

            if timeout.triggered:
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
                f"{self.node.type} {self.node.id} -> Sensing channels does not contain only one (primary) channel! (Channels: {', '.join(map(str, self.node.phy_layer.sensing_channels_ids))})"
            )

        primary_channel_id = self.node.phy_layer.get_primary_channel_id()

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
        yield AnyOf(self.env, idle_channel_events)

        # Get list of channels that completed the idle period
        idle_channels = [
            sensing_channels[i]
            for i, event in enumerate(idle_channel_events)
            if event.triggered
        ]

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Channels {', '.join(map(str, idle_channels))} have been idle for {str(ch_duration_us) + 'μs' if isinstance(ch_duration_us, float) else ', '.join(map(str, [ch_duration_us[ch_id] for ch_id in idle_channels])) + 'μs respectively'}"
        )

        return idle_channels

    def _initialize_backoff_slots(self):
        if self.backoff_slots > 0:
            self.logger.debug(
                f"{self.node.type} {self.node.id} -> Backoff slots already set ({self.backoff_slots})"
            )
            return

        if self.rl_driven:
            self.backoff_slots = random.randint(0, self.cw_current - 1)
        else:
            cw = min(self.sparams.CW_MIN * (2**self.retries), self.sparams.CW_MAX)
            self.backoff_slots = random.randint(0, max(0, cw - 1))
        if not self.is_first_tx:
            self.backoff_slots += 1

        self.logger.info(
            f"{self.node.type} {self.node.id} -> Backoff slots: {self.backoff_slots} (retries: {self.retries})"
        )

    def _standard_backoff(self):
        while self.backoff_slots > 0:
            event = self.node.phy_layer.get_busy_event(
                self.node.phy_layer.get_primary_channel_id()
            )
            slot_start_time = self.env.now
            yield self.env.timeout(self.sparams.SLOT_TIME_us) | event

            if (
                event.triggered
                and self.env.now < slot_start_time + self.sparams.SLOT_TIME_us
            ):
                self.logger.debug(
                    f"{self.node.type} {self.node.id} -> Primary Channel busy, pausing backoff ({self.backoff_slots})..."
                )
                return -1

            self.backoff_slots -= 1
            self.logger.debug(
                f"{self.node.type} {self.node.id} -> Backoff slots reduced ({self.backoff_slots})"
            )

    def _toy_backoff(self, channels_ids: set[int]):
        if not channels_ids:
            self.logger.critical(
                f"{self.node.type} {self.node.id} -> No channels to backoff on!"
            )
            return -1

        bo_channels = set(channels_ids)
        slot_remaining_time = self.sparams.SLOT_TIME_us

        while self.backoff_slots > 0:
            busy_events = {
                ch_id: self.node.phy_layer.get_busy_event(ch_id)
                for ch_id in bo_channels
            }
            event = self.env.any_of(list(busy_events.values()))
            slot_start_time = self.env.now
            yield self.env.timeout(slot_remaining_time) | event

            if event.triggered and self.env.now < slot_start_time + slot_remaining_time:
                busy_channels = {
                    ch_id for ch_id, ev in busy_events.items() if ev.triggered
                }
                slot_remaining_time = self.env.now - slot_start_time
                bo_channels -= busy_channels

                if busy_channels:
                    self.logger.debug(
                        f"{self.node.type} {self.node.id} -> Channels {', '.join(map(str, busy_channels))} became busy. Remaining idle: {', '.join(map(str, bo_channels))}"
                    )

                if not bo_channels:
                    self.logger.debug(
                        f"{self.node.type} {self.node.id} -> All channels are busy, pausing backoff ({self.backoff_slots})..."
                    )
                    return -1

            self.backoff_slots -= 1
            self.logger.debug(
                f"{self.node.type} {self.node.id} -> Backoff slots reduced ({self.backoff_slots})"
            )

    def _handle_scb_transmission(self):
        # SCB: transmit on bonded channel only if secondary channels have been idle for at least during PIFS
        primary = self.node.phy_layer.get_primary_channel_id()
        secondary = set(self.node.phy_layer.channels_ids) - {primary}

        idle_secondaries = {
            ch_id
            for ch_id in secondary
            if self.node.phy_layer.has_been_idle_during_duration(
                ch_id, self.sparams.PIFS_us
            )
        }

        if idle_secondaries != secondary:
            self.logger.debug(
                f"{self.node.type} {self.node.id} -> Secondary channels [{', '.join(map(str, secondary - idle_secondaries))}] not idle during PIFS. Skipping transmission!"
            )
            self.backoff_slots = -1
            self.logger.header(
                f"{self.node.type} {self.node.id} -> Backoff finished but skipping transmission..."
            )
            return -1

        self.node.phy_layer.set_transmitting_channels(self.node.phy_layer.channels_ids)

    def _handle_dcb_transmission(self):
        # DCB: select the widest contiguous subset of channels (including the primary) that were idle for at least a PIFS duration for transmission according to IEEE 802.11 channelization
        primary = self.node.phy_layer.get_primary_channel_id()
        secondary = set(self.node.phy_layer.channels_ids) - {primary}

        idle_secondaries = {
            ch_id
            for ch_id in secondary
            if self.node.phy_layer.has_been_idle_during_duration(
                ch_id, self.sparams.PIFS_us
            )
        }

        available_channels = {
            primary
        } | idle_secondaries  # Union primary and idle secondaries
        valid_bonds = self.node.phy_layer.get_valid_bonds()

        valid_idle_bonds = [
            bond
            for bond in valid_bonds
            if primary in bond and set(bond).issubset(available_channels)
        ]

        # Pick the widest subset for transmission (breaking ties at random)
        if valid_idle_bonds:
            max_len = max(len(b) for b in valid_idle_bonds)
            longest_bonds = [b for b in valid_idle_bonds if len(b) == max_len]
            selected = set(random.choice(longest_bonds))
        else:
            selected = {primary}
        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Selected channel bond for transmission: {', '.join(map(str, selected))}"
        )

        self.node.phy_layer.set_transmitting_channels(selected)

    def _handle_toy_transmission(self, channels_ids: set[int]):
        if channels_ids:
            self.node.phy_layer.set_transmitting_channels(set(channels_ids))

    def _update_tx_stats(self):
        if self.is_first_tx:
            self.node.tx_stats.first_tx_attempt_us = self.env.now
            self.is_first_tx = False
        else:
            self.node.tx_stats.last_tx_attempt_us = self.env.now

        self.node.tx_stats.tx_attempts += 1
        self.logger.header(f"{self.node.type} {self.node.id} -> Backoff finished")

    def _handle_post_backoff(self, channels_ids: set[int]):
        if self.sparams.BONDING_MODE == 0:
            result = self._handle_scb_transmission()
            if result == -1:
                return
        elif self.sparams.BONDING_MODE == 1:
            self._handle_dcb_transmission()
        elif self.sparams.BONDING_MODE == 2:
            self._handle_toy_transmission(channels_ids)

        self._update_tx_stats()

    def backoff(self, channels_ids: set[int] = None):
        self.bo_start_time_us = self.env.now
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Starting Backoff{'' if channels_ids is None else f' on channels'} {', '.join(map(str, self.node.phy_layer.sensing_channels_ids)) if channels_ids is not None else ''}..."
        )

        self._initialize_backoff_slots()

        if self.sparams.BONDING_MODE in [0, 1]:
            result = yield from self._standard_backoff()
        else:
            result = yield from self._toy_backoff(channels_ids)

        if result == -1:
            return

        self.bo_duration_us += self.env.now - self.bo_start_time_us

        self._handle_post_backoff(channels_ids)

    def _get_aggregatable_mpdus(self):
        agg_mpdus = []
        total_size = 0
        first_mpdu = cast(MPDU, self.tx_queue.items[0])
        destination = first_mpdu.dst_id

        for mpdu in self.tx_queue.items[:]:
            mpdu = cast(MPDU, mpdu)
            if mpdu.dst_id != destination:
                continue
            if total_size + mpdu.size_bytes > self.sparams.MAX_AMPDU_SIZE_bytes:
                break
            agg_mpdus.append(mpdu)
            total_size += mpdu.size_bytes

        return agg_mpdus, total_size

    def ampdu_aggregation(self):
        """Aggregates MPDUs into an A-MPDU frame."""
        if self.tx_ampdu and self.tx_ampdu.retries == 0:
            # If retries is 0 it means that CTS timeout occurred, so no need to aggregate
            return

        self.logger.header(f"{self.node.type} {self.node.id} -> Aggregating MPDUs...")
        agg_mpdus, total_size = self._get_aggregatable_mpdus()

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

    def _handle_cts_timeout(self):
        self.logger.warning(f"{self.node.type} {self.node.id} -> CTS timeout...")
        self.cts_event = None
        self.cts_timedout = True
        self.retries += 1
        self.node.tx_stats.tx_failures += 1

    def wait_for_cts(self):
        self.set_state(MACState.RX)
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Waiting for CTS from {self.tx_ampdu.dst_id}..."
        )
        self.cts_event = self.env.event()
        yield self.env.timeout(CTS_TIMEOUT_us) | self.cts_event

        if not self.cts_event.triggered:
            self._handle_cts_timeout()
            return

        self.retries = 0
        self.cts_timedout = False
        yield self.env.timeout(self.sparams.SIFS_us)
        yield self.env.process(self.transmit_ampdu())

    def transmit_ampdu(self):
        self.node.tx_stats.ampdus_tx += 1
        self.tx_start_time_us = self.env.now
        yield self.env.process(self.transmit(self.tx_ampdu))
        yield self.env.process(self.wait_for_back())

    def _handle_back_timeout(self):
        self.logger.warning(f"{self.node.type} {self.node.id} -> BACK timeout...")
        self.back_event = None
        self.retries += 1
        self.node.tx_stats.tx_failures += 1

    def _process_back(self):
        sent_mpdus = self.tx_ampdu.mpdus
        lost_mpdus = self.rx_back.lost_mpdus
        rx_mpdus = set(sent_mpdus) - set(lost_mpdus)

        self.node.tx_stats.pkts_tx += len(sent_mpdus)
        self.node.tx_stats.pkts_success += len(rx_mpdus)

        for mpdu in sent_mpdus:
            self.node.tx_stats.tx_app_bytes += mpdu.packet.size_bytes
            mpdu.back_reception_time_us = self.env.now

        self.logger.info(
            f"{self.node.type} {self.node.id} -> {len(rx_mpdus)} Packets successfully transmitted"
        )

        if lost_mpdus:
            self.node.tx_stats.pkts_fail += len(lost_mpdus)
            self.logger.warning(
                f"{self.node.type} {self.node.id} -> {len(lost_mpdus)} Packet failures"
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

        for mpdu in rx_mpdus:
            self.del_mpdu_from_queue(mpdu)

        self.node.tx_stats.add_to_tx_queue_history(
            self.env.now, len(self.tx_queue.items)
        )

        self.tx_ampdu = None
        self.retries = 0

    def wait_for_back(self):
        self.set_state(MACState.RX)
        self.logger.header(
            f"{self.node.type} {self.node.id} -> Waiting for BACK from {self.tx_ampdu.dst_id}..."
        )
        self.back_event = self.env.event()

        yield self.env.timeout(BACK_TIMEOUT_us) | self.back_event

        self.tx_duration_us += self.env.now - self.tx_start_time_us

        if self.sparams.ENABLE_RTS_CTS:
            self.node.phy_layer.end_nav()

        if not self.back_event.triggered:
            self._handle_back_timeout()
            return

        self._process_back()

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

    def _update_rx_timestamps(self):
        if self.is_first_rx:
            self.node.rx_stats.first_rx_time_us = self.env.now
            self.is_first_rx = False
        else:
            self.node.rx_stats.last_rx_time_us = self.env.now

    def _update_rx_stats(self, data_unit: DataUnit):
        self.node.rx_stats.data_units_rx += 1
        self.node.rx_stats.rx_mac_bytes += data_unit.size_bytes

    def _log_to_wandb_rx(self, ampdu: AMPDU):
        if wandb.run:
            total_app_bits_rx = sum(mpdu.packet.size_bytes * 8 for mpdu in ampdu.mpdus)
            total_app_bits_success = sum(
                mpdu.packet.size_bytes * 8
                for mpdu in ampdu.mpdus
                if not mpdu.is_corrupted
            )
            delta_t_sec = (
                self.env.now - self.prev_rx_time_us
            ) / 1e6  # microseconds to seconds
            instant_throughput_mbps = (
                (total_app_bits_rx / delta_t_sec) / 1e6 if delta_t_sec != 0 else 0
            )
            instant_goodput_mbps = (
                (total_app_bits_success / delta_t_sec) / 1e6 if delta_t_sec != 0 else 0
            )

            # Time-weighted EMA
            ema_tau = 0.2
            alpha = 1 - math.exp(-delta_t_sec / ema_tau)
            self.ema_goodput_mbps = (
                alpha * instant_goodput_mbps + (1 - alpha) * self.ema_goodput_mbps
            )

            wandb.log(
                {
                    f"node_{self.node.id}/rx_stats/instant_app_throughput_mbps": instant_throughput_mbps,
                    f"node_{self.node.id}/rx_stats/instant_goodput_mbps": instant_goodput_mbps,
                    f"node_{self.node.id}/rx_stats/ema_goodput_mbps": self.ema_goodput_mbps,
                    "env_time_us": self.env.now,
                }
            )

    def _process_received_ampdu(self, ampdu: AMPDU):
        back = BACK(ampdu, self.node.id, ampdu.src_id, self.env.now)
        self._log_to_wandb_rx(ampdu)

        for mpdu in ampdu.mpdus:
            self.node.rx_stats.pkts_rx += 1
            if mpdu.is_corrupted:
                self.node.rx_stats.pkts_fail += 1
                back.add_lost_mpdus(mpdu)
            else:
                self.node.rx_stats.pkts_success += 1
                self.node.rx_stats.rx_app_bytes += mpdu.packet.size_bytes
                self.node.app_layer.packet_from_mac(mpdu.packet)

        self.node.tx_stats.backs_tx += 1
        self.prev_rx_time_us = self.env.now
        self.env.process(self.send_response(back))

    def _process_received_rts(self, rts: RTS):
        self.node.rx_stats.rts_rx += 1

        if self.state == MACState.IDLE:
            cts = CTS(rts.dst_id, rts.src_id, self.env.now)
            self.env.process(self.send_response(cts))
            self.node.tx_stats.cts_tx += 1
        else:
            self.logger.warning(
                f"{self.node.type} {self.node.id} -> Ignoring RTS received from {rts.src_id} (busy)"
            )

    def _process_received_cts(self, cts: CTS):
        self.node.rx_stats.cts_rx += 1

        if self.cts_event and not self.cts_event.triggered:
            if self.sparams.ENABLE_RTS_CTS:
                self.node.tx_stats.tx_successes += 1

            self.cts_event.succeed()
        else:
            self.logger.warning(
                f"{self.node.type} {self.node.id} -> Ignoring CTS received from {cts.src_id} (outdated)"
            )

    def _process_received_back(self, back: BACK):
        self.node.rx_stats.backs_rx += 1

        if self.back_event and not self.back_event.triggered:
            self.rx_back = back
            if not self.sparams.ENABLE_RTS_CTS:
                self.node.tx_stats.tx_successes += 1
            elif (
                back.ampdu_id == self.tx_ampdu.id
                and self.tx_ampdu.size_bytes <= self.sparams.RTS_THRESHOLD_bytes
            ):
                self.node.tx_stats.tx_successes += 1

            self.back_event.succeed()
            self.set_state(MACState.IDLE)
        else:
            self.logger.warning(
                f"{self.node.type} {self.node.id} -> Ignoring BACK received from {back.src_id} (outdated)"
            )

    def receive(self, data_unit: DataUnit):
        self._update_rx_timestamps()
        self._update_rx_stats(data_unit)

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
            self._process_received_ampdu(data_unit)
        elif data_unit.type == "RTS":
            self._process_received_rts(data_unit)
        elif data_unit.type == "CTS":
            self._process_received_cts(data_unit)
        elif data_unit.type == "BACK":
            self._process_received_back(data_unit)

    def send_response(self, data_unit):
        yield self.env.timeout(self.sparams.SIFS_us)
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

    def _get_channel_durations(self):
        ch_duration_us = {}
        ch_waited_times = {}

        for ch_id in self.node.phy_layer.sensing_channels_ids:
            ch_duration_us[ch_id] = self.sparams.DIFS_us
            ch_waited_times[ch_id] = 0

        # If an AMDPU/RTS collision occured while not contending it should be sensed during EIFS
        # This happens when an RTS and AMPDU (sent due to size smaller than RTS_THRESHOLD_SIZE) collide
        # The RTS sender does not count the channel as idle during the time elapsed from the delivery of the AMPDU until its CTS timeout occurs.
        for ch_id in self.node.phy_layer.get_ampdu_collisions_channels_ids():
            waited_time_us = self.env.now - self.node.phy_layer.get_last_collision_time(
                ch_id
            )
            duration_us = self.sparams.DIFS_us + BACK_TIMEOUT_us
            ch_waited_times[ch_id] = waited_time_us
            ch_duration_us[ch_id] = duration_us

            self.logger.debug(
                f"{self.node.type} {self.node.id} -> AMPDU collision on ch {ch_id}, waiting EIFS ({duration_us} us), already waited {waited_time_us} us"
            )

        return ch_duration_us, ch_waited_times

    def _csma_ca(self):
        ch_duration_us, ch_waited_times = self._get_channel_durations()

        self.node.phy_layer.reset_collision_events()
        idle_channels = None

        self.sensing_start_time_us = self.env.now

        if self.sparams.BONDING_MODE in [0, 1]:  # Standard behavior
            yield self.env.process(
                self.wait_until_primary_idle(ch_duration_us, ch_waited_times)
            )
        else:  # Toy behavior
            idle_channels = yield self.env.process(
                self.wait_until_any_idle(ch_duration_us, ch_waited_times)
            )

        self.sensing_duration_us += self.env.now - self.sensing_start_time_us

        if (
            self.rl_driven
            and self.backoff_slots == 0
            and self.cfg.DISABLE_SIMULTANEOUS_ACTION_SELECTION
            and not self.cts_timedout
            and self.rl_mode == 1
        ):
            cw_freq = self.rl_settings.get("cw_frequency", 1)
            if self.tx_counter % cw_freq == 0:
                self._run_cw_agent()

        yield self.env.process(self.backoff(idle_channels))

    def _transmit_data(self):
        if self.tx_ampdu is None:
            return

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

    def _run_joint_agent(self):
        current_channel = self.node.phy_layer.channels_ids
        contenders_per_channel = [
            (
                c
                if i + 1 not in current_channel
                else c - 1 - len(self.node.associated_stas)
            )
            for i, c in enumerate(self.node.phy_layer.get_contender_count())
        ]  # do not count itself nor associated STAs as contenders
        normalized_contenders = [
            c / sum(contenders_per_channel) if sum(contenders_per_channel) > 0 else 0
            for c in contenders_per_channel
        ]
        busy_flags_per_channel = self.node.phy_layer.get_busy_flags()
        queue_size = len(self.tx_queue.items)

        joint_ctx = [
            *normalized_contenders,  # normalized in range [0, 1]
            *busy_flags_per_channel,  # already in range [0, 1]
            queue_size
            / self.sparams.MAX_TX_QUEUE_SIZE_pkts,  # normalized in range [0, 1]
        ]

        joint_action = self.rl_controller.decide_joint_action(np.array(joint_ctx))

        self.node.phy_layer.set_channels(CHANNEL_MAP[joint_action[0]])

        self.node.phy_layer.set_sensing_channels(PRIMARY_CHANNEL_MAP[joint_action[1]])

        self.cw_current = CW_MAP[joint_action[1]]

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Agent selected action {joint_action}, channel changed to {CHANNEL_MAP[joint_action[0]]}, primary channel changed to {PRIMARY_CHANNEL_MAP[joint_action[1]]}, CW size changed to {self.cw_current}"
        )

    def _run_channel_agent(self):
        current_channel = self.node.phy_layer.channels_ids
        contenders_per_channel = [
            (
                c
                if i + 1 not in current_channel
                else c - 1 - len(self.node.associated_stas)
            )
            for i, c in enumerate(self.node.phy_layer.get_contender_count())
        ]  # do not count itself nor associated STAs as contenders
        normalized_contenders = [
            c / sum(contenders_per_channel) if sum(contenders_per_channel) > 0 else 0
            for c in contenders_per_channel
        ]
        busy_flags_per_channel = self.node.phy_layer.get_busy_flags()
        queue_size = len(self.tx_queue.items)

        ch_ctx = [
            *normalized_contenders,  # normalized in range [0, 1]
            *busy_flags_per_channel,  # already in range [0, 1]
            queue_size
            / self.sparams.MAX_TX_QUEUE_SIZE_pkts,  # normalized in range [0, 1]
        ]
        ch_action = self.rl_controller.decide_channel(np.array((ch_ctx)))

        self.node.phy_layer.set_channels(CHANNEL_MAP[ch_action])

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Channel agent selected action {ch_action}, channel changed to {CHANNEL_MAP[ch_action]}"
        )

    def _run_primary_agent(self):
        current_channel = self.node.phy_layer.channels_ids
        contenders_per_channel = [
            (
                c
                if i + 1 not in current_channel
                else c - 1 - len(self.node.associated_stas)
            )
            for i, c in enumerate(self.node.phy_layer.get_contender_count())
        ]  # do not count itself nor associated STAs as contenders
        normalized_contenders = [
            c / sum(contenders_per_channel) if sum(contenders_per_channel) > 0 else 0
            for c in contenders_per_channel
        ]
        busy_flags_per_channel = self.node.phy_layer.get_busy_flags()

        channel_key = next(
            (k for k, v in CHANNEL_MAP.items() if v == current_channel), None
        )

        primary_ctx = [
            channel_key / len(CHANNEL_MAP),  # normalized in range [0, 1]
            *normalized_contenders,  # normalized in range [0, 1]
            *busy_flags_per_channel,  # already in range [0, 1]
        ]
        primary_action = self.rl_controller.decide_primary(
            np.array(primary_ctx), current_channel
        )

        self.node.phy_layer.set_sensing_channels(PRIMARY_CHANNEL_MAP[primary_action])

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> Primary agent selected action {primary_action}, primary channel changed to {PRIMARY_CHANNEL_MAP[primary_action]}"
        )

    def _run_cw_agent(self):
        current_channel = self.node.phy_layer.channels_ids
        current_primary = self.node.phy_layer.sensing_channels_ids
        contenders_per_channel = [
            (
                c
                if i + 1 not in current_channel
                else c - 1 - len(self.node.associated_stas)
            )
            for i, c in enumerate(self.node.phy_layer.get_contender_count())
        ]  # do not count itself nor associated STAs as contenders
        normalized_contenders = [
            c / sum(contenders_per_channel) if sum(contenders_per_channel) > 0 else 0
            for c in contenders_per_channel
        ]
        busy_flags_per_channel = self.node.phy_layer.get_busy_flags()
        queue_size = len(self.tx_queue.items)

        channel_key = next(
            (k for k, v in CHANNEL_MAP.items() if v == current_channel), None
        )
        primary_key = next(
            (k for k, v in PRIMARY_CHANNEL_MAP.items() if v == current_primary), None
        )

        cw_ctx = [
            channel_key / len(CHANNEL_MAP),  # normalized in range [0, 1]
            primary_key / len(PRIMARY_CHANNEL_MAP),  # normalized in range [0, 1]
            *normalized_contenders,  # normalized in range [0, 1]
            *busy_flags_per_channel,  # already in range [0, 1]
            queue_size
            / self.sparams.MAX_TX_QUEUE_SIZE_pkts,  # normalized in range [0, 1]
        ]
        cw_action = self.rl_controller.decide_cw(np.array(cw_ctx))

        self.cw_current = CW_MAP[cw_action]

        self.logger.debug(
            f"{self.node.type} {self.node.id} -> CW agent selected action {cw_action}, CW size changed to {self.cw_current}"
        )

    def _log_to_wandb_delays(self, delay_components: dict):
        if wandb.run:
            wandb.log(
                {
                    f"node_{self.node.id}/delay/sensing": delay_components[
                        "sensing_delay"
                    ],
                    f"node_{self.node.id}/delay/backoff": delay_components[
                        "backoff_delay"
                    ],
                    f"node_{self.node.id}/delay/tx": delay_components["tx_delay"],
                    f"node_{self.node.id}/delay/residual": delay_components[
                        "residual_delay"
                    ],
                    f"node_{self.node.id}/delay/total": sum(delay_components.values()),
                    "env_time_us": self.env.now,
                }
            )

    def _update_rl_agents(self):
        self.tx_counter += 1

        sensing_delay = self.sensing_duration_us
        backoff_delay = self.bo_duration_us
        tx_delay = self.tx_duration_us

        total_delay = self.env.now - self.tx_attempt_time_us

        residual_delay = total_delay - sensing_delay - backoff_delay - tx_delay

        delay_components = {
            "sensing_delay": sensing_delay,
            "backoff_delay": backoff_delay,
            "tx_delay": tx_delay,
            "residual_delay": residual_delay,
        }

        self._log_to_wandb_delays(delay_components)

        if self.rl_driven:
            self.rl_controller.update_agents(delay_components)

    def run(self):
        """Handles channel access, contention, and transmission"""
        while True:
            if (
                self.non_full_event is not None
                and not len(self.tx_queue.items) == self.sparams.MAX_TX_QUEUE_SIZE_pkts
            ):
                (
                    self.non_full_event.succeed()
                    if not self.non_full_event.triggered
                    else None
                )
            if self.tx_queue.items:
                if self.rl_driven and self.backoff_slots == 0 and not self.cts_timedout:
                    (
                        self._update_rl_agents()
                        if self.tx_attempt_time_us is not None
                        else None
                    )
                    self.tx_attempt_time_us = self.env.now
                    self.sensing_duration_us = 0
                    self.bo_duration_us = 0
                    self.tx_duration_us = 0

                    if self.rl_mode == 1:
                        ch_freq = self.rl_settings.get("channel_frequency", 1)
                        prim_freq = self.rl_settings.get("primary_frequency", 1)
                        cw_freq = self.rl_settings.get("cw_frequency", 1)

                        if not self.cfg.DISABLE_SIMULTANEOUS_ACTION_SELECTION:
                            if self.tx_counter % ch_freq == 0:
                                self._run_channel_agent()
                            if self.tx_counter % prim_freq == 0:
                                self._run_primary_agent()
                            if self.tx_counter % cw_freq == 0:
                                self._run_cw_agent()
                        else:
                            if self.tx_counter % ch_freq == 0:
                                self._run_channel_agent()
                            if self.tx_counter % prim_freq == 0:
                                self._run_primary_agent()
                    else:
                        joint_freq = self.rl_settings.get("joint_frequency", 1)

                        if self.tx_counter % joint_freq == 0:
                            self._run_joint_agent()

                yield self.env.process(self._csma_ca())

                if self.backoff_slots > 0 or self.backoff_slots == -1:
                    continue

                self.ampdu_aggregation()

                yield self.env.process(self._transmit_data())
            else:
                self.tx_queue_event = self.env.event()
                yield self.tx_queue_event
