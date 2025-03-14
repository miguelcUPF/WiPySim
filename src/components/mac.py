import simpy
import random

from src.sim_config import COMMON_RETRY_LIMIT, CW_MIN, CW_MAX, SLOT_TIME_us, DIFS_us, SIFS_us, MAX_TX_QUEUE_SIZE_pkts, MAX_RX_QUEUE_SIZE_pkts, MAX_AMPDU_SIZE_bytes, BACK_TIMEOUT_us
from src.utils.data_units import Packet, AMPDU, MPDU, BACK
from src.utils.states import MACState


class MAC:
    """
    MAC layer transmitter. Handles transmission logic including packet queuing, channel access (DCF), frame aggregation (A-MPDU) and retransmissions
    MAC layer receiver. Handles frame reception, ACK/BACK responses, and demultiplexing MPDUs.
    """

    def __init__(self, env: simpy.Environment, node_id: int):
        self.env = env
        self.name = "MAC"
        self.app_layer = None
        self.phy_layer = None

        self.node_id = node_id

        # TX
        self.state = MACState.IDLE

        self.selected_channels = []

        self.tx_queue = simpy.Store(env, capacity=MAX_TX_QUEUE_SIZE_pkts)

        self.tx_ampdu = None
        self.retries = 0

        self.backoff_finished = False

        self.ampdu_counter = 0
        self.pkt_drop_counter = 0

        self.rx_back = None

        # RX
        self.rx_queue = simpy.Store(env, capacity=MAX_RX_QUEUE_SIZE_pkts)

        self.env.process(self.run())

    def tx_enqueue(self, packet: Packet):
        """Enqueues a packet for transmission"""
        if len(self.tx_queue.items) < MAX_TX_QUEUE_SIZE_pkts:
            mpdu = MPDU(packet, self.env.now)
            self.tx_queue.put(mpdu)

            self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                        f"{self.name}: Packet {packet.id} added to tx queue.", type="debug")
        else:
            self.pkt_drop_counter += 1
            self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                        f"{self.name}: Packet {packet.id} dropped due to full tx queue.", type="warn")
        
    def set_selected_channels(self, channels: list[int]):
        # TODO: Implement channel selection and BW selection (bonding)
        self.selected_channels = channels

    def run(self):
        """Handles channel access and transmission"""
        while True:
            if self.state == MACState.IDLE:
                if len(self.tx_queue.items) == 0:
                    yield self.env.timeout(1)
                    continue
                else:
                    yield self.env.process(self.contend())
                    yield self.env.process(self.backoff())

                    if self.backoff_finished:
                        # TODO: implement RTS/CTS
                        yield self.env.process(self.transmit())

    def contend(self):
        self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Starting contending for channels {self.selected_channels}...", type="header")

        self.set_state(MACState.CONTEND)

        yield self.env.process(self.env.network.medium.are_channels_idle_for(self.selected_channels, DIFS_us))

    def backoff(self):
        self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Starting backoff...", type="header")
        self.set_state(MACState.BACKOFF)

        self.backoff_finished = False

        cw = min(CW_MIN * (2 ** self.retries), CW_MAX)
        backoff_slots = random.randint(0, cw)

        for _ in range(backoff_slots):
            yield self.env.timeout(SLOT_TIME_us)
            if not self.env.network.medium.are_all_channels_idle(self.selected_channels):
                self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                            f"{self.name}: Backoff interrupted by channel activity...", type="warn")
                self.backoff_finished = False
                return
        self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Backoff finished", type="debug")
        self.backoff_finished = True

    def ampdu_aggregation(self):
        """Aggregates MDPUS from the queue into an A-MPDU frame."""
        self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Starting aggregation...", type="header")
        agg_mpdus = []
        total_size = 0

        while self.tx_queue.items and total_size + self.tx_queue.items[0].size <= MAX_AMPDU_SIZE_bytes:
            mpdu = yield self.tx_queue.get()
            agg_mpdus.append(mpdu)
            total_size += mpdu.size

        self.ampdu_counter += 1
        self.tx_ampdu = AMPDU(
            self.ampdu_counter, agg_mpdus, self.env.now)
        self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Created AMPDU {self.tx_ampdu.id} with {len(agg_mpdus)} MPDUs and size {total_size}", type="debug")

    def transmit(self):
        self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Starting transmission...", type="header")
        self.set_state(MACState.TX)

        if not self.tx_ampdu:
            yield self.env.process(self.ampdu_aggregation())

        self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Transmitting A-MPDU {self.tx_ampdu.id}", type="debug")

        yield self.env.process(self.phy_layer.transmit_ampdu(self.tx_ampdu))

        yield self.env.process(self.wait_for_back())

    def handle_retransmission(self, lost_mpdu):
        for mpdu in lost_mpdu:
            mpdu.packet.retries += 1
            if mpdu.packet.retries <= COMMON_RETRY_LIMIT:
                self.tx_enqueue(mpdu.packet)
            else:
                self.pkt_drop_counter += 1
                self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                            f"{self.name}: MPDU {mpdu.packet.id} dropped after max retries", type="error")
        yield self.env.timeout(0)

    def wait_for_back(self):
        self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Waiting for back...", type="header")
        self.set_state(MACState.WAIT_FOR_BACK)
        
        back_event = self.env.event()
        self.env.process(self.receive_back(back_event))

        yield self.env.timeout(BACK_TIMEOUT_us) | back_event

        if back_event.triggered:
            lost_mpdus = self.rx_back.lost_mpdus
            if lost_mpdus:
                self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                            f"{self.name}: {len(lost_mpdus)} MPDUS lost, retransmitting in next transmission", type="warn")
                yield self.env.process(self.handle_retransmission(lost_mpdus))

            yield self.env.process(self.next_tx())
        else:
            self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                        f"{self.name}: BACK timeout, retransmitting entire AMPDU", type="warn")
            yield self.env.process(self.rtx_ampdu())

    def receive_back(self, back_event):
        back_packet = yield self.env.process(self.phy_layer.receive_back())

        if back_packet:
            self.rx_back = back_packet
            self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                    f"{self.name}: Reception of BACK from AMPDU {self.tx_ampdu.id}", type="info")
            back_event.succeed()
        else:
            pass

    def rtx_ampdu(self):
        self.retries += 1
        self.tx_ampdu.retries += 1
        if self.retries < COMMON_RETRY_LIMIT+1:
            self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                        f"{self.name}: Retransmitting A-MPDU...", type="header")
            yield self.env.process(self.backoff())
            yield self.env.process(self.transmit())
        else:
            self.retries = 0
            self.env.event_logger.log_event(self.env.now, f"Node {self.node_id}",
                                        f"{self.name}: AMPDU {self.tx_ampdu.id} dropped after max retries", type="error")
            self.tx_ampdu = None
            self.set_state(MACState.IDLE)

    def next_tx(self):
        self.tx_ampdu = None
        self.retries = 0
        self.set_state(MACState.IDLE)
        yield self.env.timeout(0)

    def set_state(self, new_state: MACState):
        if self.state != new_state:
            self.env.event_logger.log_event(self.env.now,  f"Node {self.node_id}", f"MAC: State changed from {self.state} to {new_state}", type="debug")
            self.state = new_state