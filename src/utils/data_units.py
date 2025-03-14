from src.sim_config import MDPU_DELIMITER_SIZE_bytes, MPDU_PADDING_SIZE_bytes, MAC_HEADER_SIZE_bytes, FCS_SIZE_bytes, BACK_SIZE_PER_MPDU_bytes


class DataUnit:
    def __init__(self, creation_time_us: float, size_bytes: int, source: int, destination: int):
        self.size_bytes = size_bytes

        self.source = source
        self.destination = destination

        self.creation_time_us = creation_time_us
        self.reception_time_us = None

        self.type = None

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size_bytes}, source={self.source}, destination={self.destination})"


class Packet(DataUnit):
    def __init__(self, id: int, size_bytes: int, source: int, destination: int, creation_time_us: float):
        super().__init__(creation_time_us, size_bytes, source, destination)
        self.id = id
        self.type = "DATA"
        self.received = False

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, {super().__repr__()})"


class MPDU(DataUnit):
    def __init__(self, packet: Packet, creation_time_us: float):
        super().__init__(creation_time_us, MAC_HEADER_SIZE_bytes + packet.size_bytes + FCS_SIZE_bytes +
                         MDPU_DELIMITER_SIZE_bytes + MPDU_PADDING_SIZE_bytes, packet.source, packet.destination)
        self.packet = packet
        self.retries = 0
        self.is_corrupted = False

        self.type = "MPDU"
        

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.packet.id}, {super().__repr__()})"


class AMPDU(DataUnit):
    def __init__(self, id: int, mpdus: list[MPDU], creation_time_us: float):
        super().__init__(creation_time_us, sum(
            [mpdu.size_bytes for mpdu in mpdus]), mpdus[0].source, mpdus[0].destination)
        self.id = id
        self.mpdus = mpdus
        self.retries = 0

        self.type = "AMPDU"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, {super().__repr__()})"


class BACK(DataUnit):
    def __init__(self, ampdu: AMPDU, source: int, destination: int, creation_time_us: float):
        super().__init__(creation_time_us, BACK_SIZE_PER_MPDU_bytes *
                         len(ampdu.mpdus), source, destination)

        self.ampdu_id = ampdu.id
        self.lost_mpdus = []
        self.type = "BACK"

    def add_lost_mpdus(self, mpdus: list[MPDU]):
        self.lost_mpdus.extend(mpdus)

    def __repr__(self):
        return f"{self.__class__.__name__}(ampdu_id={self.ampdu_id}, {super().__repr__()})"
