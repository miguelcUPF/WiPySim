from src.sim_params import (
    MDPU_DELIMITER_SIZE_bytes,
    MPDU_PADDING_SIZE_bytes,
    MAC_HEADER_SIZE_bytes,
    PHY_HEADER_SIZE_bytes,
    FCS_SIZE_bytes,
    BACK_SIZE_PER_MPDU_bytes,
    RTS_SIZE_bytes,
    CTS_SIZE_bytes,
)


class DataUnit:
    def __init__(
        self, creation_time_us: float, size_bytes: int, src_id: int, dst_id: int
    ):
        self.size_bytes = size_bytes

        self.src_id = src_id
        self.dst_id = dst_id

        self.creation_time_us = creation_time_us
        self.reception_time_us = None

        self.is_mgmt_ctrl_frame = False

        self.type = None

    def __repr__(self):
        return f"size={self.size_bytes}, src_id={self.src_id}, dst_id={self.dst_id}"


class Packet(DataUnit):
    def __init__(
        self,
        id: int,
        size_bytes: int,
        src_id: int,
        dst_id: int,
        creation_time_us: float,
    ):
        super().__init__(creation_time_us, size_bytes, src_id, dst_id)
        self.id = id
        self.type = "DATA"
        self.received = False

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, {super().__repr__()})"


class MPDU(DataUnit):
    def __init__(self, packet: Packet, creation_time_us: float):
        super().__init__(
            creation_time_us,
            MAC_HEADER_SIZE_bytes
            + packet.size_bytes
            + FCS_SIZE_bytes
            + MDPU_DELIMITER_SIZE_bytes
            + MPDU_PADDING_SIZE_bytes,
            packet.src_id,
            packet.dst_id,
        )
        self.packet = packet
        self.retries = 0
        self.is_corrupted = False

        self.type = "MPDU"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.packet.id}, {super().__repr__()})"


class AMPDU(DataUnit):
    def __init__(self, id: int, mpdus: list[MPDU], creation_time_us: float):
        super().__init__(
            creation_time_us,
            sum([mpdu.size_bytes for mpdu in mpdus]),
            mpdus[0].src_id,
            mpdus[0].dst_id,
        )
        self.id = id
        self.mpdus = mpdus
        self.retries = 0

        self.type = "AMPDU"
        self.is_mgmt_ctrl_frame = False

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, {super().__repr__()})"


class RTS(DataUnit):
    def __init__(self, src_id: int, dst_id: int, creation_time_us: float):
        super().__init__(creation_time_us, RTS_SIZE_bytes, src_id, dst_id)

        self.type = "RTS"
        self.is_mgmt_ctrl_frame = True

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


class CTS(DataUnit):
    def __init__(self, src_id: int, dst_id: int, creation_time_us: float):
        super().__init__(creation_time_us, CTS_SIZE_bytes, src_id, dst_id)

        self.type = "CTS"
        self.is_mgmt_ctrl_frame = True

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


class BACK(DataUnit):
    def __init__(self, ampdu: AMPDU, src_id: int, dst_id: int, creation_time_us: float):
        super().__init__(
            creation_time_us,
            BACK_SIZE_PER_MPDU_bytes * len(ampdu.mpdus),
            src_id,
            dst_id,
        )

        self.ampdu_id = ampdu.id
        self.lost_mpdus = []
        self.type = "BACK"
        self.is_mgmt_ctrl_frame = True

    def add_lost_mpdus(self, mpdus: list[MPDU]):
        self.lost_mpdus.append(mpdus)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(ampdu_id={self.ampdu_id}, {super().__repr__()})"
        )


class PPDU(DataUnit):
    def __init__(self, data_unit: DataUnit, creation_time_us: float):
        super().__init__(
            creation_time_us,
            data_unit.size_bytes + PHY_HEADER_SIZE_bytes,
            data_unit.src_id,
            data_unit.dst_id,
        )

        self.data_unit = data_unit
        self.type = "PPDU"
