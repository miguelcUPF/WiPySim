from src.sim_params import SimParams as sparams


class DataUnit:
    """Abstract base class for all data units in the simulation."""

    def __init__(
        self, creation_time_us: float, size_bytes: int, src_id: int, dst_id: int
    ):
        """
        Initializes a DataUnit.

        Args:
            creation_time_us (float): The time of creation in microseconds.
            size_bytes (int): The size of the data unit in bytes.
            src_id (int): The source node ID.
            dst_id (int): The destination node ID.
        """
        self.size_bytes: int = size_bytes

        self.src_id: int = src_id
        self.dst_id: int = dst_id

        self.creation_time_us: float = (
            creation_time_us  # When the data unit was created
        )
        self.reception_time_us: float | None = None  # When the data unit was received

        self.retries: int = 0  # Number of times the data unit was retransmitted

        self.is_mgmt_ctrl_frame: bool = (
            False  # Whether the data unit is a management/control frame
        )

        self.type: str | None = None  # Type of the data unit

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
        """
        Initializes a Packet.

        Args:
            id (int): The ID of the packet.
            size_bytes (int): The size of the packet in bytes.
            src_id (int): The source node ID.
            dst_id (int): The destination node ID.
            creation_time_us (float): The time of creation in microseconds.
        """
        super().__init__(creation_time_us, size_bytes, src_id, dst_id)

        self.id: int = id  # ID of the packet

        self.type: str = "DATA"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, {super().__repr__()})"


class MPDU(DataUnit):
    def __init__(self, packet: Packet, creation_time_us: float):
        """
        Initializes an MPDU (MAC Protocol Data Unit).

        Args:
            packet (Packet):  The Packet object encapsulated in this MPDU.
            creation_time_us (float): The time of creation in microseconds.
        """
        super().__init__(
            creation_time_us,
            sparams.MAC_HEADER_SIZE_bytes
            + packet.size_bytes
            + sparams.FCS_SIZE_bytes
            + sparams.MDPU_DELIMITER_SIZE_bytes  # already accounted the size of the delimiter and padding that surround each MPDU inside an aggregate
            + sparams.MPDU_PADDING_SIZE_bytes,
            packet.src_id,
            packet.dst_id,
        )

        self.packet: Packet = packet  # The Packet object encapsulated in this MPDU
        self.is_corrupted: bool = False  # Whether the MPDU is corrupted

        self.type: str = "MPDU"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.packet.id}, {super().__repr__()})"


class AMPDU(DataUnit):
    def __init__(self, id: int, mpdus: list[MPDU], creation_time_us: float):
        """
        Initializes an AMPDU (Aggregated MAC Protocol Data Unit).

        Args:
            id (int): The unique identifier for the AMPDU.
            mpdus (list[MPDU]): A list of MPDUs aggregated into this AMPDU.
            creation_time_us (float): The time of creation in microseconds.
        """
        super().__init__(
            creation_time_us,
            sum([mpdu.size_bytes for mpdu in mpdus]),  # Total size of all MPDUs
            mpdus[0].src_id,
            mpdus[0].dst_id,
        )

        self.id: int = id  # ID of the AMPDU

        self.mpdus: list[MPDU] = mpdus  # List of MPDUs aggregated into this AMPDU

        self.is_mgmt_ctrl_frame: bool = False

        self.type: str = "AMPDU"

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, {super().__repr__()})"


class RTS(DataUnit):
    def __init__(self, src_id: int, dst_id: int, creation_time_us: float):
        """
        Initializes an RTS (Request To Send) frame.

        Args:
            src_id (int): The source node ID.
            dst_id (int): The destination node ID.
            creation_time_us (float): The time of creation in microseconds.
        """
        super().__init__(creation_time_us, sparams.RTS_SIZE_bytes, src_id, dst_id)

        self.is_mgmt_ctrl_frame: bool = True

        self.type: str = "RTS"

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


class CTS(DataUnit):
    def __init__(self, src_id: int, dst_id: int, creation_time_us: float):
        """
        Initializes a CTS (Clear To Send) frame.

        Args:
            src_id (int): The source node ID.
            dst_id (int): The destination node ID.
            creation_time_us (float): The time of creation in microseconds.
        """
        super().__init__(creation_time_us, sparams.CTS_SIZE_bytes, src_id, dst_id)

        self.is_mgmt_ctrl_frame: bool = True

        self.type: str = "CTS"

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"


class BACK(DataUnit):
    def __init__(self, ampdu: AMPDU, src_id: int, dst_id: int, creation_time_us: float):
        """
        Initializes a BACK (Block ACK) frame.

        Args:
            ampdu (AMPDU): The AMPDU that this BACK frame is associated with.
            src_id (int): The source node ID.
            dst_id (int): The destination node ID.
            creation_time_us (float): The time of creation in microseconds.
        """
        super().__init__(
            creation_time_us,
            sparams.BACK_SIZE_PER_MPDU_bytes * len(ampdu.mpdus),
            src_id,
            dst_id,
        )

        self.ampdu_id: int = ampdu.id  # ID of the associated AMPDU

        self.lost_mpdus: list[MPDU] = (
            []
        )  # List of MPDUs that were lost during transmission of the associated AMPDU

        self.is_mgmt_ctrl_frame: bool = True

        self.type: str = "BACK"

    def add_lost_mpdus(self, mpdus: list[MPDU]):
        """
        Adds a list of MPDUs to the lost MPDUs list.

        Args:
            mpdus (list[MPDU]): The list of MPDUs that were lost and need to be tracked.
        """
        self.lost_mpdus.append(mpdus)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(ampdu_id={self.ampdu_id}, {super().__repr__()})"
        )


class PPDU(DataUnit):
    def __init__(self, data_unit: DataUnit, creation_time_us: float):
        """
        Initializes a PPDU (PHY Protocol Data Unit).

        Args:
            data_unit (DataUnit): The DataUnit to be encapsulated in the PPDU.
            creation_time_us (float): The time of creation in microseconds.
        """
        super().__init__(
            creation_time_us,
            data_unit.size_bytes + sparams.PHY_HEADER_SIZE_bytes,  # Add PHY header
            data_unit.src_id,
            data_unit.dst_id,
        )

        self.data_unit: DataUnit = data_unit  # The encapsulated DataUnit

        self.type: str = "PPDU"
