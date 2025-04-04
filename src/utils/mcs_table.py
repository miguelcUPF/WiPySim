from src.utils.event_logger import COLORS


"""
802.11ax MCS Table (OFDM) - Modulation, Coding Rate, and Sensitivity Levels
Reference: https://semfionetworks.com/blog/mcs-table-updated-with-80211ax-data-rates/
"""

# Mapping of MCS index to modulation type
MCS_MODULATION = {
    0: "BPSK",
    1: "QPSK",
    2: "QPSK",
    3: "16-QAM",
    4: "16-QAM",
    5: "64-QAM",
    6: "64-QAM",
    7: "64-QAM",
    8: "256-QAM",
    9: "256-QAM",
    10: "1024-QAM",
    11: "1024-QAM",
}

# Bits per subcarrier for each modulation scheme
N_BPSCS = {
    "BPSK": 1,  # 1 bit per subcarrier (BPSK)
    "QPSK": 2,  # 2 bits per subcarrier (QPSK)
    "16-QAM": 4,  # 4 bits per subcarrier (16-QAM)
    "64-QAM": 6,  # 6 bits per subcarrier (64-QAM)
    "256-QAM": 8,  # 8 bits per subcarrier (256-QAM)
    "1024-QAM": 10,  # 10 bits per subcarrier (1024-QAM)
}

# Coding rate for each MCS
CODING_RATE_R = {
    0: 1 / 2,
    1: 1 / 2,
    2: 3 / 4,
    3: 1 / 2,
    4: 3 / 4,
    5: 2 / 3,
    6: 3 / 4,
    7: 5 / 6,
    8: 3 / 4,
    9: 5 / 6,
    10: 3 / 4,
    11: 5 / 6,
}

# Minimum sensitivity levels for each MCS index (in dBm) based on channel width
MCS_min_sensitivity = {
    0: {"20MHz": -82, "40MHz": -79, "80MHz": -76, "160MHz": -73},
    1: {"20MHz": -79, "40MHz": -76, "80MHz": -73, "160MHz": -70},
    2: {"20MHz": -77, "40MHz": -74, "80MHz": -71, "160MHz": -68},
    3: {"20MHz": -74, "40MHz": -71, "80MHz": -68, "160MHz": -65},
    4: {"20MHz": -70, "40MHz": -67, "80MHz": -64, "160MHz": -61},
    5: {"20MHz": -66, "40MHz": -63, "80MHz": -60, "160MHz": -57},
    6: {"20MHz": -65, "40MHz": -62, "80MHz": -59, "160MHz": -56},
    7: {"20MHz": -64, "40MHz": -61, "80MHz": -58, "160MHz": -55},
    8: {"20MHz": -59, "40MHz": -65, "80MHz": -53, "160MHz": -50},
    9: {"20MHz": -57, "40MHz": -54, "80MHz": -51, "160MHz": -48},
    10: {"20MHz": -54, "40MHz": -51, "80MHz": -48, "160MHz": -45},
    11: {"20MHz": -52, "40MHz": -49, "80MHz": -46, "160MHz": -43},
}

# Number of subcarriers for each channel width (in MHz)
N_SD = {
    20: 234,  # 20 MHz => 234 subcarriers
    40: 468,  # 40 MHz => 468 subcarriers
    80: 980,  # 80 MHz => 980 subcarriers
    160: 1960,  # 160 MHz => 1960 subcarriers
}

# OFDM symbol duration in microseconds
T_DFT_us = 12.8


# Function to calculate data rate based on MCS, channel width, number of spatial streams, and guard interval
def calculate_data_rate_bps(
    mcs_index: int,
    channel_width_mhz: int,
    num_spatial_streams: int,
    guard_interval_us: float,
) -> float:
    """
    Calculate the data rate in bps based on MCS, channel width, number of spatial streams, and guard interval.

    Args:
        mcs_index (int): The Modulation and Coding Scheme index.
        channel_width_mhz (int): The channel width in MHz.
        num_spatial_streams (int): The number of spatial streams.
        guard_interval_us (float): The guard interval in microseconds.

    Returns:
        float: The calculated data rate in bits per second (bps).
    """

    # Validate input parameters
    if mcs_index not in MCS_MODULATION:
        raise ValueError(f"Invalid MCS index: {mcs_index}")
    if channel_width_mhz not in N_SD:
        raise ValueError(f"Invalid channel width: {channel_width_mhz}")
    if num_spatial_streams not in [1, 2, 3]:
        raise ValueError(f"Invalid number of spatial streams: {num_spatial_streams}")
    if guard_interval_us not in [0.8, 1.6, 3.2]:
        raise ValueError(f"Invalid guard interval: {guard_interval_us}")

    modulation = MCS_MODULATION[mcs_index]
    num_subcarriers = N_SD[channel_width_mhz]
    num_bits_per_subcarrier = N_BPSCS[modulation]
    coding_rate = CODING_RATE_R[mcs_index]

    data_rate_bps = (
        (num_subcarriers * num_bits_per_subcarrier * coding_rate * num_spatial_streams)
        / (T_DFT_us + guard_interval_us)
        * 1e6
    )

    return round(data_rate_bps, 1)


def get_min_sensitivity_dBm(mcs_index: int, channel_width_mhz: int) -> float:
    """
    Retrieve the minimum sensitivity (in dBm) for the given MCS index and channel width.

    Args:
        mcs_index (int): The Modulation and Coding Scheme index.
        channel_width_mhz (int): The channel width in MHz.

    Returns:
        float: The minimum sensitivity for the given MCS index and channel width.

    Raises:
        ValueError: If any of the input parameters are invalid.
    """

    if mcs_index not in MCS_min_sensitivity:
        raise ValueError(f"Invalid MCS index: {mcs_index}")

    if channel_width_mhz not in N_SD:
        raise ValueError(f"Invalid channel width: {channel_width_mhz}")

    return MCS_min_sensitivity[mcs_index][f"{channel_width_mhz}MHz"]


def get_highest_mcs_index(
    rssi_dbm: float, channel_width_mhz: int, ap_id: int = None, sta_id: int = None
) -> int:
    """
    Retrieve the highest MCS index that can be supported given the eceived signal strength indicator (RSSI) and channel width.

    Args:
        rssi_dbm (float): The received signal strength indicator (RSSI) in dBm.
        channel_width_mhz (int): The channel width in MHz.

    Returns:
        int: The highest MCS index that can be supported. If no MCS index can be supported, returns -1.
    """

    if channel_width_mhz not in N_SD:
        raise ValueError(f"Invalid channel width: {channel_width_mhz}")

    if rssi_dbm < MCS_min_sensitivity[0][f"{channel_width_mhz}MHz"]:
        return -1

    highest_mcs_index = 0
    for mcs_index, sensitivity_dbm in MCS_min_sensitivity.items():
        if sensitivity_dbm[f"{channel_width_mhz}MHz"] <= rssi_dbm:
            highest_mcs_index = mcs_index
        else:
            break

    return highest_mcs_index
