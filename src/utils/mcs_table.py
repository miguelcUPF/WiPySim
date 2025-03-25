# 802.11ax MCS table (OFDM) https://semfionetworks.com/blog/mcs-table-updated-with-80211ax-data-rates/
# Constants for MCS index, modulation, coding rate, etc.
MCS_MODULATION = {
    0: "BPSK", 1: "QPSK", 2: "QPSK", 3: "16-QAM", 4: "16-QAM",
    5: "64-QAM", 6: "64-QAM", 7: "64-QAM",
    8: "256-QAM", 9: "256-QAM", 10: "1024-QAM",
    11: "1024-QAM"
}

N_BPSCS = {
    "BPSK": 1,        # 1 bit per subcarrier (BPSK)
    "QPSK": 2,        # 2 bits per subcarrier (QPSK)
    "16-QAM": 4,      # 4 bits per subcarrier (16-QAM)
    "64-QAM": 6,      # 6 bits per subcarrier (64-QAM)
    "256-QAM": 8,     # 8 bits per subcarrier (256-QAM)
    "1024-QAM": 10    # 10 bits per subcarrier (1024-QAM)
}

# Coding rate for each MCS
CODING_RATE_R = {
    0: 1/2,
    1: 1/2,
    2: 3/4,
    3: 1/2,
    4: 3/4,
    5: 2/3,
    6: 3/4,
    7: 5/6,
    8: 3/4,
    9: 5/6,
    10: 3/4,
    11: 5/6
}

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
    11: {"20MHz": -52, "40MHz": -49, "80MHz": -46, "160MHz": -43}
}

# Subcarriers based on channel width (in MHz)
N_SD = {
    20: 234,   # 20 MHz => 234 subcarriers
    40: 468,   # 40 MHz => 468 subcarriers
    80: 980,   # 80 MHz => 980 subcarriers
    160: 1960  # 160 MHz => 1960 subcarriers
}

# OFDM symbol duration in microseconds
T_DFT_us = 12.8

# Function to calculate data rate based on MCS, channel width, number of spatial streams, and guard interval
def calculate_data_rate_bps(mcs_index: int, channel_width: int, spatial_streams: int, guard_interval: int):
    if mcs_index not in MCS_MODULATION.keys():
        raise ValueError(f"Invalid MCS index: {mcs_index}")
    if channel_width not in N_SD.keys():
        raise ValueError(f"Invalid channel width: {channel_width}")
    if spatial_streams not in [1, 2, 3]:
        raise ValueError(
            f"Invalid number of spatial streams: {spatial_streams}")
    if guard_interval not in [0.8, 1.6, 3.2]:
        raise ValueError(f"Invalid guard interval: {guard_interval}")

    # Number of subcarriers for the given channel width
    n_sd_val = N_SD[channel_width]
    # Number of bits per subcarrier for the given modulation
    n_bpscs_val = N_BPSCS[MCS_MODULATION[mcs_index]]
    # Get coding rate based on MCS
    coding_rate_r_val = CODING_RATE_R[mcs_index]

    # Calculate the data rate
    data_rate = (n_sd_val * n_bpscs_val * coding_rate_r_val *
                 spatial_streams) / (T_DFT_us + guard_interval)

    return round(data_rate, 1) * 1e6  # Convert to bps


def get_min_sensitivity(mcs_index: int, channel_width: int):
    if mcs_index not in MCS_min_sensitivity.keys():
        raise ValueError(f"Invalid MCS index: {mcs_index}")
    if channel_width not in N_SD.keys():
        raise ValueError(f"Invalid channel width: {channel_width}")

    return MCS_min_sensitivity[mcs_index][f"{channel_width}MHz"]
