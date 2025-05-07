class SimParams:
    # --- MAC Layer Parameters --- #
    MAX_TX_QUEUE_SIZE_pkts = 100

    SLOT_TIME_us = 9
    SIFS_us = 16
    DIFS_us = SIFS_us + 2 * SLOT_TIME_us  # equals 34
    PIFS_us = SIFS_us + SLOT_TIME_us  # equals 25

    CW_MIN = 16
    CW_MAX = 1024

    COMMON_RETRY_LIMIT = 7

    MAC_HEADER_SIZE_bytes = 32

    FCS_SIZE_bytes = 4

    MAX_AMPDU_SIZE_bytes = 65535
    MDPU_DELIMITER_SIZE_bytes = 4
    MPDU_PADDING_SIZE_bytes = 3

    BACK_SIZE_PER_MPDU_bytes = 2

    ENABLE_RTS_CTS = True
    RTS_THRESHOLD_bytes = 2346
    RTS_SIZE_bytes = 20
    CTS_SIZE_bytes = 14

    # --- PHY Layer Parameters --- #
    PHY_HEADER_SIZE_bytes = 24

    SPATIAL_STREAMS = 2  # 1, 2, or 3
    GUARD_INTERVAL_us = 0.8  # 0.8, 1.6, or 3.2

    TX_POWER_dBm = 20
    TX_GAIN_dB = 0
    RX_GAIN_dB = 0

    BONDING_MODE = 0

    # --- Channel Parameters --- #
    FREQUENCY_GHz = 5

    NUM_CHANNELS = 4  # 1, 2, 4, or 8

    PATH_LOSS_EXPONENT = 4  # 4 considering indoor NLOS environment

    ENABLE_SHADOWING = False
    SHADOWING_STD_dB = 6.8  # 6.8 considering indoor NLOS environment

    MPDU_ERROR_PROBABILITY = 0.1
