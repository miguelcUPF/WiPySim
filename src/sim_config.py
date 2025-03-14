# ---Simulation Parameters--- #
SIMULATION_TIME_us = 1e6

# ---Network Topology Parameters--- #


# ---Traffic Loading Parameters--- #
ENABLE_TRAFFIC_LOADING = True  # Whether to load traffic from a file

TRAFFIC_SOURCES_LOADING = {
    1: {  # Node 1 as a source
        "destinations": [2, 3],  # Destination nodes
        "traffic_paths": [  # Traffic trace files
            "tests/sim_traces/traffic_trace_node_1_to_node_2_VR.csv",
            "tests/sim_traces/traffic_trace_node_1_to_node_2_Poisson.csv"
        ]
    }
}


# ---Traffic Generation Parameters--- #
TRAFFIC_MODEL = "Poisson"

## Poisson/Bursty/VR Traffic Parameters ##
APP_TRAFFIC_LOAD_kbps = 100e3
MAX_PACKET_SIZE_bytes = 1280

## Bursty/VR Traffic Parameters ##
BURST_SIZE_pkts = 20
AVG_INTER_PACKET_TIME_us = 6

## VR Traffic Parameters ##
FPS = 90

# ---MAC Layer Parameters--- #
SLOT_TIME_us = 9

SIFS_us = 16
DIFS_us = SIFS_us + 2 * SLOT_TIME_us  # 34e3

CW_MIN = 16
CW_MAX = 1023

COMMON_RETRY_LIMIT = 7

MAX_TX_QUEUE_SIZE_pkts = 100
MAX_RX_QUEUE_SIZE_pkts = 200

MAC_HEADER_SIZE_bytes = 32
FCS_SIZE_bytes = 4

MAX_AMPDU_SIZE_bytes = 65535
MDPU_DELIMITER_SIZE_bytes = 4
MPDU_PADDING_SIZE_bytes = 3

BACK_SIZE_PER_MPDU_bytes = 2
BACK_TIMEOUT_us = SIFS_us + SLOT_TIME_us + 20

TTL_ns = 5e8

# ---PHY Layer Parameters--- #


# ---Medium Parameters--- #

# ---Event Logging--- #
ENABLE_CONSOLE_LOGGING = True  # Show logs in console
USE_COLORS_IN_EVENT_LOGS = True  # Enable/disable colors

EXCLUDED_CONSOLE_LEVELS = []
EXCLUDED_CONSOLE_MODULES = ["GEN", "LOAD", "PLOTTER"]

ENABLE_EVENT_RECORDING = True
EVENT_RECORDING_PATH = "data/events"

# ---Packet Traffic Recording--- #
ENABLE_TRAFFIC_GEN_RECORDING = False  # enabling this may affect performance
TRAFFIC_GEN_RECORD_PATH = "data/sim_traces/run1"

# ---Statistics Collection--- #
ENABLE_STATS_COLLECTION = False
STATS_SAVE_PATH = "data/statistics"

# ---Visualization--- #
ENABLE_FIGS_DISPLAY = False

ENABLE_FIGS_SAVING = False
FIGS_SAVE_PATH = "figs"

ENABLE_FIGS_OVERWRITE = True
