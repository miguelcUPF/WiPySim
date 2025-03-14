# ---Simulation Parameters--- #
SIMULATION_TIME_us = 1e6

# ---Network Topology Parameters--- #


# ---Traffic Configuration--- #

## ---Traffic Loading--- ##
# This specifies the loading of traffic traces from files between source and destination nodes.
# Each entry consists of a source node, and for each destination, a list of traffic trace files and their corresponding simulation starting time.
# Keys:
# - "source": the source node number (int).
# - "destinations": a list of dictionaries for each destination node.
# - "destination": the destination node number (int).
# - "traffic_files": a list of dictionaries specifying traffic trace files for that destination and starting time.
# - "file": path to the traffic trace file (string).
# - "start_time_us": the simulation time to start loading the traffic file (int, in microseconds) If not provided, 0 is assumed.
TRAFFIC_LOAD_CONFIG = [
    {
        "source": 1,
        "destinations": [
            {
                "destination": 2,
                "traffic_files": [
                    {"file": "tests/sim_traces/traffic_trace_node_1_to_node_2_VR.csv",
                        "start_time_us": 5000},  # Start after 5000 us
                    {"file": "tests/sim_traces/traffic_trace_node_1_to_node_2_Poisson.csv"}
                ]
            },
            {
                "destination": 3,
                "traffic_files": [
                    {"file": "tests/ws_traces/tshark_processed_traffic.tsv",
                        "start_time_us": 15000}
                ]
            }
        ]
    }
]


## ---Traffic Generation--- ##
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
