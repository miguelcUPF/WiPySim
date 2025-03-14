# ---Simulation Parameters--- #
SIMULATION_TIME_us = 1e6

# ---Network Topology--- #

## ---Nodes--- ##
# This defines the nodes in the network and their respective positions.
# Keys:
# - "id": the unique node identifier (int).
# - "pos": a tuple (x, y, z) specifying the node’s coordinates.
NODES = [
    {"id": 1, "pos": (0, 0, 0)}, 
    {"id": 2, "pos": (3, 4, 0)},
    {"id": 3, "pos": (6, 8, 2)},
    {"id": 4, "pos": (5, 5, 5)},
    {"id": 5, "pos": (1, 2, 3)},
    {"id": 6, "pos": (7, 6, 4)},
    {"id": 7, "pos": (9, 1, 8)},
    {"id": 8, "pos": (2, 9, 7)}   
]

## --- Links --- ##
# This defines bidirectional links between nodes, with an optional channel specification.
# Keys:
# - "nodes": a tuple (source, destination) representing the link.
# - "channel" (optional): the channel assigned to the link. If no channel is provided, dynamic channel selection is used.
LINKS = [
    {"nodes": (1, 2), "channel": 1},
    {"nodes": (3, 4), "channel": 2},
    {"nodes": (5, 6), "channel": 1},
    {"nodes": (7, 8), "channel": 3},
]

# ---Traffic Configuration--- #

## ---Traffic Loading--- ##
# This specifies the loading of traffic traces from files between source and destination nodes.
# Each entry consists of a source node, and for each destination, a list of traffic trace files and their corresponding simulation starting time.
# Keys:
# - "source": the source node number (int).
# - "destinations": a list of dictionaries for each destination node.
#   - "destination": the destination node number (int).
#   - "traffic_files": a list of dictionaries specifying traffic trace files for that destination and starting time.
#      - "file": path to the traffic trace file (string).
#      - "start_time_us" (optional): the simulation time to start loading the traffic file (int, in microseconds). Defaults to 0.
TRAFFIC_LOAD_CONFIG = [
    {
        "source": 1,
        "destinations": [
            {
                "destination": 2,
                "traffic_files": [
                    {"file": "tests/sim_traces/traffic_trace_node_1_to_node_2.csv",
                        "start_time_us": 5000},  # Start after 5000 us
                    {"file": "tests/sim_traces/traffic_trace_node_3_to_node_1.csv"}
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
# This defines the traffic generation models (e.g., Poisson, VR, Bursty) between source and destination nodes,and their respective parameters.
# Each entry consists of a source node, and for each destination, a list of traffic generation models and their corresponding parameters.
# Keys:
# - "source": the source node number (int).
# - "destinations": a list of dictionaries for each destination node.
#   - "destination": the destination node number (int).
#   - "models": a list of dictionaries specifying traffic models and parameters.
#      - "model": type of traffic model (str) e.g., "Poisson", "Bursty", or "VR".
#      - "start_time_us" (optional): the simulation time to start loading the traffic file (int, in microseconds) If not provided, 0 is assumed.
#      - "app_traffic_load_kbps" (optional): traffic load in kbps (int, in kbps). Defaults to 100e3 kbps.
#      - "max_packet_size_bytes" (optional): maximum packet size (int, in bytes). Defaults to 1280 bytes.
#      - "burst_size_pkts" (optional, Bursty and VR models only): number of packets per burst (int). Defaults to 20.
#      - "avg_inter_packet_time_us" (optional, Bursty and VR models only): average inter-packet time (int, in microseconds). Defaults to 6.
#      - "fps" (optional, VR model only): generation framerate (int, in frames per second). Defaults to 90 fps.
TRAFFIC_GEN_CONFIG = [
    {
        "source": 1,  # Source node 1
        "destinations": [  # List of destinations for source node 1
            {
                "destination": 2,  # Destination node 2
                "models": [  # List of traffic models to generate from node 1 to node 2
                    {
                        "model": "Poisson",  # Poisson model
                        "traffic_load_kbps": 50e3,
                        "packet_size_bytes": 1280
                    },
                    {
                        "model": "VR",  # VR model
                        "start_time_us": 2000,  # Start time in microseconds for VR model
                        "fps": 60  # Frames per second for VR model
                    }
                ]
            },
            {
                "destination": 3,  # Destination node 3
                "models": [  # List of traffic models to generate from node 1 to node 3
                    {
                        "model": "Bursty",  # Bursty model
                        "traffic_load_kbps": 50e3,
                        "packet_size_bytes": 1024,
                        "burst_size_pkts": 30  # Number of packets in each burst
                    }
                ]
            }
        ]
    }
]

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

EXCLUDED_CONSOLE_LEVELS = ["DEBUG"]
EXCLUDED_CONSOLE_MODULES = []

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
