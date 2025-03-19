# --- Simulation Parameters --- #
SIMULATION_TIME_us = 1e6  # Total simulation time in microseconds

SEED = 1  # Set to None for random behavior

# --- Logging Configuration --- #
# Enable/disable displaying logs in the console (useful for debugging)
ENABLE_CONSOLE_LOGGING = True
USE_COLORS_IN_LOGS = True  # Enable/disable colored logs

# Enable/disable recording logs (may affect performance)
ENABLE_LOGS_RECORDING = True
LOGS_RECORDING_PATH = "data/events"

# Logging exclusions
# Format: { "<module_name>": ["<excluded_log_level_1>", "<excluded_log_level_2>", ...] }
# <module_name>: Module name (e.g., "NETWORK", "NODE", "GEN", "LOAD", "APP", "MAC", "PHY", "MEDIUM", "CHANNEL")
# <excluded_log_level>: Log levels to exclude (e.g., "HEADER","DEBUG", "INFO", "WARNING", "ALL")
EXCLUDED_LOGS = {
    "NETWORK": [],
    "NODE": [],
    "GEN": ["ALL"],
    "LOAD": ["ALL"],
    "APP": [],
    "MAC": [],
    "PHY": [],
    "MEDIUM": [],
    "CHANNEL": [],
}

# --- Visualization --- #
ENABLE_FIGS_DISPLAY = False  # Enable/disable displaying figures
ENABLE_FIGS_SAVING = False  # Enable/disable saving figures
FIGS_SAVE_PATH = "figs/sim"

# --- Traffic Recording --- #
# Enable/disable traffic generation recording (may affect performance)
ENABLE_TRAFFIC_GEN_RECORDING = False
TRAFFIC_GEN_RECORDING_PATH = "data/sim_traces/run_1"

# --- Statistics Collection --- #
ENABLE_STATS_COLLECTION = False  # Enable/disable collecting statistics
STATS_SAVE_PATH = "data/statistics"

# --- Network Configuration (Basic Service Sets - BSSs) --- #
# Defines the BSSs in the network and their associated nodes.
# Keys:
# - "id": Unique BSS identifier (int).
# - "ap": The Access Point (AP) of the BSS.
#   - "id": Unique AP ID (int).
#   - "pos": Tuple (x, y, z) specifying the AP’s coordinates in meters.
# - "stas": List of associated Stations (STAs).
#   - "id": Unique STA ID (int).
#   - "pos": Tuple (x, y, z) specifying the STA’s coordinates in meters.
# - "traffic_flows": List of traffic flows from the AP to a destination STA.
#   - "destination": The STA receiving traffic (int).
#   - "file": (Optional) Loads traffic from a file. Required columns: "frame.time_relative", "frame.len"
#       - "path": Path to the traffic trace file (CSV/TSV).
#       - "start_time_us" (optional): When to start loading the file (int, microseconds). Defaults to 0.
#       - "end_time_us" (optional): When to stop loading the file (int, microseconds). Defaults to full duration.
#   - "model": (Optional) Generates traffic using a traffic model.
#       - "name": Traffic model name. Options: "Poisson", "Bursty", "VR".
#       - "start_time_us" (optional): When to start generating traffic (int, microseconds). Defaults to 0.
#       - "end_time_us" (optional): When to stop generating traffic (int, microseconds). Defaults to full duration.
#       - "traffic_load_kbps" (optional): Traffic load (int, in kbps). Defaults to 100e3.
#       - "max_packet_size_bytes" (optional): Maximum packet size (int, bytes). Defaults to 1280.
#       - "burst_size_pkts" (optional, Bursty model only): Number of packets per burst (int). Defaults to 20.
#       - "avg_inter_packet_time_us" (optional, Bursty and VR models only): Average inter-packet time (int, microseconds). Defaults to 6.
#       - "fps" (optional, VR model only): Frame rate (int, frames per second). Defaults to 90 fps.
BSSs = [
    {
        "id": 1,  # A BSS
        "ap": {"id": 1, "pos": (0, 0, 0)},  # BSS Access Point (AP)
        "stas": [
            {"id": 2, "pos": (3, 4, 0)},  # Associated Stations (STAs)
            {"id": 3, "pos": (6, 8, 2)}
        ],
        "traffic_flows": [
            {"destination": 2, "file": {"path": "tests/sim_traces/traffic_trace_node_1_to_node_2.csv",
                                        "start_time_us": 5000, "end_time_us": 10000}},
            {"destination": 3, "file": {
                "path": "tests/ws_traces/tshark_processed_traffic.tsv"}}
        ]
    },
    {
        "id": 2,  # Another BSS
        "ap": {"id": 4, "pos": (5, 5, 5)},
        "stas": [
            {"id": 5, "pos": (1, 2, 3)}
        ],
        "traffic_flows": [
            {
                "destination": 5,
                "model":
                    {"name": "Bursty",
                     "traffic_load_kbps": 50e3,
                     "max_packet_size_bytes": 1240,
                     "burst_size_pkts": 30,
                     "avg_inter_packet_time_us": 5
                     }
            }
        ]
    }
]
