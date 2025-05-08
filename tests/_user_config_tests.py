class UserConfig:
    # --- Simulation Parameters --- #
    SIMULATION_TIME_us = 2e6

    SEED = 1

    # --- RL Configuration --- #
    ENABLE_RL = False
    RL_MODE = 1  # 0: SARL or 1: MARL
    USE_WANDB = False
    USE_CODECARBON = False
    WANDB_PROJECT_NAME = "marl-802.11"
    WANDB_RUN_NAME = "test_run"
    DISABLE_SIMULTANEOUS_ACTION_SELECTION = False
    ENABLE_REWARD_DECOMPOSITION = False

    CHANNEL_AGENT_WEIGHTS = {
        "sensing_delay": 0.3,
        "backoff_delay": 0.1,
        "tx_delay": 0.3,
        "residual_delay": 0.3,
    }
    PRIMARY_AGENT_WEIGHTS = {
        "sensing_delay": 0.4,
        "backoff_delay": 0.2,
        "tx_delay": 0.1,
        "residual_delay": 0.3,
    }
    CW_AGENT_WEIGHTS = {
        "sensing_delay": 0,
        "backoff_delay": 0.35,
        "tx_delay": 0.35,
        "residual_delay": 0.3,
    }
    AGENTS_SETTINGS = {
        "strategy": "sw_linucb",
        "channel_frequency": 8,
        "primary_frequency": 4,
        "cw_frequency": 1,
        "epsilon": 0.1,
    }

    # --- Logging Configuration --- #
    ENABLE_CONSOLE_LOGGING = True
    USE_COLORS_IN_LOGS = True

    ENABLE_LOGS_RECORDING = False
    LOGS_RECORDING_PATH = "tests/events"
    EXCLUDED_LOGS = {
        "NETWORK": ["ALL"],
        "NODE": ["ALL"],
        "GEN": ["ALL"],
        "LOAD": ["ALL"],
        "APP": ["ALL"],
    }
    EXCLUDED_IDS = []

    # --- Visualization --- #
    ENABLE_FIGS_DISPLAY = False
    ENABLE_FIGS_SAVING = False
    FIGS_SAVE_PATH = "figs/tests"

    # --- Traffic Recording --- #
    ENABLE_TRAFFIC_GEN_RECORDING = False
    TRAFFIC_GEN_RECORDING_PATH = "tests/sim_traces"

    # --- Statistics Collection --- #
    ENABLE_STATS_COMPUTATION = True
    ENABLE_STATS_COLLECTION = False
    STATS_SAVE_PATH = "tests/statistics"

    # --- Network Configuration --- #
    NETWORK_BOUNDS_m = (10, 10, 2)

    ## --- Network Configuration (Basic) --- ##
    NUMBER_OF_BSSS = 1
    TRAFFIC_MODEL = "Poisson"
    TRAFFIC_LOAD_kbps = 100e3

    ## --- Network Configuration (Advanced) --- ##
    ENABLE_ADVANCED_NETWORK_CONFIG = False
    BSSs_Advanced = [
        {
            "id": 1,  # A BSS
            "ap": {
                "id": 1,
                "pos": (0, 0, 0),
                "channels": [1],
                "primary_channel": 1,
            },  # BSS Access Point (AP)
            "stas": [
                {"id": 2, "pos": (3, 4, 0)},  # Associated Stations (STAs)
                {"id": 3, "pos": (6, 8, 2)},
            ],
            "traffic_flows": [
                {
                    "destination": 2,
                    "file": {
                        "path": "tests/sim_traces/traffic_trace_node_1_to_node_2.csv",
                        "start_time_us": 5000,
                        "end_time_us": 10000,
                    },
                },
                {
                    "destination": 3,
                    "file": {"path": "tests/ws_traces/tshark_processed_traffic.tsv"},
                },
            ],
        },
        {
            "id": 2,  # Another BSS
            "ap": {"id": 4, "pos": (5, 5, 1)},
            "stas": [{"id": 5}],
            "traffic_flows": [
                {
                    "destination": 5,
                    "model": {
                        "name": "Bursty",
                        "traffic_load_kbps": 50e3,
                        "max_packet_size_bytes": 1240,
                        "burst_size_pkts": 30,
                        "avg_inter_packet_time_us": 5,
                    },
                }
            ],
        },
    ]
