class UserConfig:
    # --- Simulation Parameters --- #
    SIMULATION_TIME_us = 1e6  # Total simulation time in microseconds

    SEED = 1  # Set to None for random behavior

    # --- RL Configuration --- #
    ENABLE_RL = (
        False  # Enable/disable RL-driven agents (NUM_CHANNELS in sim_params must be 4)
    )
    RL_MODE = 1  # 0: SARL or 1: MARL

    USE_WANDB = (
        True  # Enable/disable Weights & Biases logging or hyperparameter optimization
    )
    WANDB_PROJECT_NAME = "marl-802.11"
    WANDB_RUN_NAME = "main_run"

    USE_CODECARBON = False  # Enable/disable Codecarbon for emissions tracking. This can significantly slow down the simulation. See .codecarbon.config for Codecarbon configuration. Recommended log_level = ERROR

    # If "MARL", it can be set to True to disable simultaneous action selection so that
    # agents select actions in a sequential manner, following the logical protocol execution
    # timeline, and thus, act at the most contextually appropriate step.
    DISABLE_SIMULTANEOUS_ACTION_SELECTION = True

    # If "MARL", whether to decompose the reward (average packet delay per transmission)
    # into distinct components for each agent (sensing delay, backoff delay, transmission delay, residual delay)
    ENABLE_REWARD_DECOMPOSITION = False

    ## --- Agents settings (only if "MARL") --- ##
    # Agents' weights for decomposed reward. Keys:
    # - "sensing_delay": time in microseconds to spent on average by every tramsmitted packet due to waiting for primary channel to become idle
    # - "backoff_delay": time in microseconds to spent on average by every tramsmitted packet due to reducing backoff slots
    # - "tx_delay": time in microseconds to spent on average by every tramsmitted packet due to its transmission over the medium (including BACK reception or its timeout)
    # - "residual_delay": time in microseconds to spent on average by every tramsmitted packet that is not accounted for the above
    ##  Weights for the channel agent if decomposed reward is enabled:
    CHANNEL_AGENT_WEIGHTS = {
        "sensing_delay": 0.3,
        "backoff_delay": 0.1,
        "tx_delay": 0.3,
        "residual_delay": 0.3,
    }
    ##  Weights for the primary agent if decomposed reward is enabled
    PRIMARY_AGENT_WEIGHTS = {
        "sensing_delay": 0.4,
        "backoff_delay": 0.2,
        "tx_delay": 0.1,
        "residual_delay": 0.3,
    }
    ##  Weights for the cw agent if decomposed reward is enabled
    CW_AGENT_WEIGHTS = {
        "sensing_delay": 0,
        "backoff_delay": 0.35,
        "tx_delay": 0.35,
        "residual_delay": 0.3,
    }

    # Algorithm settings for each agent. Keys:
    # - strategy (optional): "sw_linucb" or "linucb" or "epsilon_greedy" or "decay_epsilon_greedy". Default: "sw_linucb"

    # only if "sw_linucb" or "linucb":
    # - alpha (optional): confidence bound parameter for LinUCB. Default: 1
    # only if "sw_linucb":
    # - window_size (optional): window size for SW-LinUCB. If None, it will be set to the number of actions. Default: None

    # only if "epsilon_greedy" or "decay_epsilon_greedy":
    # - epsilon (optional): epsilon value for epsilon-greedy strategy. Default: 0.1
    # - decay_rate (optional): decay rate for decay epsilon-greedy strategy. Default: 0.99
    # - eta (optional): learning rate. Default: 0.1
    # - gamma (optional): RMSProp decay factor. Default: 0.9
    # - alpha_ema (optional): EMA smoothing factor. Default: 0.1

    # only if "MARL":
    # - channel_frequency (optional): frequency of the channel agent (i.e., how often it selects an action, in transmissions attempts). Default: 1
    # - primary_frequency (optional): frequency of the primary agent (i.e., how often it selects an action, in transmissions attempts). Default: 1
    # - cw_frequency (optional): frequency of the cw agent (i.e., how often it selects an action, in transmissions attempts). Default: 1

    # only if "SARL":
    # - joint_frequency (optional): frequency of the joint agent (i.e., how often it selects an action, in transmissions attempts). Default: 1
    AGENTS_SETTINGS = {
        "strategy": "sw_linucb",
        "channel_frequency": 8,
        "primary_frequency": 4,
        "cw_frequency": 1,
        "epsilon": 0.1,
    }

    ### --- Context settings --- ###
    UTILIZATION_WINDOW_DURATION_US = 100e3 # Duration of the utilization sliding window in microseconds

    # --- Logging Configuration --- #
    ENABLE_CONSOLE_LOGGING = True  # Enable/disable displaying logs in the console (useful for debugging, may affect performance)
    USE_COLORS_IN_LOGS = True  # Enable/disable colored logs

    ENABLE_LOGS_RECORDING = (
        False  # Enable/disable recording logs (may affect performance)
    )
    LOGS_RECORDING_PATH = "data/events"  # Path to the directory where logs will be recorded. A subfolder can be created for each simulation

    # Logging exclusions (if ENABLE_CONSOLE_LOGGING or ENABLE_LOGS_RECORDING is enabled)
    # Format: { "<module_name>": ["<excluded_log_level_1>", "<excluded_log_level_2>", ...] }
    # <module_name>: Module name (e.g., "NETWORK", "NODE", "GEN", "LOAD", "APP", "MAC", "PHY", "MEDIUM", "CHANNEL", "STATS", "MARL")
    # <excluded_log_level>: Log levels to exclude (e.g., "HEADER","DEBUG", "INFO", "WARNING", "ALL")
    EXCLUDED_LOGS = {
        "NETWORK": ["ALL"],
        "NODE": ["ALL"],
        "GEN": ["ALL"],
        "LOAD": ["ALL"],
        "APP": ["ALL"],
        "MAC": ["ALL"],
        "PHY": ["ALL"],
        "MEDIUM": ["ALL"],
        "CHANNEL": ["ALL"],
        "STATS": ["ALL"],
        "MARL": ["ALL"],
    }
    # Logging exclusions for specific nodes IDs (if ENABLE_CONSOLE_LOGGING or ENABLE_LOGS_RECORDING is enabled)
    EXCLUDED_IDS = []

    # --- Visualization --- #
    ENABLE_FIGS_DISPLAY = False  # Enable/disable displaying figures
    ENABLE_FIGS_SAVING = False  # Enable/disable saving figures
    FIGS_SAVE_PATH = "figs/sim"

    # --- Traffic Recording --- #
    # Enable/disable traffic generation recording (may affect performance)
    ENABLE_TRAFFIC_GEN_RECORDING = False
    TRAFFIC_GEN_RECORDING_PATH = "data/sim_traces/run_1"

    # --- Statistics Collection --- #
    ENABLE_STATS_COMPUTATION = False  # Enable/disable computing tx/rx statistics (can affect performance)
    ENABLE_STATS_COLLECTION = False  # Enable/disable collecting statistics
    STATS_SAVE_PATH = "data/statistics"

    # --- Network Configuration --- #
    NETWORK_BOUNDS_m = (10, 10, 2)  # spatial limits of the network in meters (x, y, z)

    ## --- Network Configuration (Basic) --- ##
    # Basic:
    # - A single STA per BSS.
    # - A single traffic generation model applied to all STAs (using specified traffic load but other default parameters).
    # - STAs and APs are randomly positioned within the network bounds.
    # - The allocated channels (and primary channel) per BSS are selected at random.
    # - The agents are considered non-RL-driven.

    NUMBER_OF_BSSS = 1  # Number of Basic Service Sets (BSSs)
    TRAFFIC_MODEL = "Poisson"  # "Poisson", "Bursty", "VR", or "Full"
    TRAFFIC_LOAD_kbps = (
        100e3  # Traffic load in kbps (only for "Poisson", "Bursty", and "VR")
    )

    ## --- Network Configuration (Advanced) --- ##
    # If ENABLE_ADVANCED_NETWORK_CONFIG is set to True, the user can fine-tune:
    # - The number of BSSs, STAs, and their exact positions.
    # - The specific traffic model for each STA, including loading traffic from a file.
    # - Custom parameters like packet size, burst size, and frame rate for different traffic models.

    ENABLE_ADVANCED_NETWORK_CONFIG = (
        False  # Enable/disable advanced network customizaiton
    )

    # Advanced:
    # - Each BSS has an AP and a list of STAs.
    # - The position of APs and STAs can be specified, or they will be placed randomly.
    # - Traffic flows can be customized and specified either from a traffic trace file or a custom model.
    # Keys:
    # - "id": Unique BSS identifier (int).
    # - "ap": The Access Point (AP) of the BSS.
    #   - "id": Unique AP ID (int).
    #   - "pos" (optional): Tuple (x, y, z) specifying the AP’s coordinates in meters. If not specified, the AP is placed at random within the network bounds.
    #   - "channels" (optional): List of 20 MHz channels (e.g., [1, 2, 3, 4]) defining the AP’s operating bandwidth. If not specified, it is randomly selected at runtime.
    #   - "primary_channel" (optional): Primary 20 MHz channel used for contention and control frames. Must be one of the channels in "channel". If not specified, it is randomly selected at runtime.
    #   - "rl_driven" (optional): Whether the AP is RL-driven. If not specified, it is considered non-RL-driven.
    # - "stas": List of associated Stations (STAs).
    #   - "id": Unique STA ID (int).
    #   - "pos" (optional): Tuple (x, y, z) specifying the STA’s coordinates in meters. If not specified, the STA is placed at random within the network bounds.
    # - "traffic_flows": List of traffic flows from the AP to a destination STA.
    #   - "destination": The STA receiving traffic (int).
    #   - "file": (Optional) Loads traffic from a file. Required columns: "frame.time_relative", "frame.len"
    #       - "path": Path to the traffic trace file (CSV/TSV).
    #       - "start_time_us" (optional): When to start loading the file (int, microseconds). Defaults to 0.
    #       - "end_time_us" (optional): When to stop loading the file (int, microseconds). Defaults to full duration.
    #   - "model": (Optional) Generates traffic using a traffic model.
    #       - "name": Traffic model name. Options: "Poisson", "Bursty", "VR", or "Full".
    #       - "start_time_us" (optional): When to start generating traffic (int, microseconds). Defaults to 0.
    #       - "end_time_us" (optional): When to stop generating traffic (int, microseconds). Defaults to full duration.
    #       - "traffic_load_kbps" (optional, only for "Poisson", "Bursty", and "VR"): Traffic load (int, in kbps). Defaults to 100e3.
    #       - "max_packet_size_bytes" (optional): Maximum packet size (int, bytes). Defaults to 1280.
    #       - "burst_size_pkts" (optional, only for "Bursty"): Number of packets per burst (int). Defaults to 20.
    #       - "avg_inter_packet_time_us" (optional, only for "Bursty" and "VR"): Average inter-packet time (int, microseconds). Defaults to 6.
    #       - "fps" (optional, only for "VR"): Frame rate (int, frames per second). Defaults to 90 fps.
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
