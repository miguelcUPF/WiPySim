from src.user_config import UserConfig as cfg_module
from src.sim_params import SimParams as sparams_module

from src.components.network import Network
from src.components.medium import VALID_BONDS
from src.traffic.generator import TrafficGenerator
from src.traffic.loader import TrafficLoader
from src.utils.messages import PRESS_TO_CONTINUE_MSG

import os
import simpy
import random
import logging
import wandb


def validate_params(sparams: sparams_module, logger: logging.Logger):
    bool_params = {
        "ENABLE_RTS_CTS": sparams.ENABLE_RTS_CTS,
        "ENABLE_SHADOWING": sparams.ENABLE_SHADOWING,
    }

    for name, value in bool_params.items():
        if not isinstance(value, bool):
            logger.critical(f"Invalid {name}: {value}. It must be a boolean.")

    if sparams.ENABLE_SHADOWING:  # TODO
        logger.critical(
            "Shadowing is not implemented yet. Please set ENABLE_SHADOWING to False."
        )

    positive_int_params = {
        "MAX_TX_QUEUE_SIZE_pkts": sparams.MAX_TX_QUEUE_SIZE_pkts,
        "SLOT_TIME_us": sparams.SLOT_TIME_us,
        "SIFS_us": sparams.SIFS_us,
        "DIFS_us": sparams.DIFS_us,
        "PIFS_us": sparams.PIFS_us,
        "MAC_HEADER_SIZE_bytes": sparams.MAC_HEADER_SIZE_bytes,
        "FCS_SIZE_bytes": sparams.FCS_SIZE_bytes,
        "MAX_AMPDU_SIZE_bytes": sparams.MAX_AMPDU_SIZE_bytes,
        "MDPU_DELIMITER_SIZE_bytes": sparams.MDPU_DELIMITER_SIZE_bytes,
        "MPDU_PADDING_SIZE_bytes": sparams.MPDU_PADDING_SIZE_bytes,
        "BACK_SIZE_PER_MPDU_bytes": sparams.BACK_SIZE_PER_MPDU_bytes,
        "RTS_THRESHOLD_bytes": sparams.RTS_THRESHOLD_bytes,
        "RTS_SIZE_bytes": sparams.RTS_SIZE_bytes,
        "CTS_SIZE_bytes": sparams.CTS_SIZE_bytes,
        "PHY_HEADER_SIZE_bytes": sparams.PHY_HEADER_SIZE_bytes,
    }

    for name, value in positive_int_params.items():
        if not isinstance(value, int) or value <= 0:
            logger.critical(f"Invalid {name}: {value}. It must be a positive integer.")

    non_negative_int_params = {
        "CW_MIN": sparams.CW_MIN,
        "CW_MAX": sparams.CW_MAX,
        "COMMON_RETRY_LIMIT": sparams.COMMON_RETRY_LIMIT,
    }

    for name, value in non_negative_int_params.items():
        if not isinstance(value, int) or value < 0:
            logger.critical(
                f"Invalid {name}: {value}. It must be a non-negative integer."
            )

    constrained_params = {
        "SPATIAL_STREAMS": (sparams.SPATIAL_STREAMS, {1, 2, 3}),
        "GUARD_INTERVAL_us": (sparams.GUARD_INTERVAL_us, {0.8, 1.6, 3.2}),
        "NUM_CHANNELS": (sparams.NUM_CHANNELS, {1, 2, 4, 8}),
        "BONDING_MODE": (sparams.BONDING_MODE, {0, 1}),
    }

    for name, (value, valid_values) in constrained_params.items():
        if value not in valid_values:
            logger.critical(
                f"Invalid {name}: {value}. It must be one of {valid_values}."
            )

    float_or_int_params = {
        "TX_POWER_dBm": sparams.TX_POWER_dBm,
        "TX_GAIN_dB": sparams.TX_GAIN_dB,
        "RX_GAIN_dB": sparams.RX_GAIN_dB,
        "FREQUENCY_MHz": sparams.FREQUENCY_GHz,
        "PATH_LOSS_EXPONENT": sparams.PATH_LOSS_EXPONENT,
        "SHADOWING_STD_dB": sparams.SHADOWING_STD_dB,
    }

    for name, value in float_or_int_params.items():
        if not isinstance(value, (int, float)):
            logger.critical(f"Invalid {name}: {value}. It must be an integer or float.")

    if not isinstance(sparams.MPDU_ERROR_PROBABILITY, (int, float)) or not (
        0 <= sparams.MPDU_ERROR_PROBABILITY <= 1
    ):
        logger.critical(
            f"Invalid MPDU_ERROR_PROBABILITY: {sparams.MPDU_ERROR_PROBABILITY}. It must be between 0 and 1."
        )

    if sparams.CW_MAX < sparams.CW_MIN:
        logger.critical(
            f"Invalid CW_MAX: {sparams.CW_MAX}. It must be greater than CW_MIN ({sparams.CW_MIN})."
        )

    logger.success("Simulation parameters validated.")


def validate_config(
    cfg: cfg_module, sparams: sparams_module, logger: logging.Logger
) -> None:
    def _is_valid_pos(pos) -> bool:
        if isinstance(pos, tuple) and len(pos) == 3:
            return all(isinstance(x, (int, float)) for x in pos)
        return False

    def _is_within_bounds(pos: tuple, bounds: tuple) -> bool:
        x, y, z = pos
        x_lim, y_lim, z_lim = bounds
        return 0 <= x <= x_lim and 0 <= y <= y_lim and 0 <= z <= z_lim

    def _param_validation(param_validations: dict):
        for param, (
            valid_types,
            min_val,
            max_val,
            inclusive_min,
            inclusive_max,
        ) in param_validations.items():
            val = cfg.AGENTS_SETTINGS.get(param)
            if val is None:
                continue

            # Type check
            if not isinstance(val, valid_types):
                logger.critical(
                    f"Invalid {param}: '{val}'. It must be a {valid_types}."
                )
                continue

            # Lower bound check
            if min_val is not None:
                if inclusive_min and val < min_val:
                    logger.critical(
                        f"Invalid {param}: '{val}'. It must be ≥ {min_val}."
                    )
                elif not inclusive_min and val <= min_val:
                    logger.critical(
                        f"Invalid {param}: '{val}'. It must be > {min_val}."
                    )

            # Upper bound check
            if max_val is not None:
                if inclusive_max and val > max_val:
                    logger.critical(
                        f"Invalid {param}: '{val}'. It must be ≤ {max_val}."
                    )
                elif not inclusive_max and val >= max_val:
                    logger.critical(
                        f"Invalid {param}: '{val}'. It must be < {max_val}."
                    )

    if not cfg.SIMULATION_TIME_us.is_integer() or cfg.SIMULATION_TIME_us <= 0:
        logger.critical(
            f"Invalid SIMULATION_TIME_us: {cfg.SIMULATION_TIME_us}. It must be a positive integer"
        )

    if cfg.SEED is not None:
        if not isinstance(cfg.SEED, int):
            logger.critical(f"Invalid SEED: {cfg.SEED}. It must be an integer.")

    bool_settings = {
        "ENABLE_RL_AGENTS": cfg.ENABLE_RL,
        "USE_WANDB": cfg.USE_WANDB,
        "USE_CODECARBON": cfg.USE_CODECARBON,
        "DISABLE_SIMULTANEOUS_ACTION_SELECTION": cfg.DISABLE_SIMULTANEOUS_ACTION_SELECTION,
        "ENABLE_REWARD_DECOMPOSITION": cfg.ENABLE_REWARD_DECOMPOSITION,
        "ENABLE_CONSOLE_LOGGING": cfg.ENABLE_CONSOLE_LOGGING,
        "USE_COLORS_IN_LOGS": cfg.USE_COLORS_IN_LOGS,
        "ENABLE_LOGS_RECORDING": cfg.ENABLE_LOGS_RECORDING,
        "ENABLE_FIGS_DISPLAY": cfg.ENABLE_FIGS_DISPLAY,
        "ENABLE_FIGS_SAVING": cfg.ENABLE_FIGS_SAVING,
        "ENABLE_TRAFFIC_GEN_RECORDING": cfg.ENABLE_TRAFFIC_GEN_RECORDING,
        "ENABLE_STATS_COLLECTION": cfg.ENABLE_STATS_COLLECTION,
        "ENABLE_STATS_COMPUTATION": cfg.ENABLE_STATS_COMPUTATION,
        "ENABLE_ADVANCED_NETWORK_CONFIG": cfg.ENABLE_ADVANCED_NETWORK_CONFIG,
        "FIRST_AS_PRIMARY": cfg.FIRST_AS_PRIMARY,
    }

    for name, value in bool_settings.items():
        if not isinstance(value, bool):
            logger.critical(f"Invalid {name}: '{value}'. It must be a boolean.")

    if cfg.ENABLE_RL and sparams.NUM_CHANNELS != 4:
        logger.critical(
            f"Invalid NUM_CHANNELS: {sparams.NUM_CHANNELS}. It must be 4 when using RL-driven agents."
        )

    if not isinstance(cfg.RL_MODE, int) and cfg.ENABLE_RL:
        logger.critical(f"Invalid RL_MODE: '{cfg.RL_MODE}'. It must be an integer.")
    if cfg.RL_MODE not in [0, 1]:
        logger.critical(f"Invalid RL_MODE: '{cfg.RL_MODE}'. It must be 0 or 1.")

    if cfg.ENABLE_RL and cfg.RL_MODE != 1:
        if cfg.DISABLE_SIMULTANEOUS_ACTION_SELECTION:
            logger.warning(
                f"RL_MODE is {cfg.RL_MODE}, ignoring DISABLE_SIMULTANEOUS_ACTION_SELECTION..."
            )
        if cfg.ENABLE_REWARD_DECOMPOSITION:
            logger.warning(
                f"RL_MODE is {cfg.RL_MODE}, ignoring ENABLE_REWARD_DECOMPOSITION..."
            )
        for name, val in zip(
            ["CHANNEL_AGENT_WEIGHTS", "PRIMARY_AGENT_WEIGHTS", "CW_AGENT_WEIGHTS"],
            [
                cfg.CHANNEL_AGENT_WEIGHTS,
                cfg.PRIMARY_AGENT_WEIGHTS,
                cfg.CW_AGENT_WEIGHTS,
            ],
        ):
            if val is not None:
                logger.warning(f"RL_MODE is {cfg.RL_MODE}, ignoring {name}...")

    if cfg.ENABLE_RL and cfg.RL_MODE == 1:
        for name, val in zip(
            ["CHANNEL_AGENT_WEIGHTS", "PRIMARY_AGENT_WEIGHTS", "CW_AGENT_WEIGHTS"],
            [
                cfg.CHANNEL_AGENT_WEIGHTS,
                cfg.PRIMARY_AGENT_WEIGHTS,
                cfg.CW_AGENT_WEIGHTS,
            ],
        ):
            if not isinstance(val, dict):
                logger.critical(f"Invalid {name}: '{val}'. It must be a dictionary.")
            if not all(isinstance(x, (float, int)) for x in val.values()):
                logger.critical(
                    f"Invalid {name}: '{val}'. All values must be floats or integers."
                )
            if not all(0 <= x <= 1 for x in val.values()):
                logger.critical(
                    f"Invalid {name}: '{val}'. All values must be between 0 and 1."
                )
            if (
                not [
                    "sensing_delay",
                    "backoff_delay",
                    "tx_delay",
                    "residual_delay",
                ].sort()
                == list(val.keys()).sort()
            ):
                logger.critical(
                    f"Invalid {name}: '{val}'. Must be a dictionary with keys 'sensing_delay', 'backoff_delay', 'tx_delay', 'residual_delay'."
                )
        cfg.AGENTS_SETTINGS.update(
            {
                "channel_weights": cfg.CHANNEL_AGENT_WEIGHTS,
                "primary_weights": cfg.PRIMARY_AGENT_WEIGHTS,
                "cw_weights": cfg.CW_AGENT_WEIGHTS,
            }
        )

    if cfg.ENABLE_RL:
        if not isinstance(cfg.AGENTS_SETTINGS, dict):
            logger.critical(
                f"Invalid AGENTS_SETTINGS: '{cfg.AGENTS_SETTINGS}'. It must be a dictionary."
            )

        strategy = cfg.AGENTS_SETTINGS.get("strategy", None)

        if strategy:
            if strategy not in [
                "sw_linucb",
                "linucb",
                "epsilon_greedy",
                "decay_epsilon_greedy",
            ]:
                logger.critical(
                    f"Invalid strategy: '{cfg.AGENTS_SETTINGS.get('strategy', None)}'. It must be 'sw_linucb', 'linucb', 'epsilon_greedy' or 'decay_epsilon_greedy'."
                )

        if cfg.AGENTS_SETTINGS.get("channel_frequency", None):
            if cfg.RL_MODE != 1:
                logger.warning(
                    f"RL_MODE is {cfg.RL_MODE}, ignoring channel_frequency..."
                )
            else:
                if not isinstance(
                    cfg.AGENTS_SETTINGS.get("channel_frequency", None), int
                ):
                    logger.critical(
                        f"Invalid channel_frequency: '{cfg.AGENTS_SETTINGS.get('channel_frequency', None)}'. It must be an integer."
                    )
                if cfg.AGENTS_SETTINGS.get("channel_frequency", None) < 0:
                    logger.critical(
                        f"Invalid channel_frequency: '{cfg.AGENTS_SETTINGS.get('channel_frequency', None)}'. It must be a positive integer."
                    )

        if cfg.AGENTS_SETTINGS.get("primary_frequency", None):
            if cfg.RL_MODE != 1:
                logger.warning(
                    f"RL_MODE is {cfg.RL_MODE}, ignoring primary_frequency..."
                )
            else:
                if not isinstance(
                    cfg.AGENTS_SETTINGS.get("primary_frequency", None), int
                ):
                    logger.critical(
                        f"Invalid primary_frequency: '{cfg.AGENTS_SETTINGS.get('primary_frequency', None)}'. It must be an integer."
                    )
                if cfg.AGENTS_SETTINGS.get("primary_frequency", None) < 0:
                    logger.critical(
                        f"Invalid primary_frequency: '{cfg.AGENTS_SETTINGS.get('primary_frequency', None)}'. It must be a positive integer."
                    )
                if cfg.AGENTS_SETTINGS.get("channel_frequency", None) is None:
                    logger.critical(
                        f"Channel frequency must be set when primary frequency is set."
                    )
                if cfg.AGENTS_SETTINGS.get(
                    "primary_frequency", None
                ) > cfg.AGENTS_SETTINGS.get("channel_frequency", None):
                    logger.critical(
                        f"Primary frequency must be less than or equal to channel frequency."
                    )
                if (
                    cfg.AGENTS_SETTINGS.get("channel_frequency", None)
                    % cfg.AGENTS_SETTINGS.get("primary_frequency", None)
                    != 0
                ):
                    logger.critical(
                        f"Channel frequency must be a multiple of primary frequency."
                    )

        if cfg.AGENTS_SETTINGS.get("cw_frequency", None):
            if cfg.RL_MODE != 1:
                logger.warning(f"RL_MODE is {cfg.RL_MODE}, ignoring cw_frequency...")
            else:
                if not isinstance(cfg.AGENTS_SETTINGS.get("cw_frequency", None), int):
                    logger.critical(
                        f"Invalid cw_frequency: '{cfg.AGENTS_SETTINGS.get('cw_frequency', None)}'. It must be an integer."
                    )
                if cfg.AGENTS_SETTINGS.get("cw_frequency", None) < 0:
                    logger.critical(
                        f"Invalid cw_frequency: '{cfg.AGENTS_SETTINGS.get('cw_frequency', None)}'. It must be a positive integer."
                    )

        if cfg.AGENTS_SETTINGS.get("joint_frequency", None):
            if cfg.RL_MODE != 0:
                logger.warning("RL_MODE is 1, ignoring joint_frequency...")
            else:
                if not isinstance(
                    cfg.AGENTS_SETTINGS.get("joint_frequency", None), int
                ):
                    logger.critical(
                        f"Invalid joint_frequency: '{cfg.AGENTS_SETTINGS.get('joint_frequency', None)}'. It must be an integer."
                    )
                if cfg.AGENTS_SETTINGS.get("joint_frequency", None) < 0:
                    logger.critical(
                        f"Invalid joint_frequency: '{cfg.AGENTS_SETTINGS.get('joint_frequency', None)}'. It must be a positive integer."
                    )

        if strategy in [
            "sw_linucb",
            "linucb",
        ]:
            unused_params = ["epsilon", "decay_rate", "eta" "gamma", "alpha_ema"]

            for param in unused_params:
                if cfg.AGENTS_SETTINGS.get(param) is not None:
                    logger.warning(f"Strategy {strategy} does not use {param}.")

            # Validation rules: (type, min_val, max_val, inclusive_min, inclusive_max)
            param_validations = {
                "alpha": ((float, int), 0, None, False, None),  # (0, ∞)
                "window_size": ((int, None), 0, None, True, None),  # [0, ∞)
            }

            _param_validation(param_validations)

        if strategy == "linucb":
            if cfg.AGENTS_SETTINGS.get("window_size") is not None:
                logger.warning(f"Strategy {strategy} does not use window_size.")

        if strategy in [
            "epsilon_greedy",
            "decay_epsilon_greedy",
        ]:
            unused_params = ["alpha", "window_size"]

            for param in unused_params:
                if cfg.AGENTS_SETTINGS.get(param) is not None:
                    logger.warning(f"Strategy {strategy} does not use {param}.")

            # Validation rules: (type, min_val, max_val, inclusive_min, inclusive_max)
            param_validations = {
                "epsilon": ((float, int), 0, 1, True, True),  # [0, 1]
                "decay_rate": ((float, int), 0, 1, True, True),  # [0, 1]
                "eta": ((float, int), 0, None, False, None),  # (0, ∞)
                "gamma": ((float, int), 0, 1, False, True),  # (0, 1]
                "alpha_ema": ((float, int), 0, 1, True, False),  # [0, 1)
            }

            _param_validation(param_validations)

    str_settings = {
        "WANDB_PROJECT_NAME": cfg.WANDB_PROJECT_NAME,
        "WANDB_RUN_NAME": cfg.WANDB_RUN_NAME,
        "LOGS_RECORDING_PATH": cfg.LOGS_RECORDING_PATH,
        "FIGS_SAVE_PATH": cfg.FIGS_SAVE_PATH,
        "TRAFFIC_GEN_RECORDING_PATH": cfg.TRAFFIC_GEN_RECORDING_PATH,
        "STATS_SAVE_PATH": cfg.STATS_SAVE_PATH,
    }
    for name, value in str_settings.items():
        if not isinstance(value, str):
            logger.critical(f"Invalid {name}: '{value}'. It must be a string.")

    valid_modules = [
        "NETWORK",
        "NODE",
        "GEN",
        "LOAD",
        "APP",
        "MAC",
        "PHY",
        "MEDIUM",
        "CHANNEL",
        "STATS",
        "MARL",
        "SARL",
    ]
    valid_log_levels = ["HEADER", "DEBUG", "INFO", "WARNING", "ALL"]

    for module, levels in cfg.EXCLUDED_LOGS.items():
        if module not in valid_modules:
            logger.warning(f"Invalid module name: '{module}' in EXCLUDED_LOGS.")

        for level in levels:
            if level not in valid_log_levels:
                logger.warning(
                    f"Invalid log level: '{level}' for module: '{module}' in EXCLUDED_LOGS."
                )

    path_settings = {
        cfg.LOGS_RECORDING_PATH: cfg.ENABLE_LOGS_RECORDING,
        cfg.FIGS_SAVE_PATH: cfg.ENABLE_FIGS_SAVING,
        cfg.TRAFFIC_GEN_RECORDING_PATH: cfg.ENABLE_TRAFFIC_GEN_RECORDING,
        cfg.STATS_SAVE_PATH: cfg.ENABLE_STATS_COLLECTION,
    }

    for path, enabled in path_settings.items():
        if enabled:
            if not os.path.exists(path):
                logger.warning(f"Path '{path}' does not exist. Creating it...")
                os.makedirs(path)

    if not _is_valid_pos(cfg.NETWORK_BOUNDS_m):
        logger.critical(
            f"Invalid NETWORK_BOUNDS_m: {cfg.NETWORK_BOUNDS_m}. It must be a tuple (x, y, z) of integers or floats."
        )

    if not cfg.ENABLE_ADVANCED_NETWORK_CONFIG:
        if not isinstance(cfg.NUMBER_OF_BSSS, int):
            logger.critical(
                f"Invalid NUMBER_OF_BSSS: {cfg.NUMBER_OF_BSSS}. It must be an integer."
            )
        if cfg.NUMBER_OF_BSSS < 1:
            logger.critical(
                f"Invalid NUMBER_OF_BSSS: {cfg.NUMBER_OF_BSSS}. It must be at least 1."
            )
        if not isinstance(cfg.TRAFFIC_MODEL, str):
            logger.critical(
                f"Invalid TRAFFIC_MODEL: {cfg.TRAFFIC_MODEL}. It must be a string."
            )
        if cfg.TRAFFIC_MODEL not in ["Poisson", "Bursty", "VR", "Full"]:
            logger.critical(
                f"Invalid TRAFFIC_MODEL: {cfg.TRAFFIC_MODEL}. It must be 'Poisson', 'Bursty', 'VR', or 'Full'."
            )
        if not isinstance(cfg.TRAFFIC_LOAD_kbps, (int, float)):
            logger.critical(
                f"Invalid TRAFFIC_LOAD_kbps: {cfg.TRAFFIC_LOAD_kbps}. It must be a float or integer."
            )
        if cfg.TRAFFIC_LOAD_kbps < 0:
            logger.critical(
                f"Invalid TRAFFIC_LOAD_kbps: {cfg.TRAFFIC_LOAD_kbps}. It must be non-negative."
            )

    if cfg.ENABLE_ADVANCED_NETWORK_CONFIG:
        if not isinstance(cfg.BSSs_Advanced, list):
            logger.critical(
                f"Invalid BSSs_Advanced: {cfg.BSSs_Advanced}. It must be a list of dictionaries."
            )

        if not all(isinstance(bss, dict) for bss in cfg.BSSs_Advanced):
            logger.critical(
                f"Invalid BSSs_Advanced: {cfg.BSSs_Advanced}. It must be a list of dictionaries."
            )
        used_node_ids = set()
        used_node_pos = set()
        bss_ids = set()
        for bss in cfg.BSSs_Advanced:
            if "id" not in bss:
                logger.critical("A BSS is missing an ID.")

            if not isinstance(bss["id"], int):
                logger.critical(f"BSS ID {bss['id']} is not an integer.")

            if bss["id"] in bss_ids:
                logger.critical(f"BSS ID {bss['id']} is not unique.")
            bss_ids.add(bss["id"])

            if "ap" not in bss or "id" not in bss["ap"]:
                logger.critical(f"BSS {bss['id']} is missing AP ID.")

            ap_id = bss["ap"]["id"]
            if ap_id in used_node_ids:
                logger.critical(
                    f"AP ID {ap_id} is reused across BSSs. Node IDs must be unique."
                )
            used_node_ids.add(ap_id)

            if "pos" in bss["ap"]:
                ap_pos = bss["ap"]["pos"]
                if not _is_valid_pos(ap_pos):
                    logger.critical(
                        f"AP position {ap_pos} is not valid. It must be a tuple (x, y, z) of integers or floats."
                    )
                if not _is_within_bounds(ap_pos, cfg.NETWORK_BOUNDS_m):
                    logger.critical(
                        f"AP position {ap_pos} is outside the network bounds {cfg.NETWORK_BOUNDS_m}."
                    )
                if ap_pos in used_node_pos:
                    logger.critical(
                        f"AP position {ap_pos} is reused across BSSs. Node positions must be unique."
                    )
                used_node_pos.add(ap_pos)

            if "channels" in bss["ap"]:
                channels = bss["ap"]["channels"]
                if not isinstance(channels, list) or not all(
                    isinstance(ch, int) for ch in channels
                ):
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has an invalid 'channels': {channels}. "
                        f"It must be a list of integers (e.g., [1, 2, 3, 4])."
                    )
                if len(set(channels)) != len(channels):
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has duplicate entries in 'channels': {channels}."
                    )
                if max(channels) > sparams.NUM_CHANNELS:
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has channels that exceed the number of channels configured in the simulation ({sparams.NUM_CHANNELS}): {channels}. Please adjust the number of channels in 'sim_params.py'."
                    )
                channels_bw = len(channels) * 20
                if channels_bw not in VALID_BONDS.keys():
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has an invalid number of channels: {channels_bw}. The simulator supports bonds of 20, 40, 80, and 160; thus, up to 8 channels."
                    )
                if set(channels) not in VALID_BONDS[channels_bw]:
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has an invalid set of channels: {channels}. Valid bonds considering a bond of {channels_bw} MHz are: {VALID_BONDS[channels_bw]}."
                    )

            if "primary_channel" in bss["ap"]:
                if not "channels" in bss["ap"]:
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has a 'primary_channel' but no 'channels'. Please add 'channels'."
                    )
                primary = bss["ap"]["primary_channel"]
                if not isinstance(primary, int):
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has an invalid 'primary_channel': {primary}. It must be an integer."
                    )
                if "channels" in bss["ap"] and primary not in bss["ap"]["channels"]:
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has a 'primary_channel' ({primary}) that is not in its 'channels': {bss['ap']['channels']}."
                    )
                if sparams.BONDING_MODE not in [0, 1]:
                    logger.warning(
                        f"AP {ap_id} in BSS {bss['id']} has a 'primary_channel' ({primary}) but bonding mode is not 0 or 1. Primary channel will be ignored and all channels will be used for sensing."
                    )

            if "rl_driven" in bss["ap"]:
                if not cfg.ENABLE_RL:
                    logger.warning(
                        f"AP {ap_id} in BSS {bss['id']} has a 'rl_driven' but RL is not enabled. 'rl_driven' will be ignored."
                    )
                if not isinstance(bss["ap"]["rl_driven"], bool):
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has an invalid 'rl_driven': {bss['ap']['rl_driven']}. It must be a boolean."
                    )
                if bss["ap"]["rl_driven"] == True and sparams.BONDING_MODE != 0:
                    logger.critical(
                        f"AP {ap_id} in BSS {bss['id']} has 'rl_driven' set to True but bonding mode is not 0 (i.e., SCB). Bonding mode must be 0 to use RL. Please set 'rl_driven' to False or set 'BONDING_MODE' to 0."
                    )

            if "stas" not in bss or not bss["stas"]:
                logger.critical(f"BSS {bss['id']} does not have any STAs.")

            for sta in bss["stas"]:
                if "id" not in sta:
                    logger.critical(f"A STA in BSS {bss['id']} is missing ID.")

                sta_id = sta["id"]
                if not isinstance(sta_id, int):
                    logger.critical(f"STA ID {sta_id} is not an integer.")
                if sta_id in used_node_ids:
                    logger.critical(
                        f"STA ID {sta_id} is reused across BSSs. Node IDs must be unique."
                    )
                if sta_id == ap_id:
                    logger.critical(
                        f"STA ID {sta_id} cannot be the same as AP ID {ap_id} in BSS {bss['id']}."
                    )
                used_node_ids.add(sta_id)

                if "pos" in sta:
                    sta_pos = sta["pos"]
                    if not _is_valid_pos(sta_pos):
                        logger.critical(
                            f"STA position {sta_pos} is not valid. It must be a tuple (x, y, z) of integers or floats."
                        )
                    if not _is_within_bounds(sta_pos, cfg.NETWORK_BOUNDS_m):
                        logger.critical(
                            f"AP position {ap_pos} is outside the network bounds {cfg.NETWORK_BOUNDS_m}."
                        )
                    if sta_pos in used_node_pos:
                        logger.critical(
                            f"STA position {sta_pos} is reused across BSSs. Node positions must be unique."
                        )
                    used_node_pos.add(sta_pos)

            has_valid_traffic_flow = False
            if "traffic_flows" not in bss or not bss["traffic_flows"]:
                logger.critical(f"BSS {bss['id']} does not have any traffic flows.")
            for traffic_flow in bss.get("traffic_flows", []):
                if "model" in traffic_flow or "file" in traffic_flow:
                    has_valid_traffic_flow = True
                    break
            if not has_valid_traffic_flow:
                logger.critical(
                    f"BSS {bss['id']} must have at least a traffic flow using either 'model' or 'file'."
                )

            valid_models = ["Poisson", "Bursty", "VR", "Full"]
            for traffic_flow in bss["traffic_flows"]:
                sta_ids = {sta["id"] for sta in bss["stas"]}
                if "destination" not in traffic_flow:
                    logger.critical(
                        f"Missing 'destination' for traffic flow in BSS {bss['id']}"
                    )
                elif traffic_flow["destination"] not in sta_ids:
                    logger.critical(
                        f"Invalid destination {traffic_flow['destination']} for traffic flow in BSS {bss['id']}. It must be one of the STAs in the BSS: {sta_ids}."
                    )

                if "file" in traffic_flow and "model" in traffic_flow:
                    logger.critical(
                        f"Traffic flow in BSS {bss['id']} cannot have both 'model' and 'file'."
                    )

                if "model" in traffic_flow:
                    model = traffic_flow["model"]
                    if model["name"] not in valid_models:
                        logger.critical(
                            f"Invalid traffic model: {model['name']} in BSS {bss['id']}"
                        )
                    if (
                        model.get("start_time_us", None) is not None
                        and not model["start_time_us"].is_integer()
                    ):
                        logger.critical(
                            f"Invalid start_time_us: {model['start_time_us']} in BSS {bss['id']}. It must be an integer."
                        )
                    if (
                        model.get("end_time_us", None) is not None
                        and not model["end_time_us"].is_integer()
                    ):
                        logger.critical(
                            f"Invalid end_time_us: {model['end_time_us']} in BSS {bss['id']}. It must be an integer."
                        )
                    if (
                        model.get("traffic_load_kbps", None) is not None
                        and not model["traffic_load_kbps"].is_integer()
                    ):
                        logger.critical(
                            f"Invalid traffic_load_kbps: {model['traffic_load_kbps']} in BSS {bss['id']}. It must be an integer."
                        )
                    if model.get(
                        "max_packet_size_bytes", None
                    ) is not None and not isinstance(
                        model["max_packet_size_bytes"], int
                    ):
                        logger.critical(
                            f"Invalid max_packet_size_bytes: {model['max_packet_size_bytes']} in BSS {bss['id']}. It must be an integer."
                        )
                    if (
                        model.get("burst_size_pkts", None) is not None
                        and model["name"] == "Bursty"
                        and not isinstance(model["burst_size_pkts"], int)
                    ):
                        logger.warning(
                            f"Invalid burst_size_pkts: {model['burst_size_pkts']} in BSS {bss['id']}. It must be an integer."
                        )
                    if (
                        model.get("avg_inter_packet_time_us", None) is not None
                        and model["name"] in ["Bursty", "VR"]
                        and not isinstance(model["avg_inter_packet_time_us"], int)
                    ):
                        logger.warning(
                            f"Invalid avg_inter_packet_time_us: {model['avg_inter_packet_time_us']} in BSS {bss['id']}. It must be an integer."
                        )
                    if (
                        model.get("fps", None) is not None
                        and model["name"] == "VR"
                        and not isinstance(model["fps"], int)
                    ):
                        logger.warning(
                            f"Invalid fps: {model['fps']} in BSS {bss['id']}. It must be an integer."
                        )

                if "file" in traffic_flow:
                    file = traffic_flow["file"]
                    if not os.path.exists(file["path"]):
                        logger.critical(
                            f"File {file['path']} does not exist in BSS {bss['id']}"
                        )
                    if (
                        file.get("start_time_us", None) is not None
                        and not file["start_time_us"].is_integer()
                    ):
                        logger.critical(
                            f"Invalid start_time_us: {file['start_time_us']} in BSS {bss['id']}. It must be an integer."
                        )
                    if (
                        file.get("end_time_us", None) is not None
                        and not file["end_time_us"].is_integer()
                    ):
                        logger.critical(
                            f"Invalid end_time_us: {file['end_time_us']} in BSS {bss['id']}. It must be an integer."
                        )

    logger.success("User configuration validated.")


def warn_overwriting_enabled_paths(cfg: cfg_module, logger: logging.Logger):
    path_settings = {
        "logs": cfg.ENABLE_LOGS_RECORDING,
        "figures": cfg.ENABLE_FIGS_SAVING,
        "traffic generated": cfg.ENABLE_TRAFFIC_GEN_RECORDING,
        "statistics": cfg.ENABLE_STATS_COLLECTION,
    }

    enabled_settings = [name for name, enabled in path_settings.items() if enabled]

    if enabled_settings:
        logger.warning(
            f"The following data will be recorded: {', '.join(enabled_settings)}."
        )

    overwriting_names = set(["logs"])
    overwriting_names.update(enabled_settings)
    logger.warning(
        f"Existing files in the configured paths for {', '.join(overwriting_names)} will be overwritten. Save existing files first if you don't want to overwrite them."
    )
    input(PRESS_TO_CONTINUE_MSG)


def validate_settings(cfg: cfg_module, sparams: sparams_module, logger: logging.Logger):
    validate_params(sparams, logger)
    validate_config(cfg, sparams, logger)
    warn_overwriting_enabled_paths(cfg, logger)


def initialize_network(
    cfg: cfg_module,
    sparams: sparams_module,
    env: simpy.Environment,
    network: Network = None,
) -> Network:
    def _get_unique_position(
        bounds: tuple, used_positions: set, rng: random.Random
    ) -> tuple:
        """Generate a unique random position within bounds."""
        while True:
            x_lim, y_lim, z_lim = bounds
            pos = (
                round(rng.uniform(0, x_lim), 2),
                round(rng.uniform(0, y_lim), 2),
                round(rng.uniform(0, z_lim), 2),
            )
            if pos not in used_positions:
                used_positions.add(pos)
                return pos

    def _get_random_channels(sparams: sparams_module, rng: random.Random) -> list:
        valid_channel_bonds = []
        for bw, bond_list in VALID_BONDS.items():
            if bw <= sparams.NUM_CHANNELS * 20:
                for bond in bond_list:
                    invalid_bond = False
                    for id_ in bond:
                        if id_ > sparams.NUM_CHANNELS:
                            invalid_bond = True
                            break
                    valid_channel_bonds.append(bond) if not invalid_bond else None
        return list(rng.choice(valid_channel_bonds))

    env.rng = random.Random(cfg.SEED)
    rng = env.rng

    if not network:
        network = Network(cfg, sparams, env)

    if not cfg.ENABLE_ADVANCED_NETWORK_CONFIG:
        bounds = cfg.NETWORK_BOUNDS_m
        used_positions = set()
        last_id = 0  # Start ID counter

        for bss_index in range(cfg.NUMBER_OF_BSSS):
            # Assign AP ID and increment the counter
            last_id += 1
            ap_id = last_id
            ap_pos = _get_unique_position(bounds, used_positions, rng)
            channels = _get_random_channels(sparams, rng)
            sensing_channels = (
                channels[0]
                if cfg.FIRST_AS_PRIMARY and sparams.BONDING_MODE in [0, 1]
                else (
                    rng.choice(channels) if sparams.BONDING_MODE in [0, 1] else channels
                )
            )
            ap = network.add_ap(
                ap_id,
                ap_pos,
                bss_index + 1,
                set(channels),
                {sensing_channels},
            )

            # Create associated STAs
            last_id += 1
            sta_id = last_id
            sta_pos = _get_unique_position(bounds, used_positions, rng)
            network.add_sta(sta_id, sta_pos, bss_index + 1, ap)

            traffic_generator = TrafficGenerator(
                cfg,
                sparams,
                env,
                ap,
                sta_id,
                name=cfg.TRAFFIC_MODEL,
                traffic_load_kbps=cfg.TRAFFIC_LOAD_kbps,
            )
            ap.add_traffic_flow(traffic_generator)
    else:
        bsss_config = cfg.BSSs_Advanced

        bounds = cfg.NETWORK_BOUNDS_m

        used_positions = set()

        for bss in bsss_config:
            bss_id = bss["id"]

            # Create the AP
            ap_id = bss["ap"]["id"]
            ap_pos = bss["ap"].get(
                "pos", _get_unique_position(bounds, used_positions, rng)
            )
            channels = bss["ap"].get("channels", _get_random_channels(sparams, rng))
            if bss["ap"].get("primary_channel", None) is None:
                sensing_channels = (
                    channels[0]
                    if cfg.FIRST_AS_PRIMARY and sparams.BONDING_MODE in [0, 1]
                    else (
                        rng.choice(channels)
                        if sparams.BONDING_MODE in [0, 1]
                        else channels
                    )
                )
            else:
                sensing_channels = bss["ap"]["primary_channel"]
            rl_driven = bss["ap"].get("rl_driven", False) if cfg.ENABLE_RL else False
            ap = network.add_ap(
                ap_id, ap_pos, bss_id, set(channels), {sensing_channels}, rl_driven
            )

            # Create associated STAs
            for sta in bss.get("stas", []):
                sta_id = sta["id"]
                sta_pos = sta.get(
                    "pos", _get_unique_position(bounds, used_positions, rng)
                )
                network.add_sta(sta_id, sta_pos, bss_id, ap)

            for flow in bss.get("traffic_flows", []):
                dst_id = flow["destination"]

                src_node = network.get_node(ap_id)

                if "file" in flow:
                    traffic_loader = TrafficLoader(
                        cfg, sparams, env, src_node, dst_id, **flow["file"]
                    )
                    src_node.add_traffic_flow(traffic_loader)
                if "model" in flow:
                    traffic_generator = TrafficGenerator(
                        cfg, sparams, env, src_node, dst_id, **flow["model"]
                    )
                    src_node.add_traffic_flow(traffic_generator)

    return network


def add_bss_automatically(BSSs, num_bss: int = 0, last_node_id: int = 0):
    bss_id = num_bss + 1
    ap_id = last_node_id + 1
    sta_id = last_node_id + 2

    new_bss = {
        "id": bss_id,
        "ap": {"id": ap_id},
        "stas": [{"id": sta_id}],
        "traffic_flows": [
            {
                "destination": sta_id,
                "model": {"name": "Poisson"},
            }
        ],
    }
    BSSs.append(new_bss)
    return BSSs


def wandb_init(cfg: cfg_module):
    if cfg.USE_WANDB and cfg.ENABLE_RL:
        agent_cfg = {
            "strategy": cfg.AGENTS_SETTINGS["strategy"],
            "alpha": cfg.AGENTS_SETTINGS.get("alpha"),
            "window_size": cfg.AGENTS_SETTINGS.get("window_size"),
            "epsilon": cfg.AGENTS_SETTINGS.get("epsilon"),
            "decay_rate": cfg.AGENTS_SETTINGS.get("decay_rate"),
            "eta": cfg.AGENTS_SETTINGS.get("eta"),
            "gamma": cfg.AGENTS_SETTINGS.get("gamma"),
            "alpha_ema": cfg.AGENTS_SETTINGS.get("alpha_ema"),
        }

        frequency_cfg = {
            "channel": cfg.AGENTS_SETTINGS.get("channel_frequency"),
            "primary": cfg.AGENTS_SETTINGS.get("primary_frequency"),
            "cw": cfg.AGENTS_SETTINGS.get("cw_frequency"),
        }

        weights_cfg = {
            "channel": cfg.CHANNEL_AGENT_WEIGHTS,
            "primary": cfg.PRIMARY_AGENT_WEIGHTS,
            "cw": cfg.CW_AGENT_WEIGHTS,
        }

        wandb.init(
            project=cfg.WANDB_PROJECT_NAME,
            name=cfg.WANDB_RUN_NAME,
            config={
                "seed": cfg.SEED,
                "rl_mode": cfg.RL_MODE,
                "reward_decomposition": cfg.ENABLE_REWARD_DECOMPOSITION,
                "disable_simultaneous_actions": cfg.DISABLE_SIMULTANEOUS_ACTION_SELECTION,
                "agent_settings": agent_cfg,
                "action_frequency": frequency_cfg,
                "reward_weights": weights_cfg,
                "sim_time_us": cfg.SIMULATION_TIME_us,
            },
        )
    elif cfg.USE_WANDB and not cfg.ENABLE_RL:
        wandb.init(
            project=cfg.WANDB_PROJECT_NAME,
            name=cfg.WANDB_RUN_NAME,
            config={"seed": cfg.SEED, "sim_time_us": cfg.SIMULATION_TIME_us},
        )
