from src.utils.support import initialize_network
from src.utils.event_logger import get_logger
from src.utils.messages import (
    STARTING_EXECUTION_MSG,
    EXECUTION_TERMINATED_MSG,
    PRESS_TO_CONTINUE_MSG,
    STARTING_SIMULATION_MSG,
    SIMULATION_TERMINATED_MSG,
)

import os
import simpy
import random
import src.user_config as cfg
import src.sim_params as sparams

logger = get_logger("MAIN")


def validate_params():
    bool_params = {
        "ENABLE_RTS_CTS": sparams.ENABLE_RTS_CTS,
        "ENABLE_SHADOWING": sparams.ENABLE_SHADOWING,
    }

    for name, value in bool_params.items():
        if not isinstance(value, bool):
            logger.critical(f"Invalid {name}: {value}. It must be a boolean.")

    positive_int_params = {
        "MAX_TX_QUEUE_SIZE_pkts": sparams.MAX_TX_QUEUE_SIZE_pkts,
        "SLOT_TIME_us": sparams.SLOT_TIME_us,
        "SIFS_us": sparams.SIFS_us,
        "DIFS_us": sparams.DIFS_us,
        "MAC_HEADER_SIZE_bytes": sparams.MAC_HEADER_SIZE_bytes,
        "FCS_SIZE_bytes": sparams.FCS_SIZE_bytes,
        "MAX_AMPDU_SIZE_bytes": sparams.MAX_AMPDU_SIZE_bytes,
        "MDPU_DELIMITER_SIZE_bytes": sparams.MDPU_DELIMITER_SIZE_bytes,
        "MPDU_PADDING_SIZE_bytes": sparams.MPDU_PADDING_SIZE_bytes,
        "BACK_SIZE_PER_MPDU_bytes": sparams.BACK_SIZE_PER_MPDU_bytes,
        "BACK_TIMEOUT_us": sparams.BACK_TIMEOUT_us,
        "RTS_THRESHOLD_bytes": sparams.RTS_THRESHOLD_bytes,
        "RTS_SIZE_bytes": sparams.RTS_SIZE_bytes,
        "CTS_SIZE_bytes": sparams.CTS_SIZE_bytes,
        "CTS_TIMEOUT_us": sparams.CTS_TIMEOUT_us,
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
        "FREQUENCY_MHz": sparams.FREQUENCY_MHz,
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


def validate_config():
    def _is_valid_pos(pos) -> bool:
        if isinstance(pos, tuple) and len(pos) == 3:
            return all(isinstance(x, (int, float)) for x in pos)
        return False

    def _is_within_bounds(pos: tuple, bounds: tuple) -> bool:
        x, y, z = pos
        x_lim, y_lim, z_lim = bounds
        return 0 <= x <= x_lim and 0 <= y <= y_lim and 0 <= z <= z_lim

    if not cfg.SIMULATION_TIME_us.is_integer() or cfg.SIMULATION_TIME_us <= 0:
        logger.critical(
            f"Invalid SIMULATION_TIME_us: {cfg.SIMULATION_TIME_us}. It must be a positive integer"
        )

    if cfg.SEED is not None:
        if not isinstance(cfg.SEED, int):
            logger.critical(f"Invalid SEED: {cfg.SEED}. It must be an integer.")
        random.seed(cfg.SEED)

    bool_settings = {
        "ENABLE_CONSOLE_LOGGING": cfg.ENABLE_CONSOLE_LOGGING,
        "USE_COLORS_IN_LOGS": cfg.USE_COLORS_IN_LOGS,
        "ENABLE_LOGS_RECORDING": cfg.ENABLE_LOGS_RECORDING,
        "ENABLE_FIGS_DISPLAY": cfg.ENABLE_FIGS_DISPLAY,
        "ENABLE_FIGS_SAVING": cfg.ENABLE_FIGS_SAVING,
        "ENABLE_TRAFFIC_GEN_RECORDING": cfg.ENABLE_TRAFFIC_GEN_RECORDING,
        "ENABLE_STATS_COLLECTION": cfg.ENABLE_STATS_COLLECTION,
    }

    for name, value in bool_settings.items():
        if not isinstance(value, bool):
            logger.critical(f"Invalid {name}: '{value}'. It must be a boolean.")

    str_settings = {
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

    used_node_ids = set()
    used_node_pos = set()
    bss_ids = set()
    for bss in cfg.BSSs:
        if "id" not in bss:
            logger.critical("A BSS is missing an ID.")

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

        if "stas" not in bss or not bss["stas"]:
            logger.critical(f"BSS {bss['id']} does not have any STAs.")

        for sta in bss["stas"]:
            if "id" not in sta:
                logger.critical(f"A STA in BSS {bss['id']} is missing ID.")

            sta_id = sta["id"]
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

        valid_models = ["Poisson", "Bursty", "VR"]
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
                ) is not None and not isinstance(model["max_packet_size_bytes"], int):
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


def warn_overwriting_enabled_paths():
    path_settings = {
        "logs": cfg.ENABLE_LOGS_RECORDING,
        "figures": cfg.ENABLE_FIGS_SAVING,
        "traffic generated": cfg.ENABLE_TRAFFIC_GEN_RECORDING,
        "statistics": cfg.ENABLE_STATS_COLLECTION,
    }

    enabled_settings = [name for name, enabled in path_settings.items() if enabled]

    if enabled_settings:
        logger.warning(
            f"The following data will be recorded and will overwrite existing files in the configured paths: {', '.join(enabled_settings)}."
        )
        input(PRESS_TO_CONTINUE_MSG)


if __name__ == "__main__":
    print(STARTING_EXECUTION_MSG)

    validate_params()
    validate_config()
    warn_overwriting_enabled_paths()

    print(STARTING_SIMULATION_MSG)

    env = simpy.Environment()
    network = initialize_network(env, cfg.BSSs, cfg.NETWORK_BOUNDS_m)

    env.run(until=cfg.SIMULATION_TIME_us)

    print(SIMULATION_TERMINATED_MSG)

    print(EXECUTION_TERMINATED_MSG)
