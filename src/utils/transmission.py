from src.sim_params import SimParams as sparams
from src.utils.mcs_table import calculate_data_rate_bps


import math
import random


def get_path_loss_dB(sparams: sparams, distance_m: float) -> float:
    """
    Calculate the path loss (in dB) given the distance (in meters) according to the free-space log-distance path loss model.

    Args:
        sparams (sparams): The SimulationParams object.
        distance_m (float): The distance in meters.

    Returns:
        float: The path loss in dB.
    """
    # Path loss at reference distance (1 meter) assuming free space
    path_loss_1m_dB = (
        20 * math.log10(1) + 20 * math.log10(sparams.FREQUENCY_GHz * 1e9) - 147.55 - sparams.TX_GAIN_dB - sparams.RX_GAIN_dB
    )
    # Path loss at the given distance using the free-space log-distance path loss model
    path_loss = path_loss_1m_dB + 10 * sparams.PATH_LOSS_EXPONENT * math.log10(
        max(distance_m, 0.1) / 1
    )

    # Add a random shadowing component if enabled
    if sparams.ENABLE_SHADOWING:
        shadowing_dB = random.gauss(0, sparams.SHADOWING_STD_dB)
        path_loss += shadowing_dB

    return path_loss


def get_tx_duration_us(
    sparams: sparams,
    mcs_index: int,
    channel_width: int,
    size_bytes: int,
    is_mgmt_ctrl_frame: bool = False,
) -> int:
    """
    Calculates the transmission duration (in microseconds) for a given packet size and MCS.

    Args:
        sparams (sparams): The SimulationParams object.
        mcs_index (int): The MCS index.
        channel_width (int): The channel width in MHz.
        size_bytes (int): The size of the packet in bytes.
        is_mgmt_ctrl_frame (bool, optional): Whether the packet is a management/control frame. Defaults to False.

    Returns:
        int: The transmission duration in microseconds.
    """
    # Calculate the data rate in bits per second
    data_rate_bps = calculate_data_rate_bps(
        mcs_index,
        channel_width,
        sparams.SPATIAL_STREAMS if not is_mgmt_ctrl_frame else 1,
        sparams.GUARD_INTERVAL_us if not is_mgmt_ctrl_frame else 0.8,
    )

    # Calculate the transmission duration in microseconds
    tx_duration_us = size_bytes * 8 / data_rate_bps * 1e6

    # Round the result to the nearest integer
    return round(tx_duration_us)


def get_rssi_dbm(sparams: sparams, distance_m: float) -> float:
    """
    Calculates the RSSI in dBm for a given distance in meters.

    Args:
        sparams (sparams): The SimulationParams object.
        distance_m (float): The distance in meters.

    Returns:
        float: The RSSI in dBm.
    """
    # Calculate the RSSI in dBm
    rssi_dbm = (
        sparams.TX_POWER_dBm
        + sparams.TX_GAIN_dB
        + sparams.RX_GAIN_dB
        - get_path_loss_dB(sparams, distance_m)
    )

    return rssi_dbm
