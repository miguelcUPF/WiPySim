from tests._user_config_tests import UserConfig as cfg_module
from tests._sim_params_tests import SimParams as sparams_module


from src.utils.mcs_table import calculate_data_rate_bps
from src.utils.event_logger import get_logger
from src.utils.messages import STARTING_TEST_MSG, TEST_COMPLETED_MSG

# Each test case is a tuple: (mcs_index, channel_width, num_streams, guard_interval, expected_data_rate)
TEST_CASES = [
    (9, 80, 2, 0.8, 960.8),
    (2, 40, 1, 1.6, 48.8),
    (7, 20, 1, 0.8, 86.0),
    (10, 160, 2, 3.2, 1837.5),
    (3, 40, 2, 0.8, 137.6),
    (5, 80, 3, 1.6, 816.7),
]

if __name__ == "__main__":
    print(STARTING_TEST_MSG)

    logger = get_logger("TEST", cfg_module, sparams_module)

    for (
        mcs_index,
        channel_width,
        num_streams,
        guard_interval,
        expected_data_rate,
    ) in TEST_CASES:
        data_rate = calculate_data_rate_bps(
            mcs_index, channel_width, num_streams, guard_interval
        )

        logger.debug(
            f"Testing with MCS {mcs_index}, {num_streams} SS, {channel_width} MHz, {guard_interval} μs GI"
        )

        logger.info(f"Calculated data rate: {round(data_rate / 1e6, 1)} Mbps")

        assert round(data_rate / 1e6, 1) == expected_data_rate, logger.error(
            f"Expected {expected_data_rate} Mbps, got {round(data_rate / 1e6, 1)} Mbps"
        )

    print(TEST_COMPLETED_MSG)
