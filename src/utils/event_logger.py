import logging
import json
import os
from src.sim_config import ENABLE_CONSOLE_LOGGING, USE_COLORS_IN_EVENT_LOGS, EXCLUDED_CONSOLE_LEVELS, EXCLUDED_CONSOLE_MODULES, ENABLE_EVENT_RECORDING, EVENT_RECORDING_PATH

COLORS = {
    "HEADER": "\033[95m",     # Magenta
    "DEBUG": "\033[94m",      # Blue
    "INFO": "\033[96m",       # Cyan
    "DEFAULT": "\033[0m",     # Default color
    "SUCCESS": "\033[92m",    # Green
    "WARNING": "\033[38;5;214m",    # Orange
    "ERROR": "\033[91m",      # Red
    "CRITICAL": "\033[1;91m"  # Bold Red
}

HEADER_LEVEL = 5
DEFAULT_LEVEL = 15
SUCCESS_LEVEL = 25

logging.addLevelName(HEADER_LEVEL, "HEADER")
logging.addLevelName(DEFAULT_LEVEL, "DEFAULT")
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def header(self, message, *args, **kwargs):
    if self.isEnabledFor(HEADER_LEVEL):
        self._log(HEADER_LEVEL, message, args, **kwargs)


def default(self, message, *args, **kwargs):
    if self.isEnabledFor(DEFAULT_LEVEL):
        self._log(DEFAULT_LEVEL, message, args, **kwargs)


def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.header = header
logging.Logger.success = success
logging.Logger.default = default

logger_cache = {}


class ConfigFilter(logging.Filter):
    def filter(self, record):
        if not ENABLE_CONSOLE_LOGGING:
            return False
        if record.levelname in EXCLUDED_CONSOLE_LEVELS:
            return False
        if record.name in EXCLUDED_CONSOLE_MODULES:
            return False
        return True


class ConsoleFormatter(logging.Formatter):
    def __init__(self, env=None):
        self.env = env

    def format(self, record):
        record.sim_time = self.env.now if self.env else 0

        if USE_COLORS_IN_EVENT_LOGS:
            log_color = COLORS.get(record.levelname, COLORS["DEFAULT"])
            record.clevelname = f"{log_color}{record.levelname}{COLORS['DEFAULT']}"
            record.cmsg = f"{log_color}{record.msg}{COLORS['DEFAULT']}"
        return f"[t = {record.sim_time:^18.1f}] {record.name:^10} {record.cmsg}"


class FileFormatter(logging.Formatter):
    def __init__(self, env=None):
        super().__init__('%(message)s')  # TODO
        self.env = env

    def format(self, record):
        record.sim_time = self.env.now if self.env else 0
        log_entry = {
            'level': record.levelname,
            'sim_time': record.sim_time,
            'module': record.name,
            'message': record.getMessage(),
        }
        return json.dumps(log_entry)


def get_logger(module_name, env=None):
    if module_name in logger_cache:
        return logger_cache[module_name]

    logger = logging.getLogger(module_name)

    logger.setLevel(HEADER_LEVEL)

    if ENABLE_CONSOLE_LOGGING:
        console_handler = logging.StreamHandler()
        console_formatter = ConsoleFormatter(env)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(ConfigFilter())
        logger.addHandler(console_handler)

    if ENABLE_EVENT_RECORDING:
        if not os.path.exists(EVENT_RECORDING_PATH):
            os.makedirs(EVENT_RECORDING_PATH)

        log_file = os.path.join(EVENT_RECORDING_PATH, f"{module_name}.json")

        file_handler = logging.FileHandler(log_file, mode="w")
        file_formatter = FileFormatter(env)
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(ConfigFilter())
        logger.addHandler(file_handler)

    logger.propagate = False  # Prevent duplicate logs

    logger_cache[module_name] = logger

    return logger


def update_logger_env(env):
    for logger in logger_cache.values():
        for handler in logger.handlers:
            if isinstance(handler.formatter, ConsoleFormatter):
                handler.formatter.env = env
            elif isinstance(handler.formatter, FileFormatter):
                handler.formatter.env = env
