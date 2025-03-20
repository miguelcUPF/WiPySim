from src.user_config import (
    ENABLE_CONSOLE_LOGGING,
    USE_COLORS_IN_LOGS,
    EXCLUDED_LOGS,
    ENABLE_LOGS_RECORDING,
    LOGS_RECORDING_PATH,
)
from src.utils.messages import EXECUTION_TERMINATED_MSG
from src.utils.file_manager import get_unique_filename

from datetime import datetime

import logging
import json
import os

def initialize_log_file(log_file: str):
    if not os.path.exists(LOGS_RECORDING_PATH):
            os.makedirs(LOGS_RECORDING_PATH)
    creation_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logs_data = {
        "creation_time": creation_time,
        "logs": []
    }

    with open(log_file, 'w') as f:
        json.dump(logs_data, f, indent=4)


LOGS_RECORDING_FILE = os.path.join(LOGS_RECORDING_PATH, "session_logs.json") # get_unique_filename(LOGS_RECORDING_PATH, "session_logs", "json")
initialize_log_file(LOGS_RECORDING_FILE)

COLORS = {
    "HEADER": "\033[95m",  # Magenta
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[96m",  # Cyan
    "DEFAULT": "\033[0m",  # Default color
    "SUCCESS": "\033[92m",  # Green
    "WARNING": "\033[38;5;214m",  # Orange
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[1;38;5;1m",  # Bold Dark Red
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
        if record.levelname in EXCLUDED_LOGS.get(
            record.name, []
        ) or "ALL" in EXCLUDED_LOGS.get(record.name, []):
            return False
        return True


class ConsoleFormatter(logging.Formatter):
    def __init__(self, env=None):
        self.env = env

    def format(self, record):
        record.sim_time = self.env.now if self.env else 0

        if USE_COLORS_IN_LOGS:
            log_color = COLORS.get(record.levelname, COLORS["DEFAULT"])
            record.clevelname = f"{log_color}{record.levelname}{COLORS["DEFAULT"]}"
            record.cmsg = f"{log_color}{record.msg}{COLORS["DEFAULT"]}"
        else:
            record.clevelname = record.levelname
            record.cmsg = record.msg

        if record.levelname == "CRITICAL":
            formatted_message = f"{record.levelname}: {record.cmsg}"
            formatted_message += "\n" + EXECUTION_TERMINATED_MSG
            raise SystemExit(formatted_message)

        formatted_message = (
            f"[t = {record.sim_time:^18.1f}] {record.name:^10} {record.cmsg}"
            if self.env
            else f"{record.levelname}: {record.cmsg}"
        )
        return formatted_message

class JSONFileHandler(logging.Handler):
    def __init__(self, filename, mode='a', encoding=None, env=None):
        super().__init__()
        self.filename = filename
        self.env = env

    def format(self, record):
        record.sim_time = self.env.now if self.env else 0
        log_entry = {
            "level": record.levelname,
            "sim_time": record.sim_time,
            "module": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_entry)

    def emit(self, record):
        log_entry = self.format(record)
        with open(self.filename, 'r+') as file:
            existing_data = json.load(file)
            existing_data["logs"].append(json.loads(log_entry))
            file.seek(0)
            json.dump(existing_data, file, indent=4)
            file.truncate()

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

    if ENABLE_LOGS_RECORDING:
        if not os.path.exists(LOGS_RECORDING_PATH):
            os.makedirs(LOGS_RECORDING_PATH)

        file_handler = JSONFileHandler(LOGS_RECORDING_FILE, mode="a", env=env)
        file_handler.setFormatter((logging.Formatter('%(message)s')))
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
