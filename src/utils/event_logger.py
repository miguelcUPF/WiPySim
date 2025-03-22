from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.utils.messages import EXECUTION_TERMINATED_MSG


from datetime import datetime

import logging
import json
import os


def header(self, message, *args, **kwargs):
    if self.isEnabledFor(HEADER_LEVEL):
        self._log(HEADER_LEVEL, message, args, **kwargs)


def default(self, message, *args, **kwargs):
    if self.isEnabledFor(DEFAULT_LEVEL):
        self._log(DEFAULT_LEVEL, message, args, **kwargs)


def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


def _initialize_log_file(cfg: cfg, log_file: str):
    if not os.path.exists(cfg.LOGS_RECORDING_PATH):
        os.makedirs(cfg.LOGS_RECORDING_PATH, exist_ok=True)
    creation_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logs_data = {"creation_time": creation_time, "logs": []}

    with open(log_file, "w") as f:
        json.dump(logs_data, f, indent=4)


LOGS_RECORDING_FILE = os.path.join(
    cfg.LOGS_RECORDING_PATH, "session_logs.json"
)  # get_unique_filename(LOGS_RECORDING_PATH, "session_logs", "json")

_initialize_log_file(cfg, LOGS_RECORDING_FILE)

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

logging.Logger.header = header
logging.Logger.success = success
logging.Logger.default = default

LOGGER_CACHE = {}


class ConfigFilter(logging.Filter):
    def __init__(self, cfg: cfg, sparams: sparams):
        super().__init__()
        self.cfg = cfg
        self.sparams = sparams

    def filter(self, record):
        if record.levelname in self.cfg.EXCLUDED_LOGS.get(
            record.name, []
        ) or "ALL" in self.cfg.EXCLUDED_LOGS.get(record.name, []):
            return False
        return True


class ConsoleFormatter(logging.Formatter):
    def __init__(self, cfg: cfg, sparams: sparams, env=None):
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

    def format(self, record):
        record.sim_time = self.env.now if self.env else 0

        if self.cfg.USE_COLORS_IN_LOGS:
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
    def __init__(self, filename, mode="a", encoding=None, env=None):
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
        with open(self.filename, "r+") as file:
            existing_data = json.load(file)
            existing_data["logs"].append(json.loads(log_entry))
            file.seek(0)
            json.dump(existing_data, file, indent=4)
            file.truncate()


def get_logger(module_name: str, cfg: cfg, sparams: sparams, env=None) -> logging.Logger:
    if module_name in LOGGER_CACHE:
        return LOGGER_CACHE[module_name]

    logger = logging.getLogger(module_name)

    logger.setLevel(HEADER_LEVEL)

    if not cfg.ENABLE_CONSOLE_LOGGING and not cfg.ENABLE_LOGS_RECORDING and module_name not in ["MAIN", "TEST"]:
        logger.addHandler(logging.NullHandler())
        return logger

    if cfg.ENABLE_CONSOLE_LOGGING or module_name in ["MAIN", "TEST"]:
        console_handler = logging.StreamHandler()
        console_formatter = ConsoleFormatter(cfg, sparams, env)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(ConfigFilter(cfg=cfg, sparams=sparams))
        logger.addHandler(console_handler)

    if cfg.ENABLE_LOGS_RECORDING:
        if not os.path.exists(cfg.LOGS_RECORDING_PATH):
            os.makedirs(cfg.LOGS_RECORDING_PATH)

        file_handler = JSONFileHandler(LOGS_RECORDING_FILE, mode="a", env=env)
        file_handler.setFormatter((logging.Formatter("%(message)s")))
        file_handler.addFilter(ConfigFilter(cfg=cfg, sparams=sparams))
        logger.addHandler(file_handler)

    logger.propagate = False  # Prevent duplicate logs

    LOGGER_CACHE[module_name] = logger

    return logger


def update_logger_env(env):
    for logger in LOGGER_CACHE.values():
        for handler in logger.handlers:
            if isinstance(handler.formatter, ConsoleFormatter):
                handler.formatter.env = env
            elif isinstance(handler, JSONFileHandler):
                handler.env = env
