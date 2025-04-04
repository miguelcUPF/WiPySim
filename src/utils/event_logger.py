from src.sim_params import SimParams as sparams
from src.user_config import UserConfig as cfg

from src.utils.messages import EXECUTION_TERMINATED_MSG

from datetime import datetime
import logging
import simpy
import json
import os


HEADER_LEVEL = 5
DEFAULT_LEVEL = 15
SUCCESS_LEVEL = 25

# Define color codes for different log levels
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


logging.addLevelName(HEADER_LEVEL, "HEADER")
logging.addLevelName(DEFAULT_LEVEL, "DEFAULT")
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

LOGGER_CACHE = {}  # Dictionary to cache loggers
ALWAYS_INCLUDED_MODULES = ["MAIN", "TEST"]


def header(self, message: str, *args, **kwargs):
    """
    Log a message with log level HEADER (5).

    HEADER is a special log level that is used to log DEBUG messages that are
    intended to be displayed as a header or a title.
    """
    if self.isEnabledFor(HEADER_LEVEL):
        self._log(HEADER_LEVEL, message, args, **kwargs)


def default(self, message: str, *args, **kwargs):
    """
    Log a message with log level DEFAULT (15).

    DEFAULT is a special log level that is used to log messages that are
    intended to be displayed as regular log messages.
    """
    if self.isEnabledFor(DEFAULT_LEVEL):
        self._log(DEFAULT_LEVEL, message, args, **kwargs)


def success(self, message: str, *args, **kwargs):
    """
    Log a message with log level SUCCESS (25).

    SUCCESS is a special log level that is used to log messages that are
    intended to be displayed as successful events.
    """
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.header = header
logging.Logger.success = success
logging.Logger.default = default


def initialize_log_file(cfg: cfg, log_file: str) -> None:
    """
    Initialize a log file in the specified path.

    Args:
        cfg (cfg): The UserConfig object.
        log_file (str): The path to the log file.
    """
    if not os.path.exists(cfg.LOGS_RECORDING_PATH):
        os.makedirs(cfg.LOGS_RECORDING_PATH, exist_ok=True)

    creation_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    log_data = {"creation_time": creation_time, "logs": []}

    with open(log_file, "w") as file:
        json.dump(log_data, file, indent=4)


LOGS_RECORDING_FILE = os.path.join(cfg.LOGS_RECORDING_PATH, "session_logs.json")
initialize_log_file(cfg, LOGS_RECORDING_FILE)


class ConfigFilter(logging.Filter):
    """Filters log messages based on configuration settings."""

    def __init__(self, cfg: cfg, sparams: sparams):
        """
        Initialize a ConfigFilter object.

        Args:
            cfg (cfg): The UserConfig object.
            sparams (sparams): The SimulationParams object.
        """
        super().__init__()
        self.cfg = cfg
        self.sparams = sparams

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log messages based on configuration settings.

        Excludes log messages if the log level is in the excluded logs list
        of the log record's name.

        Args:
            record (logging.LogRecord): The log record to filter.

        Returns:
            bool: True if the log record passes the filter, False otherwise.
        """
        excluded_logs = self.cfg.EXCLUDED_LOGS.get(record.name, [])
        if record.levelname in excluded_logs or "ALL" in excluded_logs:
            return False
        return True


class ConsoleFormatter(logging.Formatter):
    """Custom formatter for console with optional color formatting."""

    def __init__(self, cfg: cfg, sparams: sparams, env: simpy.Environment = None):
        """
        Initialize a ConsoleFormatter object.

        Args:
            cfg (cfg): The UserConfig object.
            sparams (sparams): The SimulationParams object.
            env (simpy.Environment, optional): The simulation environment. Defaults to None.
        """
        self.cfg = cfg
        self.sparams = sparams
        self.env = env

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record into a string.

        Includes the log record's name, log level, log message, and the current
        simulation time if the simulation environment is provided.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record string.
        """
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
    """Custom handler for logging to a JSON file."""

    def __init__(self, filename: str, env: simpy.Environment = None):
        """
        Initialize a JSONFileHandler object.

        Args:
            filename (str): The path to the JSON log file.
            env (simpy.Environment, optional): The simulation environment. Defaults to None.
        """

        super().__init__()
        self.filename = filename
        self.env = env

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record into a JSON string.

        Includes the log record's level, simulation time, module name, and log message.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record JSON string.
        """
        log_entry = {
            "level": record.levelname,
            "sim_time": record.sim_time if self.env else 0,
            "module": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_entry)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Log a record to the JSON file.

        Args:
            record (logging.LogRecord): The log record to log.
        """
        formatted_record = self.format(record)
        with open(self.filename, "r+") as file:
            data = json.load(file)
            data["logs"].append(json.loads(formatted_record))
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()


def get_logger(
    module_name: str, cfg: cfg, sparams: sparams, env: simpy.Environment = None
) -> logging.Logger:
    """
    Get a logger instance with the specified module name.

    The logger is configured according to the provided UserConfig and SimulationParams objects.

    Args:
        module_name (str): The name of the logger instance.
        cfg (cfg): The UserConfig object.
        sparams (sparams): The SimulationParams object.
        env (simpy.Environment, optional): The simulation environment. Defaults to None.

    Returns:
        logging.Logger: The logger instance.
    """

    if module_name in LOGGER_CACHE:
        return LOGGER_CACHE[module_name]

    logger = logging.getLogger(module_name)
    logger.setLevel(HEADER_LEVEL)

    if (
        not cfg.ENABLE_CONSOLE_LOGGING
        and not cfg.ENABLE_LOGS_RECORDING
        and module_name not in ALWAYS_INCLUDED_MODULES
    ):
        logger.addHandler(logging.NullHandler())
        return logger

    if cfg.ENABLE_CONSOLE_LOGGING or module_name in ALWAYS_INCLUDED_MODULES:
        console_handler = logging.StreamHandler()
        console_formatter = ConsoleFormatter(cfg=cfg, sparams=sparams, env=env)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(ConfigFilter(cfg=cfg, sparams=sparams))
        logger.addHandler(console_handler)

    if cfg.ENABLE_LOGS_RECORDING:
        if not os.path.exists(cfg.LOGS_RECORDING_PATH):
            os.makedirs(cfg.LOGS_RECORDING_PATH)

        file_handler = JSONFileHandler(LOGS_RECORDING_FILE, mode="a", env=env)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler.addFilter(ConfigFilter(cfg=cfg, sparams=sparams))
        logger.addHandler(file_handler)

    logger.propagate = False

    LOGGER_CACHE[module_name] = logger

    return logger


def update_loggers_environment(env):
    """Update the environment attribute of all loggers in the cache."""
    for logger in LOGGER_CACHE.values():
        for handler in logger.handlers:
            if isinstance(handler.formatter, ConsoleFormatter):
                handler.formatter.env = env
            elif isinstance(handler, JSONFileHandler):
                handler.env = env
