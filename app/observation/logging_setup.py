import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from app.config.config import DEFAULT_LOG_CONFIG, LogConfig


def _get_log_filename(config: LogConfig) -> str:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    date_str = datetime.now().strftime(config.DATE_FORMAT)

    n = 1
    while True:
        filename = os.path.join(config.LOG_DIR, f"{date_str}_{n}.log")
        if not os.path.exists(filename):
            return filename
        n += 1


def setup_logging(config: LogConfig = DEFAULT_LOG_CONFIG) -> None:
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            _get_log_filename(config),
            maxBytes=config.MAX_BYTES,
            backupCount=config.BACKUP_COUNT,
            encoding=config.ENCODING,
        )
    ]

    if config.CONSOLE_OUTPUT:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=getattr(logging, config.LEVEL.upper()),
        format=config.FORMAT,
        handlers=handlers,
    )