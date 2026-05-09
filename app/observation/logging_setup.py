import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from app.config.log_config import LogConfig, DEFAULT_LOG_CONFIG


def _get_log_filename(config: LogConfig) -> str:
    os.makedirs(config.log_dir, exist_ok=True)
    date_str = datetime.now().strftime(config.date_format)

    n = 1
    while True:
        filename = os.path.join(config.log_dir, f"{date_str}_{n}.log")
        if not os.path.exists(filename):
            return filename
        n += 1


def setup_logging(config: LogConfig = DEFAULT_LOG_CONFIG) -> None:
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            _get_log_filename(config),
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding=config.encoding,
        )
    ]

    if config.console_output:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        handlers=handlers,
    )