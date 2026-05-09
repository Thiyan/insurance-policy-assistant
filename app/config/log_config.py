from dataclasses import dataclass


@dataclass
class LogConfig:
    log_dir: str = "logs"
    max_bytes: int = 500_000_000
    backup_count: int = 3
    encoding: str = "utf-8"
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d"
    console_output: bool = True


# Default config — import and override fields as needed
DEFAULT_LOG_CONFIG = LogConfig()