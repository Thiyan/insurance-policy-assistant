from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")


def get_path(env_name: str, default: str) -> Path:
    return Path(os.getenv(env_name, default)).resolve()


@dataclass(frozen=True)
class AppConfig:
    PDF_PATH: Path = get_path(
        "PDF_PATH",
        str(PROJECT_ROOT / "resources" / "Insurance_Handbook_20103.pdf")
    )

    DB_PATH: Path = get_path(
        "DB_PATH",
        str(PROJECT_ROOT / "data" / "chroma_db")
    )

    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_collection")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    N_RESULTS: int = int(os.getenv("N_RESULTS", 3))

    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5.5")

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-3-small"
    )

    EMBEDDING_DIMS: int = int(os.getenv("EMBEDDING_DIMS", 1536))
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", 100))

    HNSW_SPACE: str = os.getenv("HNSW_SPACE", "cosine")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0))

    SUPPORTED_EXTENSION: str = ".pdf"

    RUN_INGESTION: bool = os.getenv("RUN_INGESTION", "true").lower() == "true"


@dataclass(frozen=True)
class LogConfig:
    LOG_DIR: Path = get_path(
        "LOG_DIR",
        str(PROJECT_ROOT / "logs")
    )

    MAX_BYTES: int = int(
        os.getenv("LOG_MAX_BYTES", 500_000_000)
    )

    BACKUP_COUNT: int = int(
        os.getenv("LOG_BACKUP_COUNT", 3)
    )

    ENCODING: str = "utf-8"

    LEVEL: str = os.getenv(
        "LOG_LEVEL",
        "INFO"
    )

    FORMAT: str = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    DATE_FORMAT: str = "%Y-%m-%d"

    CONSOLE_OUTPUT: bool = (
        os.getenv(
            "LOG_CONSOLE_OUTPUT",
            "true"
        ).lower() == "true"
    )

APP_CONFIG = AppConfig()
DEFAULT_LOG_CONFIG = LogConfig()