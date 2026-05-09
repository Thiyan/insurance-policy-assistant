from dataclasses import dataclass

from app.config.config import APP_CONFIG
from app.model.chunk import Chunk


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]

    def __post_init__(self) -> None:
        if len(self.embedding) != APP_CONFIG.EMBEDDING_DIMS:
            raise ValueError(
                f"Expected {APP_CONFIG.EMBEDDING_DIMS}-dim embedding, got {len(self.embedding)}"
            )
