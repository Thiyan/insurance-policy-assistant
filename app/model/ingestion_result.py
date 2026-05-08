from dataclasses import dataclass

from app.model.doc_metadata import DocMetadata
from app.model.stored_result import StoreResult


@dataclass
class IngestionResult:
    doc_metadata: DocMetadata
    pages_extracted: int
    chunks_created: int
    store_result: StoreResult
    elapsed_seconds: float

    @property
    def summary(self) -> str:
        return (
            f"Ingested '{self.doc_metadata.title or self.doc_metadata.filename}' — "
            f"{self.pages_extracted} pages, "
            f"{self.chunks_created} chunks, "
            f"{self.elapsed_seconds:.1f}s"
        )