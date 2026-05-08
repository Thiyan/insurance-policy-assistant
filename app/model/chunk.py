from dataclasses import dataclass

from app.model.doc_metadata import DocMetadata


@dataclass
class Chunk:
    id: str
    text: str
    page_number: int
    chunk_index: int
    doc_metadata: DocMetadata

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)