from dataclasses import dataclass

@dataclass
class QueryMatch:
    text: str
    source: str
    title: str
    author: str
    page_number: int
    total_pages: int
    chunk_index: int

    @property
    def preview(self, length: int = 200) -> str:
        return self.text[:length] + ("..." if len(self.text) > length else "")