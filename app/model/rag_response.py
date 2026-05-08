from dataclasses import dataclass

from app.model.query_result import QueryResult


@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieval: QueryResult

    @property
    def source_pages(self) -> list[int]:
        return sorted({m.page_number for m in self.retrieval.matches})