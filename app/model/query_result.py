from dataclasses import dataclass

from app.model.query_match import QueryMatch


@dataclass
class QueryResult:
    question: str
    matches: list[QueryMatch]

    @property
    def top(self) -> QueryMatch | None:
        return self.matches[0] if self.matches else None