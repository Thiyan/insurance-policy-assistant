from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str


class MatchedChunk(BaseModel):
    text: str
    page_number: int
    source: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    page_numbers: list[int]
    matched_chunks: list[MatchedChunk]