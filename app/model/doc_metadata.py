from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocMetadata:
    source: str
    total_pages: int
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: list[str] = field(default_factory=list)
    creation_date: str = ""

    @property
    def filename(self) -> str:
        return Path(self.source).name