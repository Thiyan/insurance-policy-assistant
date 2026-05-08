from dataclasses import dataclass

@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())