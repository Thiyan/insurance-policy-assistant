from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.model.chunk import Chunk
from app.model.doc_metadata import DocMetadata
from app.model.page_content import PageContent

def chunk_pages(
    pages: list[PageContent],
    doc_metadata: DocMetadata,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split pages into overlapping text chunks enriched with metadata.

    Args:
        pages: Extracted page contents to chunk.
        doc_metadata: Document-level metadata attached to every chunk.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        Ordered list of Chunk objects across all pages.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [
        _make_chunk(sub_chunk, page.page_number, chunk_index, doc_metadata)
        for chunk_index, (page, sub_chunk) in enumerate(
            (page, sub_chunk)
            for page in pages
            for sub_chunk in splitter.split_text(page.text)
            if sub_chunk.strip()
        )
    ]


def _make_chunk(
    text: str,
    page_number: int,
    chunk_index: int,
    doc_metadata: DocMetadata,
) -> Chunk:
    return Chunk(
        id=f"chunk_{chunk_index}",
        text=text,
        page_number=page_number,
        chunk_index=chunk_index,
        doc_metadata=doc_metadata,
    )