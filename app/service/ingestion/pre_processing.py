from pathlib import Path

import fitz
import pymupdf4llm

from app.model.doc_metadata import DocMetadata
from app.model.page_content import PageContent


def extract_doc_metadata(pdf_path: str) -> DocMetadata:
    """Extract metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocMetadata dataclass populated with the PDF's metadata.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        fitz.FileDataError: If the file is not a valid PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        raw = doc.metadata
        return DocMetadata(
            source=str(path.resolve()),
            title=raw.get("title") or "",
            author=raw.get("author") or "",
            subject=raw.get("subject") or "",
            keywords=_parse_keywords(raw.get("keywords") or ""),
            creation_date=raw.get("creationDate") or "",
            total_pages=doc.page_count,
        )

def extract_page_markdowns(pdf_path: str) -> list[PageContent]:
    """Extract non-empty Markdown content per page from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageContent for each non-empty page, in order.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    raw_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    return [
        PageContent(
            page_number=_resolve_page_number(page, i),
            text=text,
        )
        for i, page in enumerate(raw_pages)
        if (text := page.get("text", "").strip())
    ]

def _parse_keywords(raw: str) -> list[str]:
    """Split a comma- or semicolon-separated keywords string into a clean list."""
    if not raw:
        return []
    import re
    return [kw.strip() for kw in re.split(r"[,;]", raw) if kw.strip()]

def _resolve_page_number(page: dict, fallback_index: int) -> int:
    """Resolve page number across pymupdf4llm metadata schema versions.

    Args:
        page: Raw page dict from pymupdf4llm.
        fallback_index: 0-based enumeration index used as last resort.

    Returns:
        The resolved page number (1-indexed).
    """
    meta = page.get("metadata", {})
    return (
        meta.get("page")
        or meta.get("page_number")
        or page.get("page")
        or fallback_index
    )
