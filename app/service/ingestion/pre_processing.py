import logging
import re
from pathlib import Path

import fitz
import pymupdf4llm

from app.config.config import SUPPORTED_EXTENSION
from app.exception.sub_exception.ingestion_exception import DocumentParseException, FileNotFoundException, \
    UnsupportedFileTypeException
from app.model.doc_metadata import DocMetadata
from app.model.page_content import PageContent

logger = logging.getLogger(__name__)


def extract_doc_metadata(pdf_path: str) -> DocMetadata:
    """Extract metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocMetadata dataclass populated with the PDF's metadata.

    Raises:
        FileNotFoundException:        File does not exist at the given path.
        UnsupportedFileTypeException: File is not a .pdf.
        DocumentParseException:       File is corrupted, encrypted, or unreadable.
    """
    path = _validate_pdf_path(pdf_path)
    logger.info("Extracting metadata from '%s'", path.name)

    try:
        with fitz.open(str(path)) as doc:
            if doc.is_encrypted:
                raise DocumentParseException(
                    file_name=path.name,
                    reason="PDF is encrypted — decrypt it before ingestion",
                )

            raw = doc.metadata or {}
            metadata = DocMetadata(
                source=str(path.resolve()),
                title=_clean(raw.get("title")),
                author=_clean(raw.get("author")),
                subject=_clean(raw.get("subject")),
                keywords=_parse_keywords(raw.get("keywords") or ""),
                creation_date=_clean(raw.get("creationDate")),
                total_pages=doc.page_count,
            )

    except fitz.FileDataError as exc:
        raise DocumentParseException(
            file_name=path.name,
            reason=f"File is corrupted or not a valid PDF: {exc}",
        ) from exc
    except Exception as exc:
        raise DocumentParseException(
            file_name=path.name,
            reason=f"Unexpected error reading metadata: {exc}",
        ) from exc

    logger.info("Metadata extracted from '%s' (%d pages)", path.name, metadata.total_pages)
    return metadata


def extract_page_markdowns(pdf_path: str) -> list[PageContent]:
    """Extract non-empty Markdown content per page from a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageContent for each non-empty page, in order.
        Returns an empty list if the PDF has no extractable text (e.g. scanned).

    Raises:
        FileNotFoundException:        File does not exist at the given path.
        UnsupportedFileTypeException: File is not a .pdf.
        DocumentParseException:       File is corrupted, encrypted, or conversion failed.
    """
    path = _validate_pdf_path(pdf_path)
    logger.info("Extracting page content from '%s'", path.name)

    try:
        raw_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    except Exception as exc:
        raise DocumentParseException(
            file_name=path.name,
            reason=f"pymupdf4llm conversion failed: {exc}",
        ) from exc

    pages = [
        PageContent(
            page_number=_resolve_page_number(page, fallback_index=i),
            text=text,
        )
        for i, page in enumerate(raw_pages)
        if (text := page.get("text", "").strip())
    ]

    if not pages:
        logger.warning("No text extracted from '%s' — PDF may be image-only", path.name)
    else:
        logger.info("Extracted %d page(s) with text from '%s'", len(pages), path.name)

    return pages


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_pdf_path(pdf_path: str) -> Path:
    """Check existence and extension before opening."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundException(file_path=str(path))

    if path.suffix.lower() != SUPPORTED_EXTENSION:
        raise UnsupportedFileTypeException(
            file_type=path.suffix or "<no extension>",
            supported=[SUPPORTED_EXTENSION],
        )

    return path


def _clean(value: object) -> str:
    """Coerce a raw metadata value to a stripped string, normalising None and literal 'none'."""
    if value is None:
        return ""
    cleaned = str(value).strip()
    return "" if cleaned.lower() == "none" else cleaned


def _parse_keywords(raw: str) -> list[str]:
    """Split a comma- or semicolon-separated keywords string into a clean list."""
    if not raw:
        return []
    return [kw.strip() for kw in re.split(r"[,;]", raw) if kw.strip()]


def _resolve_page_number(page: dict, fallback_index: int) -> int:
    """Resolve a 1-based page number from a pymupdf4llm page chunk dict.

    Args:
        page:           Raw page chunk dict from pymupdf4llm.
        fallback_index: 0-based index from enumerate() used as last resort.

    Returns:
        Resolved page number, always >= 1.
    """
    meta = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}

    for candidate in (meta.get("page"), meta.get("page_number"), page.get("page")):
        if isinstance(candidate, int) and candidate > 0:
            return candidate

    return fallback_index
