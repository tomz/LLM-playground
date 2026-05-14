"""Bytes → clean unicode text."""
from __future__ import annotations
from .acquire import RawDoc


def extract(doc: RawDoc) -> str | None:
    """Dispatch on MIME. Return None to drop the doc."""
    if doc.mime == "text/html":
        return _html_to_text(doc.payload)
    if doc.mime == "application/pdf":
        return _pdf_to_text(doc.payload)
    if doc.mime.startswith("text/"):
        return doc.payload.decode("utf-8", errors="replace")
    return None


def _html_to_text(b: bytes) -> str | None:
    # real: trafilatura.extract(b, include_comments=False, favor_recall=False)
    raise NotImplementedError


def _pdf_to_text(b: bytes) -> str | None:
    # real: pdfplumber for born-digital, nougat for scanned scientific PDFs
    raise NotImplementedError
