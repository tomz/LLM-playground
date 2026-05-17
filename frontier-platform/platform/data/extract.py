"""Bytes → clean unicode text."""
from __future__ import annotations
import re
from .acquire import RawDoc

_TAG_RE = re.compile(rb"<[^>]+>")
_SCRIPT_RE = re.compile(rb"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def extract(doc: RawDoc) -> str | None:
    if doc.mime == "text/html":
        return _html_to_text(doc.payload)
    if doc.mime == "application/pdf":
        return _pdf_to_text(doc.payload)
    if doc.mime.startswith("text/"):
        return doc.payload.decode("utf-8", errors="replace")
    return None


def _html_to_text(b: bytes) -> str | None:
    try:
        import trafilatura  # type: ignore
        out = trafilatura.extract(b, include_comments=False, favor_recall=False)
        if out:
            return out
    except ImportError:
        pass
    # Fallback: drop scripts/styles, strip tags, decode entities, collapse whitespace.
    cleaned = _SCRIPT_RE.sub(b" ", b)
    cleaned = _TAG_RE.sub(b" ", cleaned)
    text = cleaned.decode("utf-8", errors="replace")
    import html
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip() or None


def _pdf_to_text(b: bytes) -> str | None:
    try:
        import pypdf  # type: ignore
    except ImportError as e:
        raise NotImplementedError("install pypdf to extract PDFs") from e
    import io
    reader = pypdf.PdfReader(io.BytesIO(b))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages).strip()
    return text or None
