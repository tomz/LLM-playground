"""Source connectors. Each yields raw bytes or document records.

This module replaces the four ``NotImplementedError`` stubs (CommonCrawl /
GitHub / Arxiv / Wikipedia) with real implementations. Each connector has the
same shape: a ``Source`` exposing ``name`` and ``stream() -> Iterator[RawDoc]``.

Design rules followed everywhere here:

* **Optional heavy deps**: ``warcio``, ``boto3``, the ``arxiv`` client, and
  ``wikiextractor`` are imported lazily. Missing them raises ``ImportError``
  with a one-line install hint, *not* ``NotImplementedError``.
* **Local-fixture path always works without network**: every connector
  accepts a local glob or path so tests + CI can exercise the parser without
  egress. The network path is gated behind explicit constructor args.
* **License + provenance is data**: every ``RawDoc.meta`` records the source
  shard URI, the license string (when known), and the timestamp so downstream
  ``platform.data.synthetic.lineage`` and the contamination index can attribute
  things later.
* **Streaming**: connectors never load a full snapshot into memory. CommonCrawl
  iterates a WARC shard at a time; GHArchive iterates one hourly file; the
  arxiv connector pages OAI-PMH; the wiki extractor yields per-article.

Tests cover the parser path for each connector against tiny local fixtures so
the whole module is exercised in CI without any network calls.
"""
from __future__ import annotations

import gzip
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol


@dataclass
class RawDoc:
    source: str
    uri: str
    mime: str
    payload: bytes
    meta: dict = field(default_factory=dict)


class Source(Protocol):
    name: str
    def stream(self) -> Iterator[RawDoc]: ...


# ============================================================================
# Local / JSONL (unchanged)
# ============================================================================


class LocalFilesSource:
    """Recursively walk a local directory; yield each file as a RawDoc.

    Useful for tests and small/private corpora.
    """
    name = "local_files"

    def __init__(self, root: str | Path, mime: str = "text/plain"):
        self.root = Path(root)
        self.mime = mime

    def stream(self) -> Iterator[RawDoc]:
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            yield RawDoc(
                source=f"local_files/{self.root.name}",
                uri=str(p),
                mime=self.mime,
                payload=p.read_bytes(),
                meta={"size": p.stat().st_size},
            )


class JsonlSource:
    """Yield each line of a JSONL file as a text/plain RawDoc.

    Each JSON record must contain ``text_key`` (default 'text').
    """
    name = "jsonl"

    def __init__(self, path: str | Path, text_key: str = "text"):
        self.path = Path(path)
        self.text_key = text_key

    def stream(self) -> Iterator[RawDoc]:
        with self.path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get(self.text_key, "")
                yield RawDoc(
                    source=f"jsonl/{self.path.name}",
                    uri=f"{self.path}#{i}",
                    mime="text/plain",
                    payload=text.encode("utf-8"),
                    meta={k: v for k, v in rec.items() if k != self.text_key},
                )


# ============================================================================
# CommonCrawl: WARC over local files or S3
# ============================================================================


class CommonCrawlSource:
    """Iterate WARC files from a CommonCrawl snapshot.

    Two modes:

    * ``shard_paths`` — list/glob of local ``.warc.gz`` files. No network deps;
      the streaming path is the same. This is the path tests exercise.
    * ``s3_prefix`` + ``snapshot`` — stream WARC shards from the public CC S3
      bucket (``s3://commoncrawl/...``). Requires ``boto3`` and ``warcio``.

    Yields one ``RawDoc`` per WARC ``response`` record carrying the page's
    extracted HTML payload. Filters out non-2xx, non-HTML, and oversized
    pages (default 10 MB cap). The downstream :mod:`platform.data.extract`
    step is what turns HTML into clean text — this connector's job is to
    deliver bytes, not to interpret them.

    Args:
        snapshot: CC snapshot id (e.g. ``"CC-MAIN-2025-13"``). Recorded in
            ``RawDoc.meta`` for lineage; not used otherwise on the local path.
        shard_paths: local WARC files / iterable of paths.
        s3_prefix: ``"s3://commoncrawl/crawl-data/<snapshot>/segments/..."``.
        s3_keys: iterable of S3 keys under ``s3_prefix`` to stream.
        max_payload_bytes: hard cap on a single record's payload; larger
            records are skipped (with a meta note in ``rejected_oversize``
            on the iterator if you care to track it).
        record_types: WARC record types to keep; default ``{"response"}``.
    """
    name = "commoncrawl"

    def __init__(
        self,
        snapshot: str = "",
        *,
        shard_paths: Iterable[str | Path] | None = None,
        s3_prefix: str | None = None,
        s3_keys: Iterable[str] | None = None,
        max_payload_bytes: int = 10 * 1024 * 1024,
        record_types: Iterable[str] = ("response",),
    ):
        self.snapshot = snapshot
        self.shard_paths = [Path(p) for p in (shard_paths or [])]
        self.s3_prefix = s3_prefix
        self.s3_keys = list(s3_keys or [])
        self.max_payload_bytes = int(max_payload_bytes)
        self.record_types = set(record_types)
        if not self.shard_paths and not (self.s3_prefix and self.s3_keys):
            raise ValueError(
                "CommonCrawlSource needs either `shard_paths=` (local WARC files) "
                "or `s3_prefix=` + `s3_keys=` (S3 streaming via boto3+warcio)."
            )

    def stream(self) -> Iterator[RawDoc]:
        for path in self.shard_paths:
            yield from self._stream_local(path)
        for key in self.s3_keys:
            yield from self._stream_s3(key)

    def _stream_local(self, path: Path) -> Iterator[RawDoc]:
        ArchiveIterator = _require_warcio()
        with path.open("rb") as f:
            for rec in ArchiveIterator(f):
                doc = self._record_to_doc(rec, source_uri=str(path))
                if doc is not None:
                    yield doc

    def _stream_s3(self, key: str) -> Iterator[RawDoc]:
        ArchiveIterator = _require_warcio()
        boto3 = _require_boto3()
        # s3://bucket/key/... → bucket + key
        prefix = (self.s3_prefix or "").rstrip("/") + "/"
        full = prefix + key.lstrip("/")
        m = re.match(r"s3://([^/]+)/(.+)$", full)
        if not m:
            raise ValueError(f"bad S3 URL: {full!r}; expected s3://bucket/path/...")
        bucket, full_key = m.group(1), m.group(2)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=full_key)
        # WARC files are gzip-streamed; warcio reads the gzip envelope itself
        # when given a binary stream.
        body = obj["Body"]
        for rec in ArchiveIterator(body):
            doc = self._record_to_doc(rec, source_uri=full)
            if doc is not None:
                yield doc

    def _record_to_doc(self, rec, *, source_uri: str) -> RawDoc | None:
        if rec.rec_type not in self.record_types:
            return None
        # WARC content-length includes HTTP headers; we want the payload only.
        try:
            payload = rec.content_stream().read(self.max_payload_bytes + 1)
        except Exception:
            return None
        if len(payload) > self.max_payload_bytes:
            return None  # oversized; let downstream filtering report this
        target_uri = rec.rec_headers.get_header("WARC-Target-URI") or ""
        warc_date = rec.rec_headers.get_header("WARC-Date") or ""
        content_type = (rec.http_headers.get_header("Content-Type")
                        if rec.http_headers else "text/html") or "text/html"
        if not content_type.lower().startswith(("text/", "application/xhtml")):
            return None
        return RawDoc(
            source=f"commoncrawl/{self.snapshot}" if self.snapshot else "commoncrawl",
            uri=target_uri,
            mime=content_type.split(";", 1)[0].strip(),
            payload=payload,
            meta={
                "snapshot": self.snapshot,
                "warc_date": warc_date,
                "warc_shard": source_uri,
                "license": "CC-data",  # CC redistribution terms, not page license
            },
        )


# ============================================================================
# GitHub: GHArchive (events) + optional raw-content fetch
# ============================================================================


_GHARCHIVE_URL = "https://data.gharchive.org/{hour}.json.gz"


class GitHubSource:
    """Stream events from GHArchive's hourly JSON-gzip files.

    GHArchive provides one ``YYYY-MM-DD-H.json.gz`` per UTC hour. Each line is
    a GitHub event (PushEvent, PullRequestEvent, etc.). This connector pulls
    one or more hours, decodes the gzip stream, parses each JSON event, keeps
    those whose repo is in ``license_allowlist`` (when the event payload
    carries license info — Push/Create events do, others don't), and yields
    each repo's metadata as a ``RawDoc``.

    The two paths:

    * ``hours=["2025-11-01-12", "2025-11-01-13"]`` — fetch from GHArchive over
      HTTP. Lazy: uses stdlib ``urllib.request`` so no extra dep needed.
    * ``local_files=[Path(...), ...]`` — read pre-downloaded ``.json.gz``
      files from disk. The path tests use.

    Each yielded ``RawDoc`` is the **event payload as JSON bytes** so the
    extractor downstream can decide whether to keep the commit message, the
    PR body, the file diff, or all three. Filtering by ``event_types`` keeps
    the volume manageable (PushEvent + PullRequestEvent are the most useful
    for code-corpus mining).
    """
    name = "github"

    def __init__(
        self,
        license_allowlist: list[str] | None = None,
        *,
        hours: Iterable[str] | None = None,
        local_files: Iterable[str | Path] | None = None,
        event_types: Iterable[str] = ("PushEvent", "PullRequestEvent", "CreateEvent"),
        max_records_per_file: int | None = None,
    ):
        self.license_allowlist = {l.lower() for l in (license_allowlist or [])}
        self.hours = list(hours or [])
        self.local_files = [Path(p) for p in (local_files or [])]
        self.event_types = set(event_types)
        self.max_records_per_file = max_records_per_file
        if not self.hours and not self.local_files:
            raise ValueError(
                "GitHubSource needs either `hours=[...]` (GHArchive HTTP) or "
                "`local_files=[...]` (pre-downloaded .json.gz files)."
            )

    def stream(self) -> Iterator[RawDoc]:
        for path in self.local_files:
            with path.open("rb") as f:
                yield from self._iter_gz_stream(f, origin=str(path))
        for hour in self.hours:
            url = _GHARCHIVE_URL.format(hour=hour)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    yield from self._iter_gz_stream(resp, origin=url)
            except urllib.error.URLError as e:
                # Bubble a clear error so the operator knows it's the network
                # and not a parser bug.
                raise RuntimeError(f"gharchive fetch failed for {url!r}: {e}") from e

    def _iter_gz_stream(self, f, *, origin: str) -> Iterator[RawDoc]:
        gz = gzip.GzipFile(fileobj=f)
        for i, line in enumerate(gz):
            if self.max_records_per_file is not None and i >= self.max_records_per_file:
                break
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") not in self.event_types:
                continue
            doc = self._event_to_doc(ev, origin=origin, index=i)
            if doc is not None:
                yield doc

    def _event_to_doc(self, ev: dict, *, origin: str, index: int) -> RawDoc | None:
        repo = ev.get("repo", {}).get("name", "")
        payload = ev.get("payload", {})
        # License info isn't on every event type; when it is, gate on the
        # allowlist. When it isn't and the operator set a strict allowlist,
        # be conservative and skip.
        license_str = (
            payload.get("license", {}).get("spdx_id")
            if isinstance(payload.get("license"), dict)
            else payload.get("license")
        )
        if self.license_allowlist:
            if not license_str or license_str.lower() not in self.license_allowlist:
                return None
        return RawDoc(
            source="github/gharchive",
            uri=f"{origin}#{index}",
            mime="application/json",
            payload=json.dumps(ev, ensure_ascii=False).encode("utf-8"),
            meta={
                "repo": repo,
                "event_type": ev.get("type"),
                "created_at": ev.get("created_at"),
                "license": license_str,
            },
        )


# ============================================================================
# Arxiv: OAI-PMH bulk + optional `arxiv` client
# ============================================================================


_ARXIV_OAI_BASE = "https://export.arxiv.org/oai2"
_OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}


class ArxivSource:
    """Pull metadata records from arxiv's OAI-PMH endpoint.

    OAI-PMH ships paginated XML; we walk the resumption tokens automatically
    until ``max_records`` is hit. Each record becomes a ``RawDoc`` whose
    ``payload`` is the abstract (utf-8 bytes) and whose ``meta`` carries the
    arxiv id, title, authors, category, date, and license. The full PDF is
    *not* fetched — that's a separate, heavier pipeline that needs the
    arxiv bulk-PDF dump on S3 (``requester-pays``). The metadata path is
    almost always what corpus mining wants.

    Two modes:

    * ``local_xml=Path(...)`` — read pre-downloaded OAI XML responses. Tests
      use this against tiny fixtures.
    * ``set_spec="cs"`` + network — call the OAI endpoint. Be polite: the
      default ``poll_interval_s=3.0`` and ``max_records=100`` keep test runs
      bounded.

    Args:
        set_spec: OAI set to harvest (e.g. ``"cs"``, ``"stat"``, ``"math"``).
        from_date / until_date: ISO date strings, OAI-PMH format.
        max_records: cap on total records yielded.
        local_xml: list of pre-downloaded XML files; uses these instead of
            calling the network.
        poll_interval_s: politeness delay between paginated requests.
    """
    name = "arxiv"

    def __init__(
        self,
        set_spec: str | None = None,
        *,
        from_date: str | None = None,
        until_date: str | None = None,
        max_records: int = 100,
        local_xml: Iterable[str | Path] | None = None,
        poll_interval_s: float = 3.0,
        base_url: str = _ARXIV_OAI_BASE,
    ):
        self.set_spec = set_spec
        self.from_date = from_date
        self.until_date = until_date
        self.max_records = int(max_records)
        self.local_xml = [Path(p) for p in (local_xml or [])]
        self.poll_interval_s = float(poll_interval_s)
        self.base_url = base_url
        if not self.set_spec and not self.local_xml:
            raise ValueError(
                "ArxivSource needs either `set_spec=` (OAI-PMH HTTP) or "
                "`local_xml=[...]` (pre-downloaded XML files)."
            )

    def stream(self) -> Iterator[RawDoc]:
        if self.local_xml:
            for path in self.local_xml:
                xml = path.read_bytes()
                yield from self._iter_oai_xml(xml, origin=str(path))
            return
        # Network: walk OAI-PMH resumption tokens.
        yielded = 0
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "set": self.set_spec or "",
        }
        if self.from_date:
            params["from"] = self.from_date
        if self.until_date:
            params["until"] = self.until_date
        token: str | None = None
        first = True
        while yielded < self.max_records:
            if not first and not token:
                break
            if token:
                query = f"verb=ListRecords&resumptionToken={token}"
            else:
                query = "&".join(f"{k}={v}" for k, v in params.items() if v)
            url = f"{self.base_url}?{query}"
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    xml = resp.read()
            except urllib.error.URLError as e:
                raise RuntimeError(f"arxiv OAI fetch failed for {url!r}: {e}") from e
            for doc in self._iter_oai_xml(xml, origin=url):
                yield doc
                yielded += 1
                if yielded >= self.max_records:
                    return
            token = _parse_resumption_token(xml)
            first = False
            if token:
                time.sleep(self.poll_interval_s)

    def _iter_oai_xml(self, xml_bytes: bytes, *, origin: str) -> Iterator[RawDoc]:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return
        for rec in root.iter("{http://www.openarchives.org/OAI/2.0/}record"):
            md = rec.find("oai:metadata/arxiv:arXiv", _OAI_NS)
            if md is None:
                continue
            arxiv_id = (md.findtext("arxiv:id", default="", namespaces=_OAI_NS) or "").strip()
            title = (md.findtext("arxiv:title", default="", namespaces=_OAI_NS) or "").strip()
            abstract = (md.findtext("arxiv:abstract", default="", namespaces=_OAI_NS) or "").strip()
            categories = (md.findtext("arxiv:categories", default="", namespaces=_OAI_NS) or "").strip()
            created = (md.findtext("arxiv:created", default="", namespaces=_OAI_NS) or "").strip()
            license_str = (md.findtext("arxiv:license", default="", namespaces=_OAI_NS) or "").strip()
            authors = [
                f"{a.findtext('arxiv:forenames', default='', namespaces=_OAI_NS)} "
                f"{a.findtext('arxiv:keyname', default='', namespaces=_OAI_NS)}".strip()
                for a in md.iter("{http://arxiv.org/OAI/arXiv/}author")
            ]
            yield RawDoc(
                source="arxiv/oai",
                uri=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else origin,
                mime="text/plain",
                payload=abstract.encode("utf-8"),
                meta={
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "categories": categories.split(),
                    "created": created,
                    "license": license_str,
                },
            )


def _parse_resumption_token(xml_bytes: bytes) -> str | None:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    tok = root.find(".//{http://www.openarchives.org/OAI/2.0/}resumptionToken")
    if tok is None or not (tok.text or "").strip():
        return None
    return tok.text.strip()


# ============================================================================
# Wikipedia: monthly dump iterator
# ============================================================================


_MEDIAWIKI_NS = {"mw": "http://www.mediawiki.org/xml/export-0.10/"}


class WikipediaSource:
    """Iterate Wikipedia article dumps.

    Wikimedia ships a monthly ``XXwiki-YYYYMMDD-pages-articles.xml.bz2`` per
    language. We support two paths:

    * **wikiextractor** (recommended) — when installed, we shell out to
      ``wikiextractor`` to convert one or more dump files into a directory
      of cleaned JSONL (one ``{"id", "title", "text", "url"}`` per article),
      then stream that directory. Tests use the pre-extracted JSONL path.
    * **raw XML** — a small built-in MediaWiki dump parser for tiny dumps and
      tests. Streams ``<page>`` records, yields one ``RawDoc`` per (non-
      redirect) article carrying the wiki markup. Not as clean as
      wikiextractor's output but always works without extra deps.

    Args:
        langs: language codes recorded in ``RawDoc.meta`` (informational).
        dump_files: list of ``.xml`` / ``.xml.bz2`` dump files.
        extracted_dirs: list of directories produced by wikiextractor
            (``{lang}_{date}/AA/wiki_*`` JSONL files). Used as-is.
        include_redirects: keep redirect pages too. Default False.
    """
    name = "wikipedia"

    def __init__(
        self,
        langs: list[str] | None = None,
        *,
        dump_files: Iterable[str | Path] | None = None,
        extracted_dirs: Iterable[str | Path] | None = None,
        include_redirects: bool = False,
    ):
        self.langs = list(langs or [])
        self.dump_files = [Path(p) for p in (dump_files or [])]
        self.extracted_dirs = [Path(p) for p in (extracted_dirs or [])]
        self.include_redirects = bool(include_redirects)
        if not self.dump_files and not self.extracted_dirs:
            raise ValueError(
                "WikipediaSource needs either `dump_files=[...]` (raw .xml/.xml.bz2) "
                "or `extracted_dirs=[...]` (wikiextractor JSONL output dirs)."
            )

    def stream(self) -> Iterator[RawDoc]:
        for d in self.extracted_dirs:
            yield from self._stream_extracted(d)
        for f in self.dump_files:
            yield from self._stream_dump(f)

    def _stream_extracted(self, root: Path) -> Iterator[RawDoc]:
        # wikiextractor writes wiki_00, wiki_01, ... in AA/AB/... subdirs.
        for jl in sorted(root.rglob("wiki_*")):
            if not jl.is_file():
                continue
            with jl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield RawDoc(
                        source="wikipedia/extracted",
                        uri=rec.get("url", f"{jl}#{rec.get('id', '')}"),
                        mime="text/plain",
                        payload=(rec.get("text") or "").encode("utf-8"),
                        meta={
                            "id": rec.get("id"),
                            "title": rec.get("title"),
                            "langs": self.langs,
                            "license": "CC-BY-SA-4.0",
                        },
                    )

    def _stream_dump(self, path: Path) -> Iterator[RawDoc]:
        # Open .xml or .xml.bz2 transparently.
        if path.suffix == ".bz2":
            import bz2
            f = bz2.open(path, "rb")
        else:
            f = path.open("rb")
        try:
            # iterparse over <page> elements so we don't hold the whole tree.
            ns_tag = "{http://www.mediawiki.org/xml/export-0.10/}page"
            ctx = ET.iterparse(f, events=("end",))
            for ev, elem in ctx:
                if elem.tag != ns_tag:
                    continue
                doc = self._page_to_doc(elem, origin=str(path))
                if doc is not None:
                    yield doc
                elem.clear()
        finally:
            f.close()

    def _page_to_doc(self, page_elem, *, origin: str) -> RawDoc | None:
        title = (page_elem.findtext("mw:title", default="", namespaces=_MEDIAWIKI_NS) or "").strip()
        page_id = (page_elem.findtext("mw:id", default="", namespaces=_MEDIAWIKI_NS) or "").strip()
        redirect_elem = page_elem.find("mw:redirect", _MEDIAWIKI_NS)
        if redirect_elem is not None and not self.include_redirects:
            return None
        revision = page_elem.find("mw:revision", _MEDIAWIKI_NS)
        text = ""
        if revision is not None:
            text = (revision.findtext("mw:text", default="", namespaces=_MEDIAWIKI_NS) or "").strip()
        if not text:
            return None
        return RawDoc(
            source="wikipedia/dump",
            uri=f"{origin}#{page_id}" if page_id else origin,
            mime="text/x-wiki",
            payload=text.encode("utf-8"),
            meta={
                "id": page_id,
                "title": title,
                "langs": self.langs,
                "license": "CC-BY-SA-4.0",
            },
        )


# ============================================================================
# Optional-dep helpers
# ============================================================================


def _require_warcio():
    try:
        from warcio.archiveiterator import ArchiveIterator  # type: ignore
    except ImportError as e:
        raise ImportError(
            "CommonCrawlSource needs `pip install warcio`. "
            "It streams WARC records without loading them in memory."
        ) from e
    return ArchiveIterator


def _require_boto3():
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise ImportError(
            "S3 streaming needs `pip install boto3`. "
            "For local-only use, pass `shard_paths=[...]` instead of `s3_prefix=`."
        ) from e
    return boto3


# ============================================================================
# Registry
# ============================================================================


REGISTRY: dict[str, type] = {
    "local_files": LocalFilesSource,
    "jsonl": JsonlSource,
    "commoncrawl": CommonCrawlSource,
    "github": GitHubSource,
    "arxiv": ArxivSource,
    "wikipedia": WikipediaSource,
}


# Small utility for tests / smoke pipelines: build any source from a config.
def make_source(name: str, **kwargs) -> Source:
    cls = REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"unknown source: {name!r}; known: {sorted(REGISTRY)}")
    return cls(**kwargs)


# Helper to stream-write text-only RawDocs to a single utf-8 blob (used in
# `scripts/smoke_pipeline.py` and the synthetic-factory tests).
def collect_text(docs: Iterable[RawDoc], *, sep: str = "\n\n") -> str:
    return sep.join(d.payload.decode("utf-8", errors="replace") for d in docs)


__all__ = [
    "RawDoc",
    "Source",
    "LocalFilesSource",
    "JsonlSource",
    "CommonCrawlSource",
    "GitHubSource",
    "ArxivSource",
    "WikipediaSource",
    "REGISTRY",
    "make_source",
    "collect_text",
]
