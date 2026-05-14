"""Source connectors. Each yields raw bytes or document records."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass
class RawDoc:
    source: str        # e.g. 'commoncrawl/CC-MAIN-2024-30'
    uri: str           # original URL or path
    mime: str          # 'text/html', 'application/pdf', 'text/x-python', ...
    payload: bytes
    meta: dict


class Source(Protocol):
    name: str
    def stream(self) -> Iterator[RawDoc]: ...


class CommonCrawlSource:
    """Iterate WARC files from a CC snapshot. Real impl: warcio + S3 streaming."""
    name = "commoncrawl"
    def __init__(self, snapshot: str, shard_glob: str): ...
    def stream(self) -> Iterator[RawDoc]:
        raise NotImplementedError("wire to warcio.ArchiveIterator over S3")


class GitHubSource:
    """Permissively-licensed code from GHArchive + repo mirror."""
    name = "github"
    def __init__(self, license_allowlist: list[str]): ...
    def stream(self) -> Iterator[RawDoc]:
        raise NotImplementedError


class ArxivSource:
    name = "arxiv"
    def stream(self) -> Iterator[RawDoc]:
        raise NotImplementedError


class WikipediaSource:
    name = "wikipedia"
    def __init__(self, langs: list[str]): ...
    def stream(self) -> Iterator[RawDoc]:
        raise NotImplementedError


REGISTRY: dict[str, type] = {
    "commoncrawl": CommonCrawlSource,
    "github": GitHubSource,
    "arxiv": ArxivSource,
    "wikipedia": WikipediaSource,
}
