"""Source connectors. Each yields raw bytes or document records."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol


@dataclass
class RawDoc:
    source: str
    uri: str
    mime: str
    payload: bytes
    meta: dict


class Source(Protocol):
    name: str
    def stream(self) -> Iterator[RawDoc]: ...


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

    Each JSON record must contain `text_key` (default 'text').
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


class CommonCrawlSource:
    """Iterate WARC files from a CC snapshot. Real impl: warcio + S3 streaming."""
    name = "commoncrawl"
    def __init__(self, snapshot: str, shard_glob: str):
        self.snapshot = snapshot
        self.shard_glob = shard_glob
    def stream(self) -> Iterator[RawDoc]:
        raise NotImplementedError("wire to warcio.ArchiveIterator over S3")


class GitHubSource:
    """Permissively-licensed code from GHArchive + repo mirror."""
    name = "github"
    def __init__(self, license_allowlist: list[str]):
        self.license_allowlist = license_allowlist
    def stream(self) -> Iterator[RawDoc]:
        raise NotImplementedError("wire to GHArchive (https://www.gharchive.org/)")


class ArxivSource:
    name = "arxiv"
    def stream(self) -> Iterator[RawDoc]:
        raise NotImplementedError("use the `arxiv` python client or OAI-PMH bulk dump")


class WikipediaSource:
    name = "wikipedia"
    def __init__(self, langs: list[str]):
        self.langs = langs
    def stream(self) -> Iterator[RawDoc]:
        raise NotImplementedError("use wikiextractor over the monthly dumps")


REGISTRY: dict[str, type] = {
    "local_files": LocalFilesSource,
    "jsonl": JsonlSource,
    "commoncrawl": CommonCrawlSource,
    "github": GitHubSource,
    "arxiv": ArxivSource,
    "wikipedia": WikipediaSource,
}
