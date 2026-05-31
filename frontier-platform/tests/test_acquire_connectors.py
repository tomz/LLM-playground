"""Tests for the source connectors that previously raised NotImplementedError.

Every connector ships two paths:

* a **local** path that takes pre-downloaded shards / dumps / extracted JSONL
  and parses them deterministically (this is what CI runs against), and
* a **network** path gated behind explicit constructor args.

These tests only exercise the local path so they're fast, deterministic, and
need no network egress. They do verify the parsers handle the real on-disk
formats produced by the upstream tools (warcio, GHArchive's hourly json.gz,
arxiv's OAI-PMH XML, wikiextractor's JSONL).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from platform.data.acquire import (
    ArxivSource,
    CommonCrawlSource,
    GitHubSource,
    LocalFilesSource,
    REGISTRY,
    WikipediaSource,
    make_source,
)


# ----- registry + factory ---------------------------------------------------


def test_registry_lists_every_connector():
    assert set(REGISTRY) == {
        "local_files", "jsonl", "commoncrawl",
        "github", "arxiv", "wikipedia",
    }


def test_make_source_constructs_known_and_rejects_unknown(tmp_path):
    s = make_source("local_files", root=tmp_path)
    assert isinstance(s, LocalFilesSource)
    with pytest.raises(KeyError):
        make_source("aol_dialup")


# ----- CommonCrawl: argument validation -------------------------------------


def test_commoncrawl_requires_shard_paths_or_s3():
    with pytest.raises(ValueError):
        CommonCrawlSource(snapshot="CC-MAIN-2025-13")


def test_commoncrawl_missing_warcio_raises_clear_importerror(monkeypatch, tmp_path):
    """Force-block the warcio import and confirm we don't fall back to
    NotImplementedError or a generic ModuleNotFoundError."""
    import sys
    # Pretend warcio's submodule can't import.
    monkeypatch.setitem(sys.modules, "warcio", None)
    monkeypatch.setitem(sys.modules, "warcio.archiveiterator", None)

    # Touch a fake file so the constructor passes its validation.
    fake = tmp_path / "fake.warc.gz"
    fake.write_bytes(b"not a real warc")
    src = CommonCrawlSource(snapshot="x", shard_paths=[fake])
    with pytest.raises(ImportError) as e:
        list(src.stream())
    assert "warcio" in str(e.value)


def test_commoncrawl_streams_with_real_warcio(tmp_path):
    """If warcio is installed, hand it a minimal WARC and check we get the doc back."""
    pytest.importorskip("warcio")
    from warcio.warcwriter import WARCWriter
    from warcio.statusandheaders import StatusAndHeaders

    path = tmp_path / "tiny.warc.gz"
    with path.open("wb") as f:
        writer = WARCWriter(f, gzip=True)
        # Build a fake HTTP response record.
        http_headers = StatusAndHeaders(
            "200 OK",
            [("Content-Type", "text/html; charset=utf-8")],
            protocol="HTTP/1.0",
        )
        record = writer.create_warc_record(
            "https://example.com/hello",
            "response",
            payload=__import__("io").BytesIO(b"<html><body>hello world</body></html>"),
            http_headers=http_headers,
        )
        writer.write_record(record)

    src = CommonCrawlSource(snapshot="CC-TEST", shard_paths=[path])
    docs = list(src.stream())
    assert len(docs) == 1
    d = docs[0]
    assert d.source.startswith("commoncrawl")
    assert d.uri == "https://example.com/hello"
    assert d.mime == "text/html"
    assert b"hello world" in d.payload
    assert d.meta["snapshot"] == "CC-TEST"
    assert d.meta["warc_shard"] == str(path)


def test_commoncrawl_skips_oversized(tmp_path):
    pytest.importorskip("warcio")
    from warcio.warcwriter import WARCWriter
    from warcio.statusandheaders import StatusAndHeaders

    path = tmp_path / "big.warc.gz"
    with path.open("wb") as f:
        writer = WARCWriter(f, gzip=True)
        big_payload = b"X" * 200_000
        http_headers = StatusAndHeaders(
            "200 OK", [("Content-Type", "text/html")], protocol="HTTP/1.0",
        )
        writer.write_record(writer.create_warc_record(
            "https://example.com/big", "response",
            payload=__import__("io").BytesIO(big_payload),
            http_headers=http_headers,
        ))

    src = CommonCrawlSource(snapshot="x", shard_paths=[path], max_payload_bytes=10_000)
    assert list(src.stream()) == []


# ----- GitHub / GHArchive ----------------------------------------------------


def _write_gharchive(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wb") as f:
        for ev in events:
            f.write((json.dumps(ev) + "\n").encode("utf-8"))


def test_github_requires_hours_or_local_files():
    with pytest.raises(ValueError):
        GitHubSource(license_allowlist=["MIT"])


def test_github_parses_local_gharchive(tmp_path):
    path = tmp_path / "2025-11-01-12.json.gz"
    _write_gharchive(path, [
        {"type": "PushEvent", "repo": {"name": "foo/bar"},
         "payload": {"license": {"spdx_id": "MIT"}},
         "created_at": "2025-11-01T12:00:00Z"},
        {"type": "WatchEvent", "repo": {"name": "baz/qux"},
         "payload": {}, "created_at": "2025-11-01T12:00:01Z"},   # wrong event type
        {"type": "PullRequestEvent", "repo": {"name": "alice/bob"},
         "payload": {"license": "Apache-2.0"},
         "created_at": "2025-11-01T12:00:02Z"},
        {"type": "PushEvent", "repo": {"name": "naughty/proprietary"},
         "payload": {"license": {"spdx_id": "Proprietary"}}},   # excluded by allowlist
    ])
    src = GitHubSource(license_allowlist=["MIT", "Apache-2.0"], local_files=[path])
    docs = list(src.stream())
    # WatchEvent filtered by event_type, proprietary filtered by license.
    repos = sorted(d.meta["repo"] for d in docs)
    assert repos == ["alice/bob", "foo/bar"]
    for d in docs:
        assert d.source == "github/gharchive"
        assert d.mime == "application/json"
        rehydrated = json.loads(d.payload)
        assert rehydrated["repo"]["name"] == d.meta["repo"]


def test_github_allowlist_empty_is_permissive_when_license_unknown(tmp_path):
    """No allowlist => keep events even when the payload has no license."""
    path = tmp_path / "h.json.gz"
    _write_gharchive(path, [
        {"type": "CreateEvent", "repo": {"name": "x/y"}, "payload": {}},
    ])
    src = GitHubSource(local_files=[path])  # empty allowlist
    docs = list(src.stream())
    assert len(docs) == 1
    assert docs[0].meta["repo"] == "x/y"


def test_github_max_records_per_file(tmp_path):
    path = tmp_path / "h.json.gz"
    _write_gharchive(path, [
        {"type": "PushEvent", "repo": {"name": f"r/{i}"}, "payload": {}}
        for i in range(10)
    ])
    src = GitHubSource(local_files=[path], max_records_per_file=3)
    assert len(list(src.stream())) == 3


# ----- Arxiv -----------------------------------------------------------------


def _arxiv_oai_xml(records: list[dict]) -> bytes:
    body = ""
    for r in records:
        authors_xml = "".join(
            f"<author><keyname>{a['keyname']}</keyname>"
            f"<forenames>{a['forenames']}</forenames></author>"
            for a in r.get("authors", [])
        )
        body += f"""
          <record>
            <header><identifier>oai:arXiv.org:{r['id']}</identifier></header>
            <metadata>
              <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
                <id>{r['id']}</id>
                <title>{r['title']}</title>
                <abstract>{r['abstract']}</abstract>
                <categories>{r.get('categories', 'cs.LG')}</categories>
                <created>{r.get('created', '2025-11-01')}</created>
                <license>{r.get('license', 'cc-by-4.0')}</license>
                {authors_xml}
              </arXiv>
            </metadata>
          </record>"""
    xml = f"""<?xml version="1.0"?>
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
      <ListRecords>
        {body}
      </ListRecords>
    </OAI-PMH>"""
    return xml.encode("utf-8")


def test_arxiv_requires_set_spec_or_local_xml():
    with pytest.raises(ValueError):
        ArxivSource()


def test_arxiv_parses_local_oai_xml(tmp_path):
    xml_path = tmp_path / "oai.xml"
    xml_path.write_bytes(_arxiv_oai_xml([
        {"id": "2501.00001", "title": "On Foo", "abstract": "We study foo.",
         "categories": "cs.LG stat.ML",
         "authors": [{"keyname": "Smith", "forenames": "Alice"},
                     {"keyname": "Doe", "forenames": "Bob"}]},
        {"id": "2501.00002", "title": "On Bar", "abstract": "Bar is great.",
         "license": "arXiv-nonexclusive-1.0"},
    ]))
    src = ArxivSource(local_xml=[xml_path])
    docs = list(src.stream())
    assert len(docs) == 2
    a, b = docs
    assert a.meta["arxiv_id"] == "2501.00001"
    assert a.meta["title"] == "On Foo"
    assert a.payload == b"We study foo."
    assert a.meta["categories"] == ["cs.LG", "stat.ML"]
    assert a.meta["authors"] == ["Alice Smith", "Bob Doe"]
    assert a.meta["license"] == "cc-by-4.0"
    assert a.uri == "https://arxiv.org/abs/2501.00001"
    assert b.meta["license"] == "arXiv-nonexclusive-1.0"


def test_arxiv_handles_malformed_xml_gracefully(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_bytes(b"<not-real-xml")
    src = ArxivSource(local_xml=[bad])
    assert list(src.stream()) == []


# ----- Wikipedia -------------------------------------------------------------


def test_wikipedia_requires_dump_or_extracted():
    with pytest.raises(ValueError):
        WikipediaSource(langs=["en"])


def test_wikipedia_streams_wikiextractor_jsonl(tmp_path):
    root = tmp_path / "enwiki_20251101"
    (root / "AA").mkdir(parents=True)
    (root / "AA" / "wiki_00").write_text("\n".join([
        json.dumps({"id": "1", "title": "Photosynthesis",
                    "text": "Photosynthesis is...", "url": "https://en.wikipedia.org/?curid=1"}),
        json.dumps({"id": "2", "title": "Mitochondria",
                    "text": "The mitochondrion is...", "url": "https://en.wikipedia.org/?curid=2"}),
        "",  # empty line tolerated
    ]) + "\n", encoding="utf-8")
    src = WikipediaSource(langs=["en"], extracted_dirs=[root])
    docs = list(src.stream())
    assert len(docs) == 2
    titles = {d.meta["title"] for d in docs}
    assert titles == {"Photosynthesis", "Mitochondria"}
    for d in docs:
        assert d.source == "wikipedia/extracted"
        assert d.meta["license"] == "CC-BY-SA-4.0"
        assert d.meta["langs"] == ["en"]
        assert d.payload.decode("utf-8")


def test_wikipedia_parses_raw_xml_dump(tmp_path):
    xml = """<?xml version="1.0"?>
    <mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
      <page>
        <title>Apple</title>
        <id>42</id>
        <revision>
          <text>An apple is a sweet, edible fruit.</text>
        </revision>
      </page>
      <page>
        <title>Apple_(disambiguation)</title>
        <id>43</id>
        <redirect title="Apple"/>
        <revision><text>#REDIRECT [[Apple]]</text></revision>
      </page>
      <page>
        <title>Banana</title>
        <id>44</id>
        <revision>
          <text>A banana is an elongated, edible fruit.</text>
        </revision>
      </page>
    </mediawiki>"""
    path = tmp_path / "tiny.xml"
    path.write_text(xml, encoding="utf-8")
    src = WikipediaSource(langs=["en"], dump_files=[path])
    docs = list(src.stream())
    titles = sorted(d.meta["title"] for d in docs)
    # Redirect skipped by default.
    assert titles == ["Apple", "Banana"]
    for d in docs:
        assert d.source == "wikipedia/dump"
        assert d.mime == "text/x-wiki"
        assert d.payload.decode("utf-8")


def test_wikipedia_include_redirects_keeps_them(tmp_path):
    xml = """<?xml version="1.0"?>
    <mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
      <page>
        <title>Foo</title><id>1</id>
        <redirect title="Bar"/>
        <revision><text>#REDIRECT [[Bar]]</text></revision>
      </page>
    </mediawiki>"""
    path = tmp_path / "r.xml"
    path.write_text(xml, encoding="utf-8")
    src = WikipediaSource(langs=["en"], dump_files=[path], include_redirects=True)
    assert len(list(src.stream())) == 1
