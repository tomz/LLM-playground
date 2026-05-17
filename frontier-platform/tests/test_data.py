import json
import numpy as np


from platform.data.acquire import LocalFilesSource, JsonlSource
from platform.data.extract import _html_to_text
from platform.data.filter import (
    detect_language, gopher_rules, quality_classifier, pipeline,
)
from platform.data.dedup import MinHashDeduper, stream_exact_dedup
from platform.data.decontaminate import Decontaminator
from platform.data.shard import tokenize_and_shard
from platform.data.mix import DomainSpec, MixtureSampler
from platform.data.loader import StreamingLoader



def test_local_files_source_yields_docs(tmp_corpus_dir):
    docs = list(LocalFilesSource(tmp_corpus_dir).stream())
    assert len(docs) == 20
    assert all(d.mime == "text/plain" for d in docs)
    assert all(d.payload for d in docs)


def test_jsonl_source(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"text": "hello"}\n{"text": "world", "src": "a"}\n')
    docs = list(JsonlSource(p).stream())
    assert [d.payload for d in docs] == [b"hello", b"world"]
    assert docs[1].meta == {"src": "a"}


def test_extract_html_strips_tags():
    b = b"<html><body><script>x=1</script><p>Hello <b>world</b>!</p></body></html>"
    out = _html_to_text(b)
    assert "Hello" in out and "world" in out
    assert "<" not in out and "script" not in out


def test_filter_pipeline_keeps_english_quality_text():
    text = ("The quick brown fox jumps over the lazy dog. " * 30) + \
           "This is a sample of reasonably written English prose with punctuation, stopwords, and structure. " * 5
    lang, conf = detect_language(text)
    assert lang == "en" and conf >= 0.65
    assert gopher_rules(text).keep
    assert quality_classifier(text) > 0.0
    v = pipeline(text, {"en"})
    assert v.keep, v.reason


def test_gopher_rules_rejects_short_docs():
    assert not gopher_rules("hello world").keep


def test_minhash_dedup_drops_near_duplicates():
    d = MinHashDeduper(num_perm=64, threshold=0.5, ngram=3)
    a = "the quick brown fox jumps over the lazy dog one two three"
    b = "the quick brown fox jumps over the lazy dog one two four"  # near-dup
    c = "completely unrelated text about transformers and attention layers"
    assert d.add("a", a) is True
    assert d.add("b", b) is False
    assert d.add("c", c) is True


def test_exact_dedup_already_works():
    out = list(stream_exact_dedup([("1", "hi"), ("2", "HI"), ("3", "bye")]))
    assert [x[0] for x in out] == ["1", "3"]


def test_decontaminator_flags_overlap(tmp_path):
    eval_file = tmp_path / "eval.txt"
    eval_file.write_text("the quick brown fox jumps over the lazy dog and then runs back home today " * 3)
    deco = Decontaminator([str(eval_file)], n=5)
    assert deco.is_contaminated("the quick brown fox jumps over the lazy dog and then runs back home today")
    assert not deco.is_contaminated("totally different sentence with unique words like xylophone refrigerator zebra")
    r = deco.report()
    assert str(eval_file) in r
    assert r[str(eval_file)]["flagged_train_docs"] >= 1


def test_shard_roundtrip(tmp_path, tiny_tokenizer):
    docs = ["hello world", "foo bar baz", "alpha beta"]
    uris = tokenize_and_shard(iter(docs), tiny_tokenizer, tmp_path, domain="d", shard_tokens=10_000)
    assert len(uris) == 1
    arr = np.fromfile(uris[0], dtype=np.uint32)
    # Each doc starts with BOS.
    assert (arr == tiny_tokenizer.bos_id).sum() == 3
    # Verify idx file.
    idx = json.loads(open(uris[0].replace(".bin", ".idx")).read())
    assert idx["schema_version"] == 1
    assert idx["tokens"] == arr.size
    assert len(idx["doc_offsets"]) == 3


def test_shard_rollover(tmp_path, tiny_tokenizer):
    docs = ["x" * 100 for _ in range(20)]
    uris = tokenize_and_shard(iter(docs), tiny_tokenizer, tmp_path, domain="d", shard_tokens=200)
    assert len(uris) > 1


def test_mixture_sampler_respects_weights(tiny_shards):
    out, _uris = tiny_shards
    # Create two "domains" pointing at the same shards but with different weights.
    specs = [
        DomainSpec("a", str(out / "synth" / "*.bin"), weight=3.0),
        DomainSpec("b", str(out / "synth" / "*.bin"), weight=1.0, epochs_cap=1e9),
    ]
    # Bump cap so we don't exhaust.
    specs[0].epochs_cap = 1e9
    samp = MixtureSampler(specs, global_seed=42)
    # Track which domain index is picked by re-deriving (peek at internal weights).
    counts = {0: 0, 1: 0}
    for step in range(1000):
        rng = samp._rng(0, step)
        idx = int(rng.choice(2, p=samp.weights))
        counts[idx] += 1
    # Expect ~75/25 within wide tolerance.
    ratio = counts[0] / (counts[0] + counts[1])
    assert 0.65 < ratio < 0.85


def test_mixture_sampler_returns_existing_shard(tiny_shards):
    out, _ = tiny_shards
    specs = [DomainSpec("a", str(out / "synth" / "*.bin"), weight=1.0, epochs_cap=1e9)]
    samp = MixtureSampler(specs, global_seed=1)
    for step in range(5):
        p = samp.next_shard(rank=0, step=step)
        assert p.endswith(".bin")


def test_streaming_loader_state_resume(tiny_shards):
    out, _ = tiny_shards
    specs = [DomainSpec("a", str(out / "synth" / "*.bin"), weight=1.0, epochs_cap=1e9)]

    def mk():
        return StreamingLoader(
            MixtureSampler(specs, global_seed=7),
            seq_len=16, micro_batch=2, rank=0, world_size=1, seed=7,
        )

    L = mk()
    it = iter(L)
    for _ in range(3):
        next(it)
    sd = L.state_dict()
    a_x, a_y = next(it)

    L2 = mk()
    L2.load_state_dict(sd)
    b_x, b_y = next(iter(L2))
    assert (a_x == b_x).all() and (a_y == b_y).all()
