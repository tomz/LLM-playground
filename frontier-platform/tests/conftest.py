import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# Evict stdlib `platform` so our local package wins (pytest imports stdlib
# platform during startup before conftest runs).
for _m in [k for k in list(sys.modules) if k == "platform" or k.startswith("platform.")]:
    del sys.modules[_m]

import pytest

from platform.data.synthetic import write_corpus
from platform.data.acquire import LocalFilesSource
from platform.data.shard import tokenize_and_shard
from platform.tokenizer.bytes import BytesTokenizer


@pytest.fixture
def tmp_corpus_dir(tmp_path):
    return write_corpus(tmp_path / "corpus", n_files=20, words_per_file=200, seed=0)


@pytest.fixture
def tiny_tokenizer():
    return BytesTokenizer()


@pytest.fixture
def tiny_shards(tmp_path, tmp_corpus_dir, tiny_tokenizer):
    src = LocalFilesSource(tmp_corpus_dir)
    out = tmp_path / "shards"
    uris = tokenize_and_shard(
        src.stream(), tiny_tokenizer, out, domain="synth", shard_tokens=4096
    )
    return out, uris
