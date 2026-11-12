"""Shared CPU fixtures; no stdlib import interception is needed."""
from __future__ import annotations

import pytest

from frontier_platform.data.acquire import LocalFilesSource
from frontier_platform.data.shard import tokenize_and_shard
from frontier_platform.data.synthetic import write_corpus
from frontier_platform.tokenizer.bytes import BytesTokenizer


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


@pytest.fixture
def tiny_model_cfg():
    from frontier_platform.model.config import ModelConfig
    return ModelConfig(
        vocab_size=512, n_layer=4, n_head=4, n_kv_head=2,
        d_model=128, d_ffn=384, max_seq_len=128,
        rope_base=10000.0,
    )


@pytest.fixture
def tiny_model(tiny_model_cfg):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    from frontier_platform.model.transformer import Transformer
    return Transformer(tiny_model_cfg)


@pytest.fixture
def gpu_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        major, _ = torch.cuda.get_device_capability(0)
        supported = torch.cuda.get_arch_list()
        if supported and not any(f"sm_{major}" in s for s in supported):
            torch.zeros(1, device="cuda") + 1
    except Exception as e:
        pytest.skip(f"CUDA present but unusable: {e}")
    return torch.device("cuda")
