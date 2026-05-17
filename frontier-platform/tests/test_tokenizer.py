import pytest
from platform.tokenizer.bytes import BytesTokenizer


def test_bytes_tokenizer_roundtrip():
    t = BytesTokenizer()
    s = "hello, world! éàî"
    assert t.decode(t.encode(s)) == s


def test_bytes_tokenizer_specials_distinct():
    t = BytesTokenizer()
    assert t.bos_id != t.eos_id != t.pad_id
    assert all(i >= 256 for i in (t.bos_id, t.eos_id, t.pad_id))
    assert t.vocab_size == 512


def test_bpe_train_either_works_or_says_so(tmp_path):
    from platform.tokenizer import bpe
    corpus = tmp_path / "c.txt"
    corpus.write_text("the quick brown fox " * 200)
    try:
        import tokenizers  # noqa
    except ImportError:
        pytest.xfail("tokenizers not installed")
    out = tmp_path / "tok.json"
    cfg = bpe.TokenizerConfig(vocab_size=300, specials=["<|bos|>", "<|eos|>", "<|pad|>"])
    bpe.train(str(corpus), cfg, str(out))
    tok = bpe.Tokenizer(str(out))
    ids = tok.encode("the quick")
    assert isinstance(ids, list) and ids
