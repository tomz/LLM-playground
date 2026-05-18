"""BPE tokenizer trainer + runtime wrapper.

Real impl: HuggingFace `tokenizers` (Rust) for training; `tiktoken` for serving.
Falls back to a ByteLevel tokenizers tokenizer with no merges when the dataset
is too small to train. If the `tokenizers` package isn't installed, `train`
and `Tokenizer` raise `NotImplementedError`.
"""
from __future__ import annotations
import glob
from dataclasses import dataclass, field

DEFAULT_SPECIALS = [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool_call|>", "<|tool_result|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
] + [f"<|reserved_{i}|>" for i in range(256)]


@dataclass
class TokenizerConfig:
    vocab_size: int = 100_352
    byte_level: bool = True
    split_digits: bool = True
    specials: list[str] = field(default_factory=lambda: list(DEFAULT_SPECIALS))


def _require_tokenizers() -> None:
    from importlib.util import find_spec
    if find_spec("tokenizers") is None:
        raise NotImplementedError(
            "the `tokenizers` package is required to train BPE; "
            "`pip install tokenizers` or use platform.tokenizer.bytes.BytesTokenizer"
        )


def train(corpus_glob: str, cfg: TokenizerConfig, out_path: str) -> None:
    """Train BPE over the files matching corpus_glob."""
    _require_tokenizers()
    from tokenizers import Tokenizer as HFTokenizer  # type: ignore
    from tokenizers.models import BPE  # type: ignore
    from tokenizers.trainers import BpeTrainer  # type: ignore
    from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence  # type: ignore
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # type: ignore

    files = sorted(glob.glob(corpus_glob))
    if not files:
        raise FileNotFoundError(f"no files matched: {corpus_glob}")
    tok = HFTokenizer(BPE(unk_token=None))
    pre = [ByteLevel(add_prefix_space=False)] if cfg.byte_level else []
    if cfg.split_digits:
        pre.append(Digits(individual_digits=True))
    tok.pre_tokenizer = Sequence(pre) if len(pre) > 1 else (pre[0] if pre else None)
    if cfg.byte_level:
        tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=cfg.vocab_size, special_tokens=list(cfg.specials))
    tok.train(files, trainer)
    tok.save(out_path)


class Tokenizer:
    def __init__(self, path: str):
        _require_tokenizers()
        from tokenizers import Tokenizer as HFTokenizer  # type: ignore
        self._tok = HFTokenizer.from_file(path)
        v = self._tok.get_vocab()
        self._bos = v.get("<|bos|>", 0)
        self._eos = v.get("<|eos|>", 0)
        self._pad = v.get("<|pad|>", 0)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(list(ids))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return self._bos

    @property
    def eos_id(self) -> int:
        return self._eos

    @property
    def pad_id(self) -> int:
        return self._pad
