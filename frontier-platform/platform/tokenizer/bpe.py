"""BPE tokenizer trainer + runtime wrapper.

Real impl: HuggingFace `tokenizers` (Rust) for training; `tiktoken` for serving.
"""
from __future__ import annotations
from dataclasses import dataclass, field

DEFAULT_SPECIALS = [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool_call|>", "<|tool_result|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
] + [f"<|reserved_{i}|>" for i in range(256)]


@dataclass
class TokenizerConfig:
    vocab_size: int = 100_352  # multiple of 128 for tensor cores
    byte_level: bool = True
    split_digits: bool = True
    specials: list[str] = field(default_factory=lambda: list(DEFAULT_SPECIALS))


def train(corpus_glob: str, cfg: TokenizerConfig, out_path: str) -> None:
    """Train BPE over a representative sample. ~12h on 96-core box for 100GB."""
    raise NotImplementedError


class Tokenizer:
    def __init__(self, path: str): ...
    def encode(self, text: str) -> list[int]: raise NotImplementedError
    def decode(self, ids: list[int]) -> str: raise NotImplementedError
    @property
    def vocab_size(self) -> int: raise NotImplementedError
