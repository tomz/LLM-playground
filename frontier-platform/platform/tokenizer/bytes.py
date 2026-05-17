"""Pure-stdlib byte-level tokenizer used for tests and as a fallback.

Layout: token IDs 0..255 are the raw bytes; 256..511 are reserved specials.
"""
from __future__ import annotations

_SPECIAL_NAMES = [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool_call|>", "<|tool_result|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
]


class BytesTokenizer:
    def __init__(self) -> None:
        self._specials = {name: 256 + i for i, name in enumerate(_SPECIAL_NAMES)}

    @property
    def vocab_size(self) -> int:
        return 512

    @property
    def bos_id(self) -> int:
        return self._specials["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self._specials["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self._specials["<|pad|>"]

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        bs = bytes(i for i in ids if 0 <= i < 256)
        return bs.decode("utf-8", errors="replace")
