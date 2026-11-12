"""CPU-only, offline midgpt adapter for the bounded search experiment.

This is an experimental adapter, not an API/agent integration. Checkpoints use
train.py's cfg/model layout; no pickle fallback, remote loaders, or training.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Generation:
    """One bounded adapter call, including any observable partial output."""
    text: str
    token_ids: tuple[int, ...] = ()
    prompt_tokens: int = 0
    forward_input_tokens: int = 0
    error: str | None = None


class Tokenizer(Protocol):
    n_vocab: int
    eos_token: int | None

    def encode_ordinary(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class ByteTokenizer:
    """UTF-8 byte IDs for random tiny models, never a GPT-2 substitute."""
    n_vocab = 256
    eos_token = None

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GPT2Tokenizer:
    """Reconstruct tiktoken's GPT-2 encoding from a LOCAL vocab.json.

    GPT-2 merge priority equals vocabulary ID, so a separate merges.txt is not
    needed. The fixed GPT-2 regex/byte alphabet must match the checkpoint.
    """
    n_vocab = 50257
    eos_token = 50256

    def __init__(self, path: Path) -> None:
        import tiktoken

        path = Path(path)
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("GPT-2 vocab must be a local file <= 16 MiB")
        raw = path.read_bytes()
        vocab = json.loads(raw)
        if not isinstance(vocab, dict) or vocab.pop("<|endoftext|>", None) != self.eos_token:
            raise ValueError("expected a GPT-2 vocab.json with endoftext ID 50256")
        if (len(vocab) != 50256 or any(type(v) is not int for v in vocab.values())
                or set(vocab.values()) != set(range(50256))):
            raise ValueError("expected contiguous GPT-2 vocabulary IDs 0..50255")
        visible = [b for b in range(256) if chr(b).isprintable() and chr(b) != " "]
        byte_map = {chr(b): b for b in visible}
        hidden = [b for b in range(256) if b not in visible]
        byte_map.update({chr(256 + i): b for i, b in enumerate(hidden)})
        ranks = {bytes(byte_map[c] for c in token): rank for token, rank in vocab.items()}
        if any(bytes([b]) not in ranks for b in range(256)):
            raise ValueError("GPT-2 vocabulary must contain every byte")
        self.encoding = tiktoken.Encoding(
            name="local-gpt2",
            pat_str=r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            mergeable_ranks=ranks,
            special_tokens={"<|endoftext|>": self.eos_token},
        )
        self.provenance = {"kind": "gpt2", "vocab_sha256": hashlib.sha256(raw).hexdigest(),
                           "tiktoken_version": tiktoken.__version__}

    def encode_ordinary(self, text: str) -> list[int]:
        return self.encoding.encode_ordinary(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)


class LocalModelAdapter:
    """Generate one token per GPT.generate call to retain partial error costs.

    Initial prompts must fit the context; subsequent steps use GPT's existing
    sliding window. There is no KV cache or wall-clock preemption. CPU only.
    """

    def __init__(self, model: object, tokenizer: Tokenizer, source: dict,
                 temperature: float = 0.0) -> None:
        import torch

        if not math.isfinite(temperature) or not 0 <= temperature <= 10:
            raise ValueError("temperature must be finite and in [0, 10]")
        if model.cfg.vocab_size < tokenizer.n_vocab:
            raise ValueError("model vocabulary is smaller than tokenizer vocabulary")
        self.model = model.to("cpu").eval()
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.provenance = {
            **source, "adapter_version": "local-cpu-gpt-v1",
            "adapter_sha256": _sha_file(Path(__file__)),
            "model_source_sha256": _sha_file(Path(__file__).with_name("model.py")),
            "torch_version": str(torch.__version__), "device": "cpu", "dtype": "float32",
            "threads": torch.get_num_threads(), "config": asdict(model.cfg),
            "temperature": temperature, "top_k": None, "top_p": None,
            "termination": "first closing brace or EOS, otherwise token limit",
            "tokenizer": getattr(tokenizer, "provenance", {"kind": "byte-v1"}),
        }

    @classmethod
    def tiny(cls, seed: int = 0, temperature: float = 0.0) -> LocalModelAdapter:
        """Seeded random 1-layer GPT; exercises inference, not task competence."""
        import torch
        from model import GPT, GPTConfig

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = GPT(GPTConfig(vocab_size=256, block_size=256, n_layer=1,
                                  n_head=2, d_model=32, d_ffn=64))
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().contiguous().numpy().tobytes())
        return cls(model, ByteTokenizer(), {
            "kind": "tiny-random-smoke", "initialization_seed": seed,
            "weights_sha256": digest.hexdigest(), "trained": False,
        }, temperature)

    @classmethod
    def from_checkpoint(cls, path: Path, vocab_path: Path | None = None,
                        temperature: float = 0.0) -> LocalModelAdapter:
        """Load train.py's cfg/model payload with restricted torch deserialization."""
        import torch
        from model import GPT, GPTConfig

        path = Path(path)
        if not path.is_file():
            raise ValueError("checkpoint must be a local file")
        digest = _sha_file(path)
        # Never fall back to weights_only=False for older/custom pickle payloads.
        saved = torch.load(path, map_location="cpu", weights_only=True)
        cfg = saved["cfg"]
        name = cfg["tokenizer"]
        if name == "byte-v1":
            tokenizer = ByteTokenizer()
        elif name == "gpt2" and vocab_path is not None:
            tokenizer = GPT2Tokenizer(vocab_path)
        else:
            raise ValueError("supported tokenizers: byte-v1, or gpt2 with --gpt2-vocab LOCAL_FILE")
        model = GPT(GPTConfig(**cfg["model"]))
        state = {key.removeprefix("_orig_mod."): value for key, value in saved["model"].items()}
        model.load_state_dict(state, strict=True)
        return cls(model, tokenizer, {"kind": "local-checkpoint", "checkpoint_sha256": digest}, temperature)

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> Generation:
        """Return exact emitted IDs and attempted forward-input work, even on error."""
        import torch

        ids: list[int] = []
        text = ""
        prompt_tokens = 0
        forward_tokens = 0
        error = None
        try:
            if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
                raise ValueError("max_new_tokens must be in [1, 512]")
            if not isinstance(prompt, str) or not 1 <= len(prompt) <= 16384:
                raise ValueError("prompt must contain 1..16384 characters")
            encoded = self.tokenizer.encode_ordinary(prompt)
            prompt_tokens = len(encoded)
            if not 1 <= prompt_tokens <= min(2048, self.model.cfg.block_size):
                raise ValueError("prompt exceeds model context or 2048-token prompt cap")
            context = torch.tensor([encoded], dtype=torch.long)
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.manual_seed(seed)
                for _ in range(max_new_tokens):
                    forward_tokens += min(context.size(1), self.model.cfg.block_size)
                    context = self.model.generate(context, 1, temperature=self.temperature)
                    token = int(context[0, -1].item())
                    ids.append(token)
                    if token == self.tokenizer.eos_token:
                        break
                    text = self.tokenizer.decode(ids)
                    if text.rstrip().endswith("}"):
                        break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:1000]
        return Generation(text, tuple(ids), prompt_tokens, forward_tokens, error)
