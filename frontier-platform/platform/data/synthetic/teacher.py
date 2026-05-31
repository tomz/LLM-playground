"""Teacher endpoints for synthetic generation.

A :class:`Teacher` is anything that maps ``(prompt, **kwargs) -> str``. In
production this is a model endpoint (vLLM, TorchEngine, or an external API);
for tests we ship deterministic teachers that don't need GPUs.

The :class:`EngineTeacher` adapter lets you plug any
:class:`platform.serving.engine.Engine` into the factory by encoding prompts
with a tokenizer, draining the async ``generate`` stream, and decoding the
generated ids back to text. It's the production path.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Protocol


class Teacher(Protocol):
    """Protocol every teacher implements. ``name`` is recorded in lineage."""

    name: str

    def generate(self, prompt: str, **kwargs) -> str: ...


@dataclass
class EchoTeacher:
    """Deterministic teacher that returns the prompt verbatim with a tag.

    Useful for testing the factory's plumbing: the rejection sampler, deduper,
    and decontaminator can all be exercised without an actual model.
    """

    name: str = "echo"
    suffix: str = ""

    def generate(self, prompt: str, **kwargs) -> str:
        return prompt if not self.suffix else f"{prompt}{self.suffix}"


@dataclass
class TemplateTeacher:
    """Deterministic teacher that fills a response template.

    The template can reference ``{prompt}`` and any kwargs passed to
    :meth:`generate`. Lets policy tests assert on exact response strings.
    """

    template: str
    name: str = "template"

    def generate(self, prompt: str, **kwargs) -> str:
        return self.template.format(prompt=prompt, **kwargs)


@dataclass
class CallableTeacher:
    """Wrap an arbitrary callable as a teacher.

    Lets you adapt e.g. ``lambda p: openai_client.responses.create(...)`` without
    writing a class. ``name`` should be set so lineage records identify the
    source.
    """

    fn: Callable[[str], str]
    name: str = "callable"

    def generate(self, prompt: str, **kwargs) -> str:
        return str(self.fn(prompt))


@dataclass
class EngineTeacher:
    """Use a :class:`platform.serving.engine.Engine` as the teacher.

    Encodes the prompt with ``tokenizer``, drains the async generate stream
    until ``done``, and decodes the generated ids back to text. The Engine can
    be the in-process TorchEngine or (when wired) the vLLM backend; the factory
    doesn't care.

    The optional ``request_kwargs`` overrides default ``GenRequest`` fields
    (e.g. ``max_new_tokens``, ``temperature``, ``stop``).
    """

    engine: object       # platform.serving.engine.Engine; typed loosely to keep this file import-light
    tokenizer: object    # has .encode(str) and .decode(list[int])
    name: str = "engine"
    request_kwargs: dict = field(default_factory=dict)

    def generate(self, prompt: str, **kwargs) -> str:
        # Local import to keep the module independent of the serving stack at import time.
        from platform.serving.engine import GenRequest

        prompt_ids = list(self.tokenizer.encode(prompt))
        params = dict(self.request_kwargs)
        params.update(kwargs)
        req = GenRequest(prompt_ids=prompt_ids, **params)

        async def _drain() -> list[int]:
            ids: list[int] = []
            async for chunk in self.engine.generate(req):
                if not chunk.get("done"):
                    ids.append(int(chunk["token_id"]))
            return ids

        # If we're already inside a running loop, defer to it; otherwise spin one up.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            gen_ids = asyncio.run(_drain())
        else:  # pragma: no cover - rare in tests
            gen_ids = asyncio.get_event_loop().run_until_complete(_drain())
        return self.tokenizer.decode(gen_ids)
