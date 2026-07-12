"""OpenAI-compatible envelope helpers and request/response value objects.

Framework-agnostic: knows the OpenAI wire shapes (chat and text-completion,
streamed and non-streamed) but nothing about FastAPI or MLX. ``server.py``
wires these into ASGI routes; ``engine.py`` re-exports ``SamplingParams`` and
``Completion`` so older call sites can keep importing from either location.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ModelRef(BaseModel):
    """Body for the Ollama routes that reference one model (`/api/show`, `/api/delete`)."""

    model: str


def _normalize_stop(stop: str | list[str] | None) -> tuple[str, ...]:
    """Coerce a `stop` field to a tuple of non-empty strings.

    A raw string becomes a one-element tuple; empty strings are dropped so a
    stray ``""`` can't halt generation immediately (``str.find("")`` is 0).
    """
    if stop is None:
        return ()
    seqs = [stop] if isinstance(stop, str) else stop
    return tuple(s for s in seqs if s)


class _SamplingRequest(BaseModel):
    """Fields shared by the chat and text completion request bodies."""

    model: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False

    def sampling(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=_normalize_stop(self.stop),
            seed=self.seed,
        )


class ChatRequest(_SamplingRequest):
    messages: list[ChatMessage]


class CompletionRequest(_SamplingRequest):
    prompt: str


@dataclass
class SamplingParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass
class Completion:
    """One streamed generation step.

    `finish_reason` and the token counts are populated only on the terminal
    step ("stop" for EOS, "length" when truncated at max_tokens).
    """

    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


ChoiceBuilder = Callable[[str, str | None], dict[str, Any]]


def _now() -> int:
    return int(time.time())


def _rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chat_choice(content: str, finish: str | None) -> dict[str, Any]:
    delta = {"content": content} if finish is None else {}
    return {"index": 0, "delta": delta, "finish_reason": finish}


def _chat_message_choice(content: str, finish: str | None) -> dict[str, Any]:
    return {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": finish,
    }


def _text_choice(content: str, finish: str | None) -> dict[str, Any]:
    return {"index": 0, "text": content, "finish_reason": finish}


@dataclass(frozen=True, slots=True)
class Shape:
    """Per-endpoint envelope shape: id prefix, object tags, and choice builders.

    ``stream_choice`` emits incremental deltas/text; ``terminal_choice`` emits
    the final non-streaming choice (assistant message or text).
    """

    rid_prefix: str
    stream_obj: str
    nonstream_obj: str
    stream_choice: ChoiceBuilder
    terminal_choice: ChoiceBuilder


CHAT_SHAPE = Shape(
    rid_prefix="chatcmpl",
    stream_obj="chat.completion.chunk",
    nonstream_obj="chat.completion",
    stream_choice=_chat_choice,
    terminal_choice=_chat_message_choice,
)
COMPLETION_SHAPE = Shape(
    rid_prefix="cmpl",
    stream_obj="text_completion",
    nonstream_obj="text_completion",
    stream_choice=_text_choice,
    terminal_choice=_text_choice,
)


def usage(final: Completion | None) -> dict[str, int]:
    """OpenAI `usage` block from the stream's terminal completion."""
    prompt = final.prompt_tokens if final else 0
    completion = final.completion_tokens if final else 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def sse_response(
    chunks: Iterator[Completion],
    *,
    cid: str,
    created: int,
    model: str,
    obj: str,
    choice: ChoiceBuilder,
) -> StreamingResponse:
    """Stream `chunks` as OpenAI SSE frames + a terminal choice, usage, [DONE].

    Errors raised by the generator are emitted in-band as
    `data: {"error": ...}` since once a 200 stream has started we can no
    longer change to an error status.
    """

    def envelope(
        choices: list[dict[str, Any]], usage_block: dict[str, int] | None
    ) -> dict[str, Any]:
        env: dict[str, Any] = {
            "id": cid,
            "object": obj,
            "created": created,
            "model": model,
            "choices": choices,
        }
        if usage_block is not None:
            env["usage"] = usage_block
        return env

    def gen() -> Iterator[str]:
        try:
            final: Completion | None = None
            for chunk in chunks:
                if chunk.text:
                    yield _sse(envelope([choice(chunk.text, None)], None))
                if chunk.finish_reason is not None:
                    final = chunk
            finish = final.finish_reason if final else "stop"
            yield _sse(envelope([choice("", finish)], None))
            # Spec-shaped trailing usage frame (choices empty, usage populated).
            yield _sse(envelope([], usage(final)))
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield _sse({"error": {"message": str(e), "type": type(e).__name__}})

    return StreamingResponse(gen(), media_type="text/event-stream")


def collect(chunks: Iterator[Completion]) -> tuple[str, Completion | None]:
    """Drain a token stream to (text, terminal completion).

    Lets generator exceptions propagate so the caller (the HTTP layer) can
    map them to a 500; this keeps the protocol module free of HTTP concerns.
    """
    parts: list[str] = []
    final: Completion | None = None
    for chunk in chunks:
        parts.append(chunk.text)
        if chunk.finish_reason is not None:
            final = chunk
    return "".join(parts), final


def json_response(
    *,
    cid: str,
    created: int,
    model: str,
    obj: str,
    choices: list[dict[str, Any]],
    usage_completion: Completion | None,
) -> dict[str, Any]:
    """Non-streaming OpenAI response envelope (mirrors `sse_response`)."""
    return {
        "id": cid,
        "object": obj,
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage(usage_completion),
    }
