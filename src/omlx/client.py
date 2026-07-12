"""Thin client for the omlx OpenAI-compatible chat SSE stream."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx2

_DATA_PREFIX = "data: "
_DONE = "[DONE]"


class StreamError(RuntimeError):
    """Raised when the server emits an error frame mid-stream."""


def stream_chat(url: str, body: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield ``(kind, text)`` deltas from a streaming chat-completions response.

    ``kind`` is ``"reasoning"`` for chain-of-thought deltas (reasoning models)
    or ``"content"`` for the answer. Reasoning precedes content within a frame.
    Raises ``StreamError`` if the server reports an error instead of tokens.
    """
    with httpx2.stream("POST", url, json=body, timeout=None) as resp:
        for line in resp.iter_lines():
            if not line.startswith(_DATA_PREFIX):
                continue
            data = line[len(_DATA_PREFIX) :]
            if data == _DONE:
                break
            obj = json.loads(data)
            if "error" in obj:
                raise StreamError(obj["error"]["message"])
            if not obj.get("choices"):
                continue
            delta = obj["choices"][0]["delta"]
            reasoning = delta.get("reasoning_content", "")
            if reasoning:
                yield "reasoning", reasoning
            content = delta.get("content", "")
            if content:
                yield "content", content
