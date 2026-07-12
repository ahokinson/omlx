from __future__ import annotations

import json

import pytest

from omlx import client
from omlx.client import StreamError, stream_chat


class FakeStream:
    """Mimics httpx2.stream(...) as a context manager over SSE lines."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self._lines


def _install(monkeypatch, lines):
    monkeypatch.setattr(client.httpx2, "stream", lambda *a, **k: FakeStream(lines))


def _delta(content):
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})


def _reasoning_delta(reasoning, content=""):
    delta = {"reasoning_content": reasoning}
    if content:
        delta["content"] = content
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


def _usage():
    return "data: " + json.dumps({"choices": [], "usage": {"total_tokens": 3}})


def test_stream_chat_yields_content_until_done(monkeypatch):
    _install(monkeypatch, [_delta("he"), "", _delta("llo"), "data: [DONE]", _delta("ignored")])
    assert list(stream_chat("http://x/v1/chat/completions", {})) == [
        ("content", "he"),
        ("content", "llo"),
    ]


def test_stream_chat_tags_reasoning_before_content(monkeypatch):
    """Reasoning deltas are tagged `reasoning`; a mixed frame emits reasoning first."""
    _install(
        monkeypatch, [_reasoning_delta("think"), _reasoning_delta("more", "ans"), "data: [DONE]"]
    )
    assert list(stream_chat("http://x", {})) == [
        ("reasoning", "think"),
        ("reasoning", "more"),
        ("content", "ans"),
    ]


def test_stream_chat_skips_non_data_lines(monkeypatch):
    _install(monkeypatch, ["event: ping", _delta("hi"), "data: [DONE]"])
    assert list(stream_chat("http://x", {})) == [("content", "hi")]


def test_stream_chat_skips_usage_frame(monkeypatch):
    _install(monkeypatch, [_delta("hi"), _usage(), "data: [DONE]"])
    assert list(stream_chat("http://x", {})) == [("content", "hi")]


def test_stream_chat_raises_on_error_frame(monkeypatch):
    _install(monkeypatch, ["data: " + json.dumps({"error": {"message": "boom"}})])
    with pytest.raises(StreamError, match="boom"):
        list(stream_chat("http://x", {}))
