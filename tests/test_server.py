from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from omlx import registry, server
from omlx.engine import Completion


class FakeManager:
    """Stands in for the MLX-backed ModelManager: yields fixed chunks."""

    def __init__(
        self,
        chunks=("hel", "lo"),
        reasonings=None,
        raises=False,
        finish_reason="stop",
        prompt_tokens=5,
        completion_tokens=2,
        tool_calls=None,
    ):
        self.chunks = chunks
        # Per-chunk reasoning deltas, parallel to `chunks`; None => no reasoning.
        self.reasonings = reasonings
        self.raises = raises
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # OpenAI-shaped tool calls to emit on the terminal chunk, if any.
        self.tool_calls = tool_calls
        self.seen_messages = None  # last messages passed to stream_chat
        self.seen_tools = None  # last tools passed to stream_chat
        self.seen_sampling = None  # last SamplingParams passed to stream_chat
        self.seen_response_format = None  # last response_format dict passed to stream_chat/text

    def loaded(self):
        return "Llama" if not self.raises else None

    def loaded_models(self):
        # Mirror the real manager's shape; tests assert on the names list.
        return [] if self.raises else [type("I", (), {"name": "Llama"})()]

    def ps(self):
        if self.raises:
            return []
        return [
            {
                "name": "Llama",
                "model": "Llama",
                "size": 0,
                "size_vram": 0,
                "digest": "",
                "expires_at": None,
            }
        ]

    def _gen(self):
        if self.raises:
            raise RuntimeError("boom")
        last = len(self.chunks) - 1
        terminal_finish = "tool_calls" if self.tool_calls else self.finish_reason
        for i, c in enumerate(self.chunks):
            yield Completion(
                text=c,
                reasoning=self.reasonings[i] if self.reasonings else "",
                finish_reason=terminal_finish if i == last else None,
                prompt_tokens=self.prompt_tokens if i == last else 0,
                completion_tokens=self.completion_tokens if i == last else 0,
                tool_calls=tuple(self.tool_calls) if (self.tool_calls and i == last) else (),
            )

    def stream_chat(self, name, messages, *a, tools=None, response_format=None, **k):
        self.seen_messages = messages
        self.seen_tools = tools
        self.seen_sampling = a[0] if a else None
        self.seen_response_format = response_format
        return self._gen()

    def stream_text(self, *a, response_format=None, **k):
        self.seen_response_format = response_format
        return self._gen()


def _sse_frames(text):
    """Parse the JSON payload of every SSE `data:` line except the [DONE] tail."""
    payloads = [ln[len("data: ") :] for ln in text.splitlines() if ln.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    return [json.loads(p) for p in payloads[:-1]]


def _usage_frame(frames):
    """The trailing usage frame (empty choices, populated usage)."""
    usage = [f for f in frames if f.get("usage")]
    assert len(usage) == 1 and usage[0]["choices"] == []
    return usage[0]


@pytest.fixture
def make_client(monkeypatch, make_entry):
    """Factory: install a (possibly failing) FakeManager and return a client.

    The "Llama" model is pre-registered by default so the unknown-model preflight
    passes for generation routes; pass ``register=False`` to keep the registry
    empty (used by tests asserting on empty-listing shapes).
    """

    def _make(*, register: bool = True, **kw):
        if register:
            registry.add(make_entry(name="Llama", repo_id="org/Llama"))
        monkeypatch.setattr(server, "MANAGER", FakeManager(**kw))
        return TestClient(server.app)

    return _make


@pytest.fixture
def client(make_client):
    return make_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["loaded"] == "Llama"
    # `loaded_models` mirrors the resident set, MRU first (added with multi-model).
    assert body["loaded_models"] == ["Llama"]


def test_health_when_empty(make_client):
    r = make_client(raises=True).get("/health")
    body = r.json()
    assert body["loaded"] is None
    assert body["loaded_models"] == []


def test_api_ps(client):
    r = client.get("/api/ps")
    assert r.status_code == 200
    body = r.json()
    assert body["models"][0]["name"] == "Llama"
    assert "size_vram" in body["models"][0]


def test_api_ps_empty(make_client):
    assert make_client(raises=True).get("/api/ps").json() == {"models": []}


def test_api_version(client):
    r = client.get("/api/version")
    assert r.status_code == 200
    assert set(r.json()) == {"version"}


def test_api_tags(client, make_entry):
    registry.add(make_entry(name="Llama", repo_id="org/Llama", quant="4bit", size_bytes=123))
    body = client.get("/api/tags").json()
    model = body["models"][0]
    assert model["name"] == "Llama" and model["model"] == "Llama"
    assert model["size"] == 123
    assert model["details"]["quantization_level"] == "4bit"
    # modified_at is an RFC-3339 string even when the path doesn't exist on disk.
    assert isinstance(model["modified_at"], str) and "T" in model["modified_at"]


def test_api_tags_empty(make_client):
    assert make_client(register=False).get("/api/tags").json() == {"models": []}


def test_api_show(client, make_entry, tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 8}))
    registry.add(make_entry(name="Llama", repo_id="org/Llama", path=str(snapshot), quant="4bit"))
    body = client.post("/api/show", json={"model": "Llama"}).json()
    assert body["details"]["quantization_level"] == "4bit"
    assert body["model_info"]["model_type"] == "llama"


def test_api_show_without_config_yields_empty_model_info(client, make_entry, tmp_path):
    """A snapshot with no readable config.json still shows, with model_info == {}."""
    registry.add(make_entry(name="Llama", repo_id="org/Llama", path=str(tmp_path), quant="4bit"))
    body = client.post("/api/show", json={"model": "Llama"}).json()
    assert body["model_info"] == {}
    assert body["details"]["quantization_level"] == "4bit"


def test_api_show_missing_is_404(client):
    assert client.post("/api/show", json={"model": "nope"}).status_code == 404


def test_api_delete(client, make_entry, no_purge):
    registry.add(make_entry(name="Llama", repo_id="org/Llama"))
    r = client.request("DELETE", "/api/delete", json={"model": "Llama"})
    assert r.status_code == 200
    assert registry.get("Llama") is None


def test_api_delete_missing_is_404(client):
    r = client.request("DELETE", "/api/delete", json={"model": "nope"})
    assert r.status_code == 404


def test_list_models(client, make_entry):
    registry.add(make_entry(name="Llama", repo_id="org/Llama", quant="4bit"))
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "Llama"
    assert body["data"][0]["quant"] == "4bit"


def test_sampling_maps_openai_fields():
    """Request sampling fields map onto SamplingParams; logit_bias keys coerce to int."""
    from omlx.protocol import ChatRequest

    req = ChatRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        frequency_penalty=0.5,
        presence_penalty=0.25,
        top_k=20,
        min_p=0.1,
        repetition_penalty=1.2,
        logit_bias={"50256": -100.0, "notanint": 1.0},
    )
    s = req.sampling()
    assert s.frequency_penalty == 0.5 and s.presence_penalty == 0.25
    assert s.top_k == 20 and s.min_p == 0.1 and s.repetition_penalty == 1.2
    assert s.logit_bias == {50256: -100.0}  # non-integer key dropped


def test_chat_non_stream(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_chat_non_stream_length_finish(make_client):
    r = make_client(finish_reason="length").post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_chat_accepts_stop_and_seed(client):
    """`stop`/`seed` are valid request fields (wiring covered in test_engine)."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["\n\n"],
            "seed": 7,
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hello"


def test_chat_stream_sse_framing(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    deltas = [f["choices"][0]["delta"].get("content", "") for f in frames if f["choices"]]
    assert "".join(deltas) == "hello"
    # The assistant-role first chunk contributes an empty content string; the
    # subsequent content chunks carry "hel" and "lo".
    assert deltas[0] == ""
    assert _usage_frame(frames)["usage"]["total_tokens"] == 7


def test_chat_stream_first_chunk_carries_assistant_role(client):
    """OpenAI spec: a chat stream's first chunk has a role-only delta."""
    r = client.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    first = frames[0]
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert first["choices"][0]["finish_reason"] is None


def test_chat_stream_usage_omitted_by_default(client):
    """Spec default: no usage frame unless `stream_options.include_usage` is set."""
    r = client.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    frames = _sse_frames(r.text)
    assert not any("usage" in f for f in frames)


def test_chat_stream_include_usage_false_suppresses_frame(client):
    """An explicit ``{"include_usage": false}`` matches the default behavior."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": False},
        },
    )
    frames = _sse_frames(r.text)
    assert not any("usage" in f for f in frames)


def test_completion_stream_has_no_role_chunk(client):
    """Text completions stream bare `text` deltas; no assistant role chunk."""
    r = client.post(
        "/v1/completions",
        json={"model": "Llama", "prompt": "hi", "stream": True},
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    # No role delta on any chunk of a text completion stream.
    assert all("role" not in f["choices"][0] for f in frames if f["choices"])


def test_completion_stream_usage_omitted_by_default(client):
    """Same `include_usage` default applies to text completions."""
    r = client.post(
        "/v1/completions",
        json={"model": "Llama", "prompt": "hi", "stream": True},
    )
    frames = _sse_frames(r.text)
    assert not any("usage" in f for f in frames)


def test_chat_non_stream_includes_reasoning_content(make_client):
    """A reasoning model's analysis rides `message.reasoning_content`."""
    r = make_client(chunks=("Hi",), reasonings=("ponder",)).post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "Hi"
    assert msg["reasoning_content"] == "ponder"


def test_chat_non_stream_omits_reasoning_when_absent(client):
    """Non-reasoning output keeps the byte-for-byte contract: no reasoning key."""
    r = client.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert "reasoning_content" not in r.json()["choices"][0]["message"]


def test_chat_stream_includes_reasoning_content(make_client):
    r = make_client(chunks=("", "Hi"), reasonings=("ponder", "")).post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    frames = _sse_frames(r.text)
    reasonings = [
        f["choices"][0]["delta"].get("reasoning_content", "") for f in frames if f["choices"]
    ]
    contents = [f["choices"][0]["delta"].get("content", "") for f in frames if f["choices"]]
    assert "".join(reasonings) == "ponder"
    assert "".join(contents) == "Hi"


def test_chat_stream_omits_reasoning_when_absent(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    frames = _sse_frames(r.text)
    assert all("reasoning_content" not in f["choices"][0]["delta"] for f in frames if f["choices"])


def test_chat_honors_max_completion_tokens(client):
    """The OpenAI reasoning-model output-cap field reaches SamplingParams."""
    from omlx.protocol import ChatRequest

    req = ChatRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=12345,
    )
    assert req.sampling().max_tokens == 12345


def test_chat_max_completion_tokens_overrides_max_tokens(client):
    """When both are sent, the reasoning-model field wins (OpenAI spec for o-series)."""
    from omlx.protocol import ChatRequest

    req = ChatRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4096,
        max_completion_tokens=16384,
    )
    assert req.sampling().max_tokens == 16384


def test_chat_reasoning_exhausts_budget_reports_length(make_client):
    """An analysis-only stream hitting the cap keeps content empty and reports length.

    Mirrors the gpt-oss failure mode: the model spends the whole output budget in
    the `analysis` channel (reasoning) and never reaches `final`, so `content`
    is empty and the terminal finish is "length", not "stop".
    """
    client = make_client(chunks=("",), reasonings=("pondering",), finish_reason="length")
    r = client.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == ""
    assert msg["reasoning_content"] == "pondering"
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_chat_default_cap_is_reasoning_friendly():
    """The default chat max_tokens is large enough for reasoning effort headroom."""
    from omlx.protocol import ChatRequest

    req = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert req.sampling().max_tokens >= 8192


def test_chat_drops_inbound_reasoning_before_templating(make_client):
    """A client echoing a prior turn's reasoning must not leak it into the prompt."""
    client = make_client()
    client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning_content": "<|channel|>analysis<|message|>secret",
                },
                {"role": "user", "content": "again"},
            ],
        },
    )
    seen = server.MANAGER.seen_messages
    assert all(set(m) == {"role", "content"} for m in seen)  # only role/content templated
    assert all("<|" not in m["content"] for m in seen)  # no channel text reached the template


_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
}
_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "description": "weather", "parameters": {}},
    }
]


def test_chat_accepts_content_parts_array(client):
    """OpenAI structured content (a parts array) is accepted and joined to text."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": " there"}],
                }
            ],
        },
    )
    assert r.status_code == 200
    assert server.MANAGER.seen_messages[0]["content"] == "hi there"


def test_chat_accepts_null_content_with_tool_calls(client):
    """An assistant tool-call turn (content null + tool_calls) templates cleanly."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL]},
                {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
            ],
        },
    )
    assert r.status_code == 200
    seen = server.MANAGER.seen_messages
    assert seen[1]["content"] == ""
    # arguments decoded from JSON string to object for the template
    assert seen[1]["tool_calls"][0]["function"]["arguments"] == {"city": "NYC"}
    assert seen[2]["role"] == "tool" and seen[2]["tool_call_id"] == "call_1"


def test_chat_forwards_tools_to_manager(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "tools": _TOOLS},
    )
    assert r.status_code == 200
    assert server.MANAGER.seen_tools == _TOOLS


def test_chat_non_stream_tool_calls(make_client):
    """A tool call surfaces on the message with finish_reason tool_calls, no index key."""
    r = make_client(chunks=("",), tool_calls=[_TOOL_CALL]).post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "tools": _TOOLS},
    )
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc == _TOOL_CALL  # id/type/function preserved, no `index`
    assert "index" not in tc


def test_chat_stream_tool_calls(make_client):
    """Streamed tool calls arrive as delta.tool_calls with an index; finish tool_calls."""
    r = make_client(chunks=("",), tool_calls=[_TOOL_CALL]).post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "stream": True,
        },
    )
    frames = _sse_frames(r.text)
    tool_deltas = [
        tc for f in frames if f["choices"] for tc in f["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["index"] == 0
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    finishes = [f["choices"][0]["finish_reason"] for f in frames if f["choices"]]
    assert "tool_calls" in finishes


def test_completions_non_stream(client):
    r = client.post("/v1/completions", json={"model": "Llama", "prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["text"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_completions_stream_sse_framing(client):
    r = client.post(
        "/v1/completions",
        json={
            "model": "Llama",
            "prompt": "hi",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    texts = [f["choices"][0].get("text", "") for f in frames if f["choices"]]
    assert "".join(texts) == "hello"
    assert _usage_frame(frames)["usage"]["total_tokens"] == 7


@pytest.mark.parametrize(
    "path, body",
    [
        (
            "/v1/chat/completions",
            {"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/v1/completions", {"model": "Llama", "prompt": "hi"}),
    ],
)
def test_non_stream_error_returns_500(make_client, path, body):
    r = make_client(raises=True).post(path, json=body)
    assert r.status_code == 500


@pytest.mark.parametrize(
    "path, body",
    [
        (
            "/v1/chat/completions",
            {"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ),
        ("/v1/completions", {"model": "Llama", "prompt": "hi", "stream": True}),
    ],
)
def test_stream_error_emits_error_chunk(make_client, path, body):
    r = make_client(raises=True).post(path, json=body)
    assert '"error"' in r.text


# --- Ollama native API ------------------------------------------------------


def _ndjson_lines(text):
    """Parse each non-empty NDJSON line into a JSON object."""
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def test_root_is_ollama_probe(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.text == "Ollama is running"


def test_api_chat_non_stream(client):
    r = client.post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "Llama"
    assert body["message"] == {"role": "assistant", "content": "hello"}
    assert body["done"] is True
    assert body["done_reason"] == "stop"
    assert body["prompt_eval_count"] == 5
    assert body["eval_count"] == 2


def test_api_chat_stream_ndjson_framing(client):
    r = client.post(
        "/api/chat",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    # NDJSON, not SSE: no `data:` prefix and no `[DONE]` sentinel.
    assert "data:" not in r.text and "[DONE]" not in r.text
    frames = _ndjson_lines(r.text)
    assert "".join(f["message"]["content"] for f in frames if not f["done"]) == "hello"
    assert [f["done"] for f in frames][:-1] == [False] * (len(frames) - 1)
    terminal = frames[-1]
    assert terminal["done"] is True
    assert terminal["done_reason"] == "stop"
    assert terminal["eval_count"] == 2


def test_api_chat_defaults_to_streaming(client):
    """Ollama `stream` defaults to true, unlike the OpenAI routes."""
    r = client.post(
        "/api/chat",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    frames = _ndjson_lines(r.text)
    assert len(frames) > 1 and frames[-1]["done"] is True


def test_api_chat_thinking_present(make_client):
    r = make_client(chunks=("answer",), reasonings=("because",)).post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "think": True,
            "stream": False,
        },
    )
    assert r.json()["message"]["thinking"] == "because"


def test_api_chat_thinking_omitted_when_think_false(make_client):
    r = make_client(chunks=("answer",), reasonings=("because",)).post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "think": False,
            "stream": False,
        },
    )
    assert "thinking" not in r.json()["message"]


def test_api_chat_stream_thinking(make_client):
    r = make_client(chunks=("answer",), reasonings=("because",)).post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "think": True,
        },
    )
    frames = _ndjson_lines(r.text)
    assert any(f["message"].get("thinking") == "because" for f in frames if not f["done"])


def test_api_chat_stream_tool_calls_arguments_are_object(make_client):
    r = make_client(chunks=("",), tool_calls=[_TOOL_CALL]).post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
        },
    )
    frames = _ndjson_lines(r.text)
    calls = [tc for f in frames if not f["done"] for tc in f["message"].get("tool_calls", [])]
    assert calls == [{"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}]


def test_api_chat_tool_calls_arguments_are_object(make_client):
    """Ollama tool calls carry `arguments` as an object, not an OpenAI JSON string."""
    r = make_client(chunks=("",), tool_calls=[_TOOL_CALL]).post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "stream": False,
        },
    )
    body = r.json()
    assert body["done_reason"] == "tool_calls"
    tc = body["message"]["tool_calls"][0]
    assert tc == {"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}


def test_api_chat_maps_options_to_sampling(client):
    client.post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 32, "temperature": 0.1, "repeat_penalty": 1.2},
            "stream": False,
        },
    )
    params = server.MANAGER.seen_sampling
    assert params.max_tokens == 32
    assert params.temperature == 0.1
    assert params.repetition_penalty == 1.2


def test_api_generate_non_stream(client):
    r = client.post("/api/generate", json={"model": "Llama", "prompt": "hi", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "hello"
    assert body["done"] is True
    assert body["eval_count"] == 2


def test_api_generate_stream_ndjson(client):
    r = client.post("/api/generate", json={"model": "Llama", "prompt": "hi"})
    frames = _ndjson_lines(r.text)
    assert "".join(f["response"] for f in frames) == "hello"
    assert frames[-1]["done"] is True


def test_api_generate_thinking(make_client):
    r = make_client(chunks=("answer",), reasonings=("because",)).post(
        "/api/generate",
        json={"model": "Llama", "prompt": "hi", "think": True, "stream": False},
    )
    assert r.json()["thinking"] == "because"


def test_api_generate_stream_thinking(make_client):
    r = make_client(chunks=("answer",), reasonings=("because",)).post(
        "/api/generate",
        json={"model": "Llama", "prompt": "hi", "think": True},
    )
    frames = _ndjson_lines(r.text)
    assert any(f.get("thinking") == "because" for f in frames if not f["done"])


def test_api_generate_stream_error_emits_error_line(make_client):
    r = make_client(raises=True).post("/api/generate", json={"model": "Llama", "prompt": "hi"})
    assert '"error"' in r.text


def test_api_generate_raw_uses_bare_prompt(client):
    """`raw` bypasses the chat template (routes through stream_text)."""
    r = client.post(
        "/api/generate",
        json={"model": "Llama", "prompt": "hi", "raw": True, "stream": False},
    )
    assert r.json()["response"] == "hello"
    assert server.MANAGER.seen_messages is None  # stream_chat not used


def test_api_generate_folds_in_system(client):
    client.post(
        "/api/generate",
        json={"model": "Llama", "prompt": "hi", "system": "be terse", "stream": False},
    )
    seen = server.MANAGER.seen_messages
    assert seen[0] == {"role": "system", "content": "be terse"}
    assert seen[1] == {"role": "user", "content": "hi"}


def test_api_generate_non_stream_error_returns_500(make_client):
    r = make_client(raises=True).post(
        "/api/generate", json={"model": "Llama", "prompt": "hi", "stream": False}
    )
    assert r.status_code == 500


def test_api_pull_stream_error_emits_error_line(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no such repo")

    monkeypatch.setattr("omlx.pull.pull", boom)
    r = client.post("/api/pull", json={"model": "org/Repo"})
    frames = _ndjson_lines(r.text)
    assert "error" in frames[-1]


def test_api_chat_non_stream_error_returns_500(make_client):
    r = make_client(raises=True).post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 500


def test_api_chat_stream_error_emits_error_line(make_client):
    r = make_client(raises=True).post(
        "/api/chat",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert '"error"' in r.text


def test_api_pull_non_stream(client, monkeypatch):
    pulled = []
    monkeypatch.setattr("omlx.pull.pull", lambda model, *a, **k: pulled.append(model))
    r = client.post("/api/pull", json={"model": "org/Repo", "stream": False})
    assert r.status_code == 200
    assert r.json() == {"status": "success"}
    assert pulled == ["org/Repo"]


def test_api_pull_stream(client, monkeypatch):
    monkeypatch.setattr("omlx.pull.pull", lambda model, *a, **k: None)
    r = client.post("/api/pull", json={"name": "org/Repo"})  # `name` legacy alias
    frames = _ndjson_lines(r.text)
    assert frames[-1] == {"status": "success"}


def test_api_pull_error_returns_500(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no such repo")

    monkeypatch.setattr("omlx.pull.pull", boom)
    r = client.post("/api/pull", json={"model": "org/Repo", "stream": False})
    assert r.status_code == 500


# --- Bundle 1: OpenAI error envelope, unknown-model 404, RFC-3339 timestamps ---


def test_unknown_model_returns_404_openai_error_body(make_client):
    """Unknown model on a generation route surfaces as 404 with the OpenAI shape."""
    tc = make_client(register=False)  # no models in the registry
    r = tc.post(
        "/v1/chat/completions",
        json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["param"] == "model"
    assert "ghost" in body["error"]["message"]


@pytest.mark.parametrize(
    "path, body",
    [
        (
            "/v1/chat/completions",
            {"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/v1/completions", {"model": "ghost", "prompt": "hi"}),
        (
            "/api/chat",
            {"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/api/generate", {"model": "ghost", "prompt": "hi"}),
    ],
)
def test_unknown_model_returns_404_on_every_generation_route(make_client, path, body):
    """All four generation routes surface unknown-model as 404, preflight-time."""
    tc = make_client(register=False)
    r = tc.post(path, json=body)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.parametrize(
    "path, body",
    [
        (
            "/v1/chat/completions",
            {
                "model": "ghost",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ),
        (
            "/api/chat",
            {"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
        ),
    ],
)
def test_unknown_model_stream_returns_404(make_client, path, body):
    """A streaming request for an unknown model returns 404 (no in-band error frame)."""
    tc = make_client(register=False)
    r = tc.post(path, json=body)
    assert r.status_code == 404
    assert "[DONE]" not in r.text  # never started streaming
    assert r.json()["error"]["code"] == "model_not_found"


def test_unknown_model_in_api_show_returns_404_openai_body(make_client):
    tc = make_client(register=False)
    r = tc.post("/api/show", json={"model": "ghost"})
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["type"] == "not_found_error"
    assert body["error"]["code"] == "model_not_found"


def test_unknown_model_in_api_delete_returns_404_openai_body(make_client):
    tc = make_client(register=False)
    r = tc.request("DELETE", "/api/delete", json={"model": "ghost"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.parametrize(
    "path, body",
    [
        # Missing required `messages` on chat.
        ("/v1/chat/completions", {"model": "Llama"}),
        # Missing required `prompt` on completions.
        ("/v1/completions", {"model": "Llama"}),
    ],
)
def test_malformed_request_returns_400_openai_body(make_client, path, body):
    """Pydantic validation failures render as the OpenAI error envelope (400)."""
    tc = make_client()
    r = tc.post(path, json=body)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_request"


def test_internal_error_returns_500_openai_body(make_client):
    """Generation failures return 500 with the OpenAI error envelope (pinned).

    Downgraded from FastAPI's `{"detail": ...}` to the spec shape; status and
    pinned test contract preserved.
    """
    tc = make_client(raises=True)
    r = tc.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["type"] == "internal_error"
    assert "boom" in err["message"]


def test_api_chat_non_stream_internal_error_returns_500_openai_body(make_client):
    tc = make_client(raises=True)
    r = tc.post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "internal_error"


def test_api_generate_non_stream_internal_error_returns_500_openai_body(make_client):
    tc = make_client(raises=True)
    r = tc.post("/api/generate", json={"model": "Llama", "prompt": "hi", "stream": False})
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "internal_error"


def test_api_pull_error_returns_500_openai_body(make_client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no such repo")

    monkeypatch.setattr("omlx.pull.pull", boom)
    tc = make_client()
    r = tc.post("/api/pull", json={"model": "org/Repo", "stream": False})
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "internal_error"


def test_timestamps_use_rfc3339_z_suffix(make_client):
    """Ollama-shaped timestamps carry fractional seconds and a `Z` suffix."""
    from datetime import datetime

    tc = make_client()
    r = tc.post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    created_at = r.json()["created_at"]
    # RFC-3339 with microseconds and a literal `Z` (no `+00:00`).
    assert created_at.endswith("Z")
    assert "." in created_at  # fractional seconds present
    # Round-trip parse: drop the Z, parse as naive UTC, succeeds.
    datetime.fromisoformat(created_at.replace("Z", "+00:00"))


def test_api_tags_modified_at_uses_rfc3339_z_suffix(client, make_entry):
    from datetime import datetime

    registry.add(make_entry(name="Llama2", repo_id="org/Llama2", quant="4bit"))
    model = client.get("/api/tags").json()["models"][0]
    modified_at = model["modified_at"]
    assert modified_at.endswith("Z")
    assert "." in modified_at
    datetime.fromisoformat(modified_at.replace("Z", "+00:00"))


def test_openai_error_body_shape():
    """The error-object builder emits the spec fields in spec order."""
    from omlx.protocol import openai_error_body

    body = openai_error_body(
        "boom", type="invalid_request_error", code="bad_value", param="temperature"
    )
    assert body == {
        "error": {
            "message": "boom",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "bad_value",
        }
    }
    # `code` is optional and omitted when not provided (per OpenAI spec).
    assert "code" not in openai_error_body("x", type="internal_error")["error"]


def test_openai_error_carries_status_and_codes():
    from omlx.protocol import OpenAIError

    e = OpenAIError(
        "nope",
        status=404,
        type="not_found_error",
        code="model_not_found",
        param="model",
    )
    assert e.status == 404
    assert e.type == "not_found_error" and e.code == "model_not_found"
    assert e.param == "model"
    assert str(e) == "nope"


# --- Bundle 3: response_format / Ollama `format` (JSON mode) ------------------


def test_chat_response_format_json_object_forwards_to_engine(client):
    """A `json_object` request reaches `stream_chat` with the response_format dict."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "give a JSON example"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert r.status_code == 200
    rf = server.MANAGER.seen_response_format
    assert rf == {"type": "json_object", "json_schema": None}


def test_completions_response_format_json_object_forwards_to_engine(client):
    r = client.post(
        "/v1/completions",
        json={
            "model": "Llama",
            "prompt": "give a JSON example",
            "response_format": {"type": "json_object"},
        },
    )
    assert r.status_code == 200
    assert server.MANAGER.seen_response_format == {"type": "json_object", "json_schema": None}


def test_chat_response_format_text_is_a_no_op(client):
    """`{"type":"text"}` is OpenAI's pass-through: the engine gets None (no mask)."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"},
        },
    )
    assert r.status_code == 200
    assert server.MANAGER.seen_response_format is None


def test_chat_response_format_json_schema_returns_400_unsupported(make_client):
    tc = make_client()
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "unsupported"
    assert err["param"] == "response_format"
    assert server.MANAGER.seen_response_format is None  # never reached the engine


def test_chat_response_format_and_tools_returns_400_mutually_exclusive(make_client):
    tc = make_client()
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "weather in SF?"}],
            "response_format": {"type": "json_object"},
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "response_format_and_tools_mutually_exclusive"
    assert err["param"] == "response_format"


def test_api_chat_format_json_forwards_to_engine(client):
    """Ollama `format: "json"` maps to `ResponseFormat(type="json_object")`."""
    r = client.post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "give a JSON example"}],
            "format": "json",
            "stream": False,
        },
    )
    assert r.status_code == 200
    assert server.MANAGER.seen_response_format == {"type": "json_object", "json_schema": None}


def test_api_generate_format_json_routes_to_json_mode(client):
    client.post(
        "/api/generate",
        json={
            "model": "Llama",
            "prompt": "give a JSON example",
            "format": "json",
            "stream": False,
        },
    )
    assert server.MANAGER.seen_response_format == {"type": "json_object", "json_schema": None}


def test_api_chat_format_schema_dict_returns_400_unsupported(make_client):
    tc = make_client()
    r = tc.post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "format": {"type": "object", "properties": {}},
            "stream": False,
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported"
    assert err["param"] == "format"


def test_api_chat_format_and_tools_returns_400_mutually_exclusive(make_client):
    tc = make_client()
    r = tc.post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "weather in SF?"}],
            "format": "json",
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        },
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "response_format_and_tools_mutually_exclusive"
    assert err["param"] == "format"


def test_api_generate_format_text_is_a_no_op(client):
    r = client.post(
        "/api/generate",
        json={
            "model": "Llama",
            "prompt": "hi",
            "format": "text",
            "stream": False,
        },
    )
    assert r.status_code == 200
    assert server.MANAGER.seen_response_format is None


def test_api_chat_format_unknown_string_returns_400(make_client):
    tc = make_client()
    r = tc.post(
        "/api/chat",
        json={
            "model": "Llama",
            "messages": [{"role": "user", "content": "hi"}],
            "format": "yaml",
            "stream": False,
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported"


def test_unhandled_exception_returns_500_openai_body(make_client, monkeypatch):
    """An exception that escapes a route (no local try/except) hits the catchall
    and surfaces as a 500 internal_error with the OpenAI envelope."""
    from fastapi.testclient import TestClient

    from omlx import registry as reg
    from omlx import server as srv

    def boom():
        raise RuntimeError("registry corrupted")

    monkeypatch.setattr(reg, "entries", boom)
    # `raise_server_exceptions=False` lets the registered handler produce the
    # 500 response instead of TestClient re-raising the underlying error.
    make_client()
    tc = TestClient(srv.app, raise_server_exceptions=False)
    r = tc.get("/api/tags")
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "internal_error"


def test_registry_vanished_between_preflight_and_load_returns_404(make_client, monkeypatch):
    """Defensive `except OpenAIError: raise` in `_complete` re-raises when an
    OpenAIError leaks past the preflight (a registry-race where the entry is
    removed between the preflight and the engine's `get()`).
    """
    from omlx.protocol import OpenAIError

    class _RaceyManager(FakeManager):
        # Preflight passes (registry has Llama); the generator raises anyway.
        def _gen(self):
            raise OpenAIError(
                "model 'Llama' not found",
                status=404,
                type="not_found_error",
                code="model_not_found",
                param="model",
            )
            yield  # make this a generator

    tc = make_client()
    monkeypatch.setattr(server, "MANAGER", _RaceyManager())
    r = tc.post(
        "/v1/chat/completions",
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.parametrize("route", ["/api/chat", "/api/generate"])
def test_api_routes_reraise_openai_error_past_preflight(make_client, monkeypatch, route):
    """Same race-defense on the Ollama routes: a post-preflight `OpenAIError`
    must surface as 404, not a 500 in-band NDJSON error line.
    """
    from omlx.protocol import OpenAIError

    class _RaceyManager(FakeManager):
        def _gen(self):
            raise OpenAIError(
                "model 'Llama' not found",
                status=404,
                type="not_found_error",
                code="model_not_found",
                param="model",
            )
            yield

    tc = make_client()
    monkeypatch.setattr(server, "MANAGER", _RaceyManager())
    body = (
        {"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        if route == "/api/chat"
        else {"model": "Llama", "prompt": "hi", "stream": False}
    )
    r = tc.post(route, json=body)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
