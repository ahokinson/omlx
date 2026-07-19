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

    def stream_chat(self, name, messages, *a, tools=None, **k):
        self.seen_messages = messages
        self.seen_tools = tools
        self.seen_sampling = a[0] if a else None
        return self._gen()

    def stream_text(self, *a, **k):
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
def make_client(monkeypatch):
    """Factory: install a (possibly failing) FakeManager and return a client."""

    def _make(**kw):
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


def test_api_tags_empty(client):
    assert client.get("/api/tags").json() == {"models": []}


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
        json={"model": "Llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    deltas = [f["choices"][0]["delta"].get("content", "") for f in frames if f["choices"]]
    assert "".join(deltas) == "hello"
    assert _usage_frame(frames)["usage"]["total_tokens"] == 7


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
    r = client.post("/v1/completions", json={"model": "Llama", "prompt": "hi", "stream": True})
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
