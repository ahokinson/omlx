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
        raises=False,
        finish_reason="stop",
        prompt_tokens=5,
        completion_tokens=2,
    ):
        self.chunks = chunks
        self.raises = raises
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

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
        for i, c in enumerate(self.chunks):
            yield Completion(
                text=c,
                finish_reason=self.finish_reason if i == last else None,
                prompt_tokens=self.prompt_tokens if i == last else 0,
                completion_tokens=self.completion_tokens if i == last else 0,
            )

    def stream_chat(self, *a, **k):
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
