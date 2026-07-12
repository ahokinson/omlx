"""HTTP server backed by mlx-lm.

Serves the OpenAI surface (`/v1/models`, `/v1/chat/completions`,
`/v1/completions`, SSE `data:` framing) plus the Ollama read/admin routes
`/api/ps`, `/api/version`, `/api/tags`, `/api/show`, and `/api/delete`.
The wire shapes live in :mod:`omlx.protocol`; this module is the FastAPI layer
wiring them to the model manager and registry.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import __version__, registry
from .engine import MANAGER, Completion
from .protocol import (
    CHAT_SHAPE,
    COMPLETION_SHAPE,
    ChatRequest,
    CompletionRequest,
    ModelRef,
    Shape,
    _now,
    _rid,
    _SamplingRequest,
    collect,
    json_response,
    sse_response,
)

app = FastAPI(title="omlx", version=__version__)


def _complete(
    req: _SamplingRequest,
    shape: Shape,
    stream_fn: Callable[[], Iterator[Completion]],
) -> StreamingResponse | dict[str, Any]:
    """Shared skeleton for the chat/completions endpoints.

    Streams an SSE chunk stream when ``req.stream`` is set (error frames are
    surfaced in-band as a `data: {"error": ...}` chunk, since once we've
    started a 200 stream we can no longer swap to an error status); otherwise
    drains the stream and returns a single JSON envelope, raising HTTP 500 on
    a load/generate failure.
    """
    cid, created = _rid(shape.rid_prefix), _now()
    if req.stream:
        return sse_response(
            stream_fn(),
            cid=cid,
            created=created,
            model=req.model,
            obj=shape.stream_obj,
            choice=shape.stream_choice,
        )
    try:
        content, reasoning, final = collect(stream_fn())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finish = final.finish_reason if final else "stop"
    return json_response(
        cid=cid,
        created=created,
        model=req.model,
        obj=shape.nonstream_obj,
        choices=[shape.terminal_choice(content, finish, reasoning)],
        usage_completion=final,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe; reports loaded models.

    `loaded` is the most-recently-used resident model (or null), for clients
    that expect a single model; `loaded_models` lists all resident models,
    MRU first.
    """
    return {
        "status": "ok",
        "loaded": MANAGER.loaded(),
        "loaded_models": [info.name for info in MANAGER.loaded_models()],
    }


@app.get("/api/ps")
def ps() -> dict[str, Any]:
    """Ollama-style listing of currently loaded models."""
    return {"models": MANAGER.ps()}


@app.get("/api/version")
def version() -> dict[str, str]:
    """Ollama `/api/version`."""
    return {"version": __version__}


def _ollama_details(entry: registry.ModelEntry) -> dict[str, Any]:
    """Ollama `details` block for `/api/tags` and `/api/show`.

    Fields omlx doesn't record (family, parameter_size) are left empty rather
    than guessed.
    """
    return {
        "parent_model": "",
        "format": "mlx",
        "family": "",
        "families": None,
        "parameter_size": "",
        "quantization_level": entry.quant or "",
    }


def _modified_at(entry: registry.ModelEntry) -> str:
    """Snapshot mtime as an ISO-8601 string, falling back to now if the path is gone."""
    try:
        mtime = os.stat(entry.path).st_mtime
    except OSError:
        mtime = _now()
    return datetime.fromtimestamp(mtime, timezone.utc).isoformat()


def _tag_entry(entry: registry.ModelEntry) -> dict[str, Any]:
    """One model in Ollama `/api/tags` shape."""
    return {
        "name": entry.name,
        "model": entry.name,
        "modified_at": _modified_at(entry),
        "size": entry.size_bytes,
        # No content digest: weights live in the HF cache, not a blob store.
        # Empty string matches /api/ps.
        "digest": "",
        "details": _ollama_details(entry),
    }


@app.get("/api/tags")
def tags() -> dict[str, Any]:
    """Ollama `/api/tags`: registered local models."""
    return {"models": [_tag_entry(e) for e in registry.entries()]}


def _read_config(entry: registry.ModelEntry) -> dict[str, Any]:
    """The snapshot's `config.json` as a dict, or `{}` if missing/unreadable."""
    try:
        return json.loads((Path(entry.path) / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


@app.post("/api/show")
def show(ref: ModelRef) -> dict[str, Any]:
    """Ollama `/api/show`: details + the raw HF config as `model_info`."""
    entry = registry.get(ref.model)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"model {ref.model!r} not found")
    return {
        "details": _ollama_details(entry),
        "model_info": _read_config(entry),
        "modelfile": "",
        "parameters": "",
        "template": "",
    }


@app.delete("/api/delete")
def delete(ref: ModelRef) -> dict[str, str]:
    """Ollama `/api/delete`: drop a model from the registry and purge its weights."""
    if registry.remove(ref.model) is None:
        raise HTTPException(status_code=404, detail=f"model {ref.model!r} not found")
    return {"status": "success"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """List registered models in OpenAI `/v1/models` shape."""
    created = _now()
    data = [
        {
            "id": e.name,
            "object": "model",
            "created": created,
            "owned_by": "omlx",
            "repo_id": e.repo_id,
            "quant": e.quant,
        }
        for e in registry.entries()
    ]
    return {"object": "list", "data": data}


# response_model=None: the return is a StreamingResponse-or-dict union, which
# FastAPI can't turn into a response schema; skip inference and pass it through.
@app.post("/v1/chat/completions", response_model=None)
def chat_completions(req: ChatRequest) -> StreamingResponse | dict[str, Any]:
    """Chat completion; streams SSE when `stream` is set, else returns JSON."""
    # Only role and content reach the chat template; inbound `reasoning_content`
    # is dropped.
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    return _complete(
        req,
        CHAT_SHAPE,
        lambda: MANAGER.stream_chat(req.model, messages, req.sampling()),
    )


@app.post("/v1/completions", response_model=None)
def completions(req: CompletionRequest) -> StreamingResponse | dict[str, Any]:
    """Text completion; streams SSE when `stream` is set, else returns JSON."""
    return _complete(
        req,
        COMPLETION_SHAPE,
        lambda: MANAGER.stream_text(req.model, req.prompt, req.sampling()),
    )
