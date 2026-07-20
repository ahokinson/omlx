"""HTTP server backed by mlx-lm.

Serves the OpenAI surface (`/v1/models`, `/v1/chat/completions`,
`/v1/completions`, SSE `data:` framing) alongside the Ollama native surface:
generation via `/api/chat` and `/api/generate` (NDJSON framing), `/api/pull`,
the `GET /` liveness probe, and the read/admin routes `/api/ps`, `/api/version`,
`/api/tags`, `/api/show`, and `/api/delete`. The wire shapes live in
:mod:`omlx.protocol`; this module is the FastAPI layer wiring them to the model
manager and registry.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from . import __version__, registry
from .engine import MANAGER, Completion
from .protocol import (
    CHAT_SHAPE,
    COMPLETION_SHAPE,
    ChatRequest,
    CompletionRequest,
    ModelRef,
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaPullRequest,
    OpenAIError,
    ResponseFormat,
    Shape,
    _now,
    _rid,
    _SamplingRequest,
    collect,
    json_response,
    ollama_chat_collect,
    ollama_chat_response,
    ollama_generate_collect,
    ollama_generate_response,
    openai_error_body,
    sse_response,
)

app = FastAPI(title="omlx", version=__version__)
logger = logging.getLogger("omlx")


@app.exception_handler(OpenAIError)
async def _openai_error_handler(_req: Request, exc: OpenAIError) -> JSONResponse:
    """Surface `OpenAIError` as the OpenAI error body with its declared status."""
    return JSONResponse(
        status_code=exc.status,
        content=openai_error_body(exc.message, type=exc.type, code=exc.code, param=exc.param),
    )


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(req: Request, exc: RequestValidationError) -> JSONResponse:
    """Malformed request bodies return the OpenAI error shape (400), not
    FastAPI's default ``{"detail": [...]}`` envelope.
    """
    from fastapi.encoders import jsonable_encoder

    return JSONResponse(
        status_code=400,
        content=openai_error_body(
            json.dumps(jsonable_encoder(exc.errors()), default=str),
            type="invalid_request_error",
            code="invalid_request",
            param=None,
        ),
    )


@app.exception_handler(Exception)
async def _internal_error_handler(req: Request, exc: Exception) -> JSONResponse:
    """Catchall: any unhandled failure becomes an OpenAI internal_error (500).

    `ClientDisconnect` is excluded — a peer going away mid-response isn't an
    error on our side, and re-rendering a body would just log noise. The
    `HTTPException` branch is defensive: no route currently raises one, but if
    that ever changes the status is preserved and the body is reshaped.
    """
    if isinstance(exc, ClientDisconnect):  # pragma: no cover - peer hung up
        raise exc
    if isinstance(exc, HTTPException):  # pragma: no cover - defensive
        status = exc.status_code
        type_ = "invalid_request_error" if status < 500 else "internal_error"
        return JSONResponse(
            status_code=status,
            content=openai_error_body(str(exc.detail), type=type_, code=None),
        )
    logger.exception("unhandled error on %s %s", req.method, req.url.path)
    return JSONResponse(
        status_code=500,
        content=openai_error_body(
            str(exc) or exc.__class__.__name__, type="internal_error", code=None
        ),
    )


def _complete(
    req: _SamplingRequest,
    shape: Shape,
    stream_fn: Callable[[], Iterator[Completion]],
) -> StreamingResponse | dict[str, Any]:
    """Shared skeleton for the chat/completions endpoints.

    Streams an SSE chunk stream when ``req.stream`` is set (error frames are
    surfaced in-band as a `data: {"error": ...}` chunk, since once we've
    started a 200 stream we can no longer swap to an error status); otherwise
    drains the stream and returns a single JSON envelope. ``OpenAIError``
    (model-not-found, etc.) propagates with its declared status to the
    exception handler; any other failure becomes a 500 internal_error there.

    Stream parity: the leading assistant-role chunk is emitted only when the
    shape carries it (chat only), and the trailing usage frame is gated on
    ``stream_options.include_usage`` (spec default off).
    """
    cid, created = _rid(shape.rid_prefix), _now()
    if req.stream:
        include_usage = bool(req.stream_options and req.stream_options.include_usage)
        return sse_response(
            stream_fn(),
            cid=cid,
            created=created,
            model=req.model,
            obj=shape.stream_obj,
            choice=shape.stream_choice,
            include_usage=include_usage,
            first_chunk_role=shape.stream_first_role,
        )
    try:
        content, reasoning, tool_calls, final = collect(stream_fn())
    except OpenAIError:
        raise
    except Exception as e:
        raise OpenAIError(
            str(e) or e.__class__.__name__,
            status=500,
            type="internal_error",
        ) from e
    finish = final.finish_reason if final else "stop"
    return json_response(
        cid=cid,
        created=created,
        model=req.model,
        obj=shape.nonstream_obj,
        choices=[shape.terminal_choice(content, finish, reasoning, tool_calls)],
        usage_completion=final,
    )


@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    """Ollama root liveness probe; clients expect the literal `Ollama is running`."""
    return "Ollama is running"


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
    """Snapshot mtime as an RFC-3339 string, falling back to now if the path is gone.

    Matches Ollama's ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` shape; non-Apple-Silicon
    clients expecting strict RFC-3339 with a ``Z`` suffix parse cleanly.
    """
    try:
        mtime = os.stat(entry.path).st_mtime
    except OSError:
        mtime = _now()
    return datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


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
        raise OpenAIError(
            f"model {ref.model!r} not found",
            status=404,
            type="not_found_error",
            code="model_not_found",
            param="model",
        )
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
        raise OpenAIError(
            f"model {ref.model!r} not found",
            status=404,
            type="not_found_error",
            code="model_not_found",
            param="model",
        )
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
    rf = _check_response_format(req.response_format, req.tools)
    _require_known_model(req.model)
    # Messages are flattened to the template shape (content parts joined, tool
    # history preserved); inbound `reasoning_content` is dropped.
    messages = [m.to_template_dict() for m in req.messages]
    rf_dict = rf.model_dump() if rf is not None else None
    return _complete(
        req,
        CHAT_SHAPE,
        lambda: MANAGER.stream_chat(
            req.model, messages, req.sampling(), tools=req.tools, response_format=rf_dict
        ),
    )


@app.post("/v1/completions", response_model=None)
def completions(req: CompletionRequest) -> StreamingResponse | dict[str, Any]:
    """Text completion; streams SSE when `stream` is set, else returns JSON."""
    rf = _check_response_format(req.response_format, None)
    _require_known_model(req.model)
    rf_dict = rf.model_dump() if rf is not None else None
    return _complete(
        req,
        COMPLETION_SHAPE,
        lambda: MANAGER.stream_text(req.model, req.prompt, req.sampling(), response_format=rf_dict),
    )


def _require_known_model(name: str) -> None:
    """Eager preflight so an unknown model surfaces as a 404 before any 200 stream.

    Once a streaming response has started (status 200 sent), we can no longer
    swap to an error status, so model-not-found must be raised *before* the
    `StreamingResponse` is constructed. Known models that later fail to load
    still surface in-band as a stream error frame (a genuine 500 path).
    """
    if registry.get(name) is None:
        raise OpenAIError(
            f"model {name!r} not found",
            status=404,
            type="not_found_error",
            code="model_not_found",
            param="model",
        )


def _check_response_format(
    rf: ResponseFormat | None, tools: list[dict[str, Any]] | None
) -> ResponseFormat | None:
    """Validate a request's `response_format` against omlx's supported subset.

    - ``{"type": "text"}`` is a no-op (OpenAI parity) and is returned as None
      so the engine doesn't add a no-op logits processor.
    - ``{"type": "json_object"}`` requests JSON-mode masking. Mutually
      exclusive with `tools` (per OpenAI); sending both returns
      ``400 response_format_and_tools_mutually_exclusive``.
    - ``{"type": "json_schema", ...}`` is unsupported in this round and
      raises ``400 unsupported`` so callers learn rather than get bare output.

    A `None` request format passes through unchanged.
    """
    if rf is None:
        return None
    if rf.type == "text":
        return None
    if rf.type == "json_object":
        if tools:
            raise OpenAIError(
                "`response_format` and `tools` are mutually exclusive",
                status=400,
                type="invalid_request_error",
                code="response_format_and_tools_mutually_exclusive",
                param="response_format",
            )
        return rf
    # Anything else (json_schema or unknown) — unsupported in this round.
    raise OpenAIError(
        f"`response_format.type={rf.type!r}` is not supported (use 'text' or 'json_object')",
        status=400,
        type="invalid_request_error",
        code="unsupported",
        param="response_format",
    )


def _normalize_ollama_format(
    format: str | dict[str, Any] | None, tools: list[dict[str, Any]] | None
) -> ResponseFormat | None:
    """Translate an Ollama `format` request field onto `ResponseFormat`.

    - Bare ``"json"`` string requests JSON-mode masking.
    - Bare ``"text"`` (Ollama's documented pass-through) is a no-op.
    - A dict (a JSON schema) requests schema-typed output, rejected with
      ``400 unsupported`` until `json_schema` lands.
    - Any other string is treated as a schema name and rejected likewise.

    Honors the same `tools` mutex as the OpenAI side, since the JSON mask and
    tool-call parsing can't share a request.
    """
    if format is None:
        return None
    if isinstance(format, dict):
        raise OpenAIError(
            '`format` as a JSON schema is not supported (use the string "json")',
            status=400,
            type="invalid_request_error",
            code="unsupported",
            param="format",
        )
    if format == "json":
        if tools:
            raise OpenAIError(
                "`format` and `tools` are mutually exclusive",
                status=400,
                type="invalid_request_error",
                code="response_format_and_tools_mutually_exclusive",
                param="format",
            )
        return ResponseFormat(type="json_object")
    if format == "text":
        return None
    raise OpenAIError(
        f'`format={format!r}` is not supported (use "json" or "text")',
        status=400,
        type="invalid_request_error",
        code="unsupported",
        param="format",
    )


# Default max_tokens for the Ollama routes, matching `_SamplingRequest.max_tokens`
# (Ollama's own default is unbounded, but a cap avoids truncating agentic replies
# mid-JSON — see `protocol._SamplingRequest`).
_OLLAMA_DEFAULT_MAX_TOKENS = 4096


@app.post("/api/chat", response_model=None)
def api_chat(req: OllamaChatRequest) -> StreamingResponse | dict[str, Any]:
    """Ollama `/api/chat`; streams NDJSON when `stream` (default true), else JSON.

    Reasoning rides `message.thinking`: emitted when the request sets `think`, or
    (when `think` is unset) whenever the model produces reasoning at all.
    """
    rf = _normalize_ollama_format(req.format, req.tools)
    _require_known_model(req.model)
    messages = [m.to_template_dict() for m in req.messages]
    think = True if req.think is None else req.think
    params = req.options.to_sampling(_OLLAMA_DEFAULT_MAX_TOKENS)
    rf_dict = rf.model_dump() if rf is not None else None
    chunks = MANAGER.stream_chat(
        req.model, messages, params, tools=req.tools, response_format=rf_dict
    )
    if req.stream:
        return ollama_chat_response(chunks, model=req.model, think=think)
    try:
        return ollama_chat_collect(chunks, model=req.model, think=think)
    except OpenAIError:
        raise
    except Exception as e:
        raise OpenAIError(str(e) or e.__class__.__name__, status=500, type="internal_error") from e


@app.post("/api/generate", response_model=None)
def api_generate(req: OllamaGenerateRequest) -> StreamingResponse | dict[str, Any]:
    """Ollama `/api/generate`; streams NDJSON when `stream` (default true), else JSON.

    The prompt is templated through the model's chat template by default (Ollama
    parity), optionally with a leading `system` message; `raw` streams the bare
    prompt with no template.
    """
    rf = _normalize_ollama_format(req.format, None)
    _require_known_model(req.model)
    think = True if req.think is None else req.think
    params = req.options.to_sampling(_OLLAMA_DEFAULT_MAX_TOKENS)
    rf_dict = rf.model_dump() if rf is not None else None
    if req.raw:
        chunks = MANAGER.stream_text(req.model, req.prompt, params, response_format=rf_dict)
    else:
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        chunks = MANAGER.stream_chat(req.model, messages, params, response_format=rf_dict)
    if req.stream:
        return ollama_generate_response(chunks, model=req.model, think=think)
    try:
        return ollama_generate_collect(chunks, model=req.model, think=think)
    except OpenAIError:
        raise
    except Exception as e:
        raise OpenAIError(str(e) or e.__class__.__name__, status=500, type="internal_error") from e


@app.post("/api/pull", response_model=None)
def api_pull(req: OllamaPullRequest) -> StreamingResponse | dict[str, str]:
    """Ollama `/api/pull`; downloads and registers a model.

    The Hugging Face download is blocking, so streamed progress is coarse — a
    `pulling` frame then `success` — rather than the per-byte layered progress
    Ollama emits. Non-stream returns `{"status": "success"}`, HTTP 500 on failure.
    """
    from .pull import pull

    if req.stream:

        def gen() -> Iterator[str]:
            yield json.dumps({"status": f"pulling {req.model}"}) + "\n"
            try:
                pull(req.model)
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"
                return
            yield json.dumps({"status": "success"}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")
    try:
        pull(req.model)
    except Exception as e:
        raise OpenAIError(str(e) or e.__class__.__name__, status=500, type="internal_error") from e
    return {"status": "success"}
