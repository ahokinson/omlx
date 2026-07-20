"""Wire-shape envelope helpers and request/response value objects.

Framework-agnostic: knows the OpenAI wire shapes (chat and text-completion, SSE)
and the Ollama native shapes (`/api/chat`, `/api/generate`, NDJSON), streamed and
non-streamed, but nothing about FastAPI or MLX. ``server.py`` wires these into
ASGI routes; ``engine.py`` re-exports ``SamplingParams`` and ``Completion`` so
older call sites can keep importing from either location.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field


class ContentPart(BaseModel):
    """One element of an OpenAI structured-content array (`{"type","text"}`)."""

    type: str
    text: str | None = None


def _decode_tool_call_args(tc: dict[str, Any]) -> dict[str, Any]:
    """Copy a tool_call with ``function.arguments`` decoded from JSON string to object.

    Chat templates render arguments as an object; the OpenAI wire form is a JSON
    string. A non-decodable string is left as-is.
    """
    out = dict(tc)
    func = out.get("function")
    if isinstance(func, dict) and isinstance(func.get("arguments"), str):
        func = dict(func)
        try:
            func["arguments"] = json.loads(func["arguments"]) if func["arguments"] else {}
        except json.JSONDecodeError:
            pass
        out["function"] = func
    return out


class ChatMessage(BaseModel):
    """One OpenAI chat message: user/assistant/system/tool, with optional tool calls.

    `content` accepts a plain string, a structured content-parts array, or null
    (an assistant tool-call turn carries its calls in `tool_calls`, not content).
    """

    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    # Accepted from clients; dropped before templating (see
    # ``server.chat_completions``).
    reasoning_content: str | None = None

    def to_template_dict(self) -> dict[str, Any]:
        """Flatten to the dict shape ``apply_chat_template`` expects.

        Content parts are joined to text and null content becomes ``""``;
        `reasoning_content` is dropped. `name`, `tool_call_id`, and `tool_calls`
        ride through only when present, the latter with arguments decoded to an
        object.
        """
        content = self.content
        if isinstance(content, list):
            content = "".join(p.text or "" for p in content if p.type == "text")
        elif content is None:
            content = ""
        msg: dict[str, Any] = {"role": self.role, "content": content}
        if self.name is not None:
            msg["name"] = self.name
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = [_decode_tool_call_args(tc) for tc in self.tool_calls]
        return msg


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


def _int_keyed(bias: dict[str, float] | None) -> dict[int, float] | None:
    """Coerce an OpenAI `logit_bias` (token-id string keys) to int keys.

    Non-integer keys are dropped; an empty or all-invalid map becomes None so
    the engine skips building a logit-bias processor.
    """
    if not bias:
        return None
    out: dict[int, float] = {}
    for key, value in bias.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            continue
    return out or None


class StreamOptions(BaseModel):
    """OpenAI `stream_options`: streaming-response knobs, currently just usage.

    Per the OpenAI spec, the trailing usage frame is emitted only when
    ``include_usage`` is true (default off omlx-side as well). Spec-correct
    clients that want totals must opt in; old omlx behavior emitted it always.
    """

    include_usage: bool | None = None


class ResponseFormat(BaseModel):
    """OpenAI `response_format` for JSON mode.

    omlx accepts ``{"type": "text"}`` (no-op) and ``{"type": "json_object"}``
    (constrains generation to a valid JSON value via :mod:`omlx._json`).
    ``{"type": "json_schema", ...}`` is rejected at the HTTP layer with
    ``400 unsupported`` (no schema-typed generation in this round).
    """

    type: str
    json_schema: dict[str, Any] | None = None


class _SamplingRequest(BaseModel):
    """Fields shared by the chat and text completion request bodies."""

    model: str
    # Default sized for agentic clients (opencode): a small cap truncates
    # replies and tool-call bodies mid-JSON, which reads to the client as a
    # failed tool call. A request-supplied `max_tokens` (or
    # `max_completion_tokens`) still overrides.
    max_tokens: int = 8192
    # OpenAI's reasoning-model output cap field. Honored when present and takes
    # precedence over `max_tokens`, since reasoning clients (and the OpenAI
    # spec for o-series / gpt-oss) send this rather than `max_tokens`. Without
    # it, reasoning effort spends the whole budget in the analysis channel and
    # `final` never fires — the model "thinks but never answers".
    max_completion_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    # OpenAI additive penalties (range [-2, 2]); 0.0 is a no-op.
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Not an OpenAI field; a multiplicative repeat penalty accepted as an extra.
    repetition_penalty: float | None = None
    # OpenAI logit bias: token-id string -> additive bias.
    logit_bias: dict[str, float] | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    # OpenAI stream options: when ``include_usage`` is set the trailing usage
    # frame is emitted; otherwise (spec default) only the terminal finish
    # chunk and ``[DONE]`` are sent.
    stream_options: StreamOptions | None = None

    def sampling(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            logit_bias=_int_keyed(self.logit_bias),
            stop=_normalize_stop(self.stop),
            seed=self.seed,
        )


class ChatRequest(_SamplingRequest):
    messages: list[ChatMessage]
    # OpenAI tool inputs: `tools` is offered to the chat template; `tool_choice`
    # is accepted for wire compatibility (advisory to the model).
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    # OpenAI JSON-mode: ``json_object`` constrains output to a valid JSON value;
    # ``json_schema`` is rejected by the HTTP layer with `400 unsupported`.
    response_format: ResponseFormat | None = None


class CompletionRequest(_SamplingRequest):
    prompt: str
    response_format: ResponseFormat | None = None


@dataclass
class SamplingParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass
class Completion:
    """One streamed generation step.

    `text` is the final-channel (answer) delta; `reasoning` is the analysis-
    channel (chain-of-thought) delta, empty for non-reasoning models.
    `finish_reason` and the token counts are populated only on the terminal
    step ("stop" for EOS, "length" when truncated at max_tokens).
    """

    text: str
    reasoning: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # OpenAI-shaped tool calls parsed from this step's output, if any.
    tool_calls: tuple[dict[str, Any], ...] = ()


# (content, finish[, reasoning]) -> choice dict. Chat builders take the optional
# trailing ``reasoning``; the text-completion builder ignores it.
ChoiceBuilder = Callable[..., dict[str, Any]]


def _now() -> int:
    return int(time.time())


def _rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chat_choice(
    content: str,
    finish: str | None,
    reasoning: str = "",
    tool_calls: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    # Terminal frame: empty delta. Otherwise include `content` / `reasoning_content`
    # / `tool_calls` only when non-empty.
    if finish is not None:
        return {"index": 0, "delta": {}, "finish_reason": finish}
    delta: dict[str, Any] = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    if tool_calls:
        delta["tool_calls"] = list(tool_calls)
    return {"index": 0, "delta": delta, "finish_reason": None}


def _chat_message_choice(
    content: str,
    finish: str | None,
    reasoning: str = "",
    tool_calls: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        # Non-streaming tool_calls carry no `index` (a streaming-delta concern).
        message["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in tool_calls]
    return {"index": 0, "message": message, "finish_reason": finish}


def _text_choice(
    content: str,
    finish: str | None,
    reasoning: str = "",
    tool_calls: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    # Text completions have no reasoning or tool fields; both are ignored.
    return {"index": 0, "text": content, "finish_reason": finish}


@dataclass(frozen=True, slots=True)
class Shape:
    """Per-endpoint envelope shape: id prefix, object tags, and choice builders.

    ``stream_choice`` emits incremental deltas/text; ``terminal_choice`` emits
    the final non-streaming choice (assistant message or text).
    ``stream_first_role`` requests a leading chunk with
    ``delta: {"role": "assistant", "content": ""}`` (OpenAI chat shape only).
    Text completions have no role delta.
    """

    rid_prefix: str
    stream_obj: str
    nonstream_obj: str
    stream_choice: ChoiceBuilder
    terminal_choice: ChoiceBuilder
    stream_first_role: bool = False


CHAT_SHAPE = Shape(
    rid_prefix="chatcmpl",
    stream_obj="chat.completion.chunk",
    nonstream_obj="chat.completion",
    stream_choice=_chat_choice,
    terminal_choice=_chat_message_choice,
    stream_first_role=True,
)
COMPLETION_SHAPE = Shape(
    rid_prefix="cmpl",
    stream_obj="text_completion",
    nonstream_obj="text_completion",
    stream_choice=_text_choice,
    terminal_choice=_text_choice,
    stream_first_role=False,
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
    include_usage: bool = False,
    first_chunk_role: bool = False,
) -> StreamingResponse:
    """Stream `chunks` as OpenAI SSE frames + a terminal choice, optional usage, [DONE].

    OpenAI streaming parity:
    - When ``first_chunk_role`` is set, a leading chunk carries
      ``delta: {"role": "assistant", "content": ""}`` (chat shape only).
    - The trailing usage frame (empty choices, ``usage`` populated) is emitted
      only when ``include_usage`` is set, per the OpenAI spec default. Older
      omlx behavior emitted it always — opt back in with
      ``stream_options.include_usage``.
    - A terminal finish chunk (``delta: {}`` + ``finish_reason``) is always
      emitted, then ``data: [DONE]\\n\\n``.

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
            if first_chunk_role:
                # OpenAI chat: first chunk carries the assistant role before any
                # content / reasoning / tool_calls flow.
                role_choice: dict[str, Any] = {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
                yield _sse(envelope([role_choice], None))
            final: Completion | None = None
            tc_index = 0  # running index across all streamed tool_calls
            for chunk in chunks:
                indexed: tuple[dict[str, Any], ...] = ()
                if chunk.tool_calls:
                    indexed = tuple(
                        {**tc, "index": tc_index + i} for i, tc in enumerate(chunk.tool_calls)
                    )
                    tc_index += len(indexed)
                if chunk.text or chunk.reasoning or indexed:
                    yield _sse(envelope([choice(chunk.text, None, chunk.reasoning, indexed)], None))
                if chunk.finish_reason is not None:
                    final = chunk
            finish = final.finish_reason if final else "stop"
            yield _sse(envelope([choice("", finish)], None))
            # Spec-shaped trailing usage frame only when the client opted in.
            if include_usage:
                yield _sse(envelope([], usage(final)))
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield _sse({"error": {"message": str(e), "type": type(e).__name__}})

    return StreamingResponse(gen(), media_type="text/event-stream")


def collect(
    chunks: Iterator[Completion],
) -> tuple[str, str, tuple[dict[str, Any], ...], Completion | None]:
    """Drain a token stream to (content, reasoning, tool_calls, terminal completion).

    Lets generator exceptions propagate so the caller (the HTTP layer) can
    map them to a 500; this keeps the protocol module free of HTTP concerns.
    """
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    final: Completion | None = None
    for chunk in chunks:
        content.append(chunk.text)
        reasoning.append(chunk.reasoning)
        tool_calls.extend(chunk.tool_calls)
        if chunk.finish_reason is not None:
            final = chunk
    return "".join(content), "".join(reasoning), tuple(tool_calls), final


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


# --- Ollama native API (`/api/chat`, `/api/generate`) -----------------------
#
# The Ollama wire form differs from OpenAI: NDJSON streaming (one JSON object
# per line, no `data:` prefix and no `[DONE]` sentinel), `stream` defaults to
# true, sampling lives under `options`, reasoning rides a `thinking` field, and
# tool-call `arguments` is a JSON object rather than an OpenAI JSON string.

_NDJSON_MEDIA_TYPE = "application/x-ndjson"


class OllamaOptions(BaseModel):
    """Ollama `options` block: sampling knobs under Ollama's field names.

    All optional; unset fields fall back to the `SamplingParams` defaults via
    `to_sampling`. `num_predict` is Ollama's `max_tokens`; `repeat_penalty` is
    the multiplicative `repetition_penalty`.
    """

    num_predict: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    def to_sampling(self, default_max_tokens: int) -> SamplingParams:
        """Map onto `SamplingParams`, using `default_max_tokens` when unset.

        A negative `num_predict` (Ollama's "infinite" / "fill context" sentinels)
        is treated as unset and falls back to `default_max_tokens`.
        """
        base = SamplingParams()
        max_tokens = default_max_tokens
        if self.num_predict is not None and self.num_predict >= 0:
            max_tokens = self.num_predict
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=base.temperature if self.temperature is None else self.temperature,
            top_p=base.top_p if self.top_p is None else self.top_p,
            top_k=base.top_k if self.top_k is None else self.top_k,
            min_p=base.min_p if self.min_p is None else self.min_p,
            frequency_penalty=(
                base.frequency_penalty if self.frequency_penalty is None else self.frequency_penalty
            ),
            presence_penalty=(
                base.presence_penalty if self.presence_penalty is None else self.presence_penalty
            ),
            repetition_penalty=self.repeat_penalty,
            stop=_normalize_stop(self.stop),
            seed=self.seed,
        )


class OllamaChatRequest(BaseModel):
    """Body for Ollama `/api/chat`.

    Reuses the OpenAI `ChatMessage` (content parts, null content, `tool_calls`,
    `role: "tool"` all templated the same way). `keep_alive` and `format` are
    accepted for wire compatibility but ignored. `stream` defaults to true, per
    Ollama.
    """

    model: str
    messages: list[ChatMessage] = []
    tools: list[dict[str, Any]] | None = None
    stream: bool = True
    think: bool | None = None
    options: OllamaOptions = OllamaOptions()
    keep_alive: str | int | None = None
    format: str | dict[str, Any] | None = None


class OllamaGenerateRequest(BaseModel):
    """Body for Ollama `/api/generate`.

    `prompt` is templated through the model's chat template by default (Ollama
    parity); `raw` streams the bare prompt with no template. `system` is folded
    in as a leading system message. `keep_alive`, `format`, `context`, `images`,
    `suffix`, and `template` are accepted but ignored.
    """

    model: str
    prompt: str = ""
    system: str | None = None
    raw: bool = False
    stream: bool = True
    think: bool | None = None
    options: OllamaOptions = OllamaOptions()
    keep_alive: str | int | None = None
    format: str | dict[str, Any] | None = None


class OllamaPullRequest(BaseModel):
    """Body for Ollama `/api/pull`; `name` is the legacy alias for `model`."""

    model: str = Field(validation_alias=AliasChoices("model", "name"))
    stream: bool = True
    insecure: bool = False


def _now_iso() -> str:
    """Current UTC instant as an RFC-3339 string for Ollama `created_at`.

    Ollama emits ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` (UTC, fractional seconds, ``Z``
    suffix); matching that shape byte-for-byte keeps strict clients happy.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ndjson(payload: dict[str, Any]) -> str:
    return f"{json.dumps(payload)}\n"


# OpenAI error envelope (`{"error": {"message","type","param","code"}}); the spec
# shape returned on every `/v1/*` and `/api/*` error. `server.py` registers FastAPI
# exception handlers that build this via `openai_error`, so non-stream failures no
# longer leak FastAPI's default `{"detail": ...}` shape to OpenAI clients.
class OpenAIError(Exception):
    """Raised anywhere in the stack to surface a structured OpenAI-shaped error.

    Carries the HTTP status, the OpenAI error `type` (e.g. ``invalid_request_error``,
    ``not_found_error``, ``internal_error``), a short snake_case ``code``
    (e.g. ``model_not_found``, ``unsupported``), the offending ``param`` if any,
    and the human-readable ``message``. ``server.py``'s exception handler turns
    this into the wire body; other exceptions fall through to a generic 500 with
    ``type=internal_error``.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int = 500,
        type: str = "internal_error",  # noqa: A002 - mirrors the OpenAI field name
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type
        self.code = code
        self.param = param


def openai_error_body(
    message: str,
    *,
    type: str = "internal_error",  # noqa: A002
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    """Build the OpenAI error object carried under the top-level ``error`` key."""
    err: dict[str, Any] = {"message": message, "type": type, "param": param}
    if code is not None:
        err["code"] = code
    return {"error": err}


def _ollama_stats(final: Completion | None, started_ns: int) -> dict[str, int]:
    """Terminal timing/count stats for an Ollama `done: true` frame.

    Token counts are exact (from the terminal `Completion`); durations are
    best-effort — only the wall-clock total is measured, so `load_duration` and
    `prompt_eval_duration` are 0 and `eval_duration` carries the whole span.
    All durations are nanoseconds, per Ollama.
    """
    total = time.perf_counter_ns() - started_ns
    prompt_tokens = final.prompt_tokens if final else 0
    completion_tokens = final.completion_tokens if final else 0
    return {
        "total_duration": total,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": completion_tokens,
        "eval_duration": total,
    }


def _ollama_tool_calls(tool_calls: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """OpenAI-shaped tool calls -> Ollama `{"function": {name, arguments}}`.

    Ollama carries `arguments` as an object (not the OpenAI JSON string), so the
    calls are decoded with `_decode_tool_call_args`; the streaming `index` and
    the OpenAI `id`/`type` fields are dropped.
    """
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = _decode_tool_call_args(tc).get("function", {})
        out.append(
            {"function": {"name": func.get("name", ""), "arguments": func.get("arguments", {})}}
        )
    return out


def _ollama_done_reason(finish: str | None) -> str:
    """Map an OpenAI finish reason to an Ollama `done_reason` (default "stop")."""
    return finish or "stop"


def ollama_chat_response(
    chunks: Iterator[Completion], *, model: str, think: bool
) -> StreamingResponse:
    """Stream `chunks` as Ollama `/api/chat` NDJSON frames.

    Each token frame carries an incremental `message` with `done: false`;
    `thinking` rides the message only when `think` is set and reasoning is
    non-empty, and `tool_calls` only when present. The terminal frame has an
    empty message, `done: true`, a `done_reason`, and timing/count stats.
    A generator error is surfaced in-band as a trailing `{"error": ...}` line,
    since a started 200 stream can't switch to an error status.
    """

    def gen() -> Iterator[str]:
        started = time.perf_counter_ns()
        try:
            final: Completion | None = None
            for chunk in chunks:
                message: dict[str, Any] = {"role": "assistant", "content": chunk.text}
                if think and chunk.reasoning:
                    message["thinking"] = chunk.reasoning
                if chunk.tool_calls:
                    message["tool_calls"] = _ollama_tool_calls(chunk.tool_calls)
                if chunk.text or (think and chunk.reasoning) or chunk.tool_calls:
                    yield _ndjson(
                        {
                            "model": model,
                            "created_at": _now_iso(),
                            "message": message,
                            "done": False,
                        }
                    )
                if chunk.finish_reason is not None:
                    final = chunk
            yield _ndjson(
                {
                    "model": model,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": _ollama_done_reason(final.finish_reason if final else None),
                    **_ollama_stats(final, started),
                }
            )
        except Exception as e:
            yield _ndjson({"error": str(e)})

    return StreamingResponse(gen(), media_type=_NDJSON_MEDIA_TYPE)


def ollama_chat_collect(chunks: Iterator[Completion], *, model: str, think: bool) -> dict[str, Any]:
    """Drain `chunks` into a single non-streaming Ollama `/api/chat` object."""
    started = time.perf_counter_ns()
    content, reasoning, tool_calls, final = collect(chunks)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if think and reasoning:
        message["thinking"] = reasoning
    if tool_calls:
        message["tool_calls"] = _ollama_tool_calls(tool_calls)
    return {
        "model": model,
        "created_at": _now_iso(),
        "message": message,
        "done": True,
        "done_reason": _ollama_done_reason(final.finish_reason if final else None),
        **_ollama_stats(final, started),
    }


def ollama_generate_response(
    chunks: Iterator[Completion], *, model: str, think: bool
) -> StreamingResponse:
    """Stream `chunks` as Ollama `/api/generate` NDJSON frames (flat `response`)."""

    def gen() -> Iterator[str]:
        started = time.perf_counter_ns()
        try:
            final: Completion | None = None
            for chunk in chunks:
                frame: dict[str, Any] = {
                    "model": model,
                    "created_at": _now_iso(),
                    "response": chunk.text,
                    "done": False,
                }
                if think and chunk.reasoning:
                    frame["thinking"] = chunk.reasoning
                if chunk.text or (think and chunk.reasoning):
                    yield _ndjson(frame)
                if chunk.finish_reason is not None:
                    final = chunk
            yield _ndjson(
                {
                    "model": model,
                    "created_at": _now_iso(),
                    "response": "",
                    "done": True,
                    "done_reason": _ollama_done_reason(final.finish_reason if final else None),
                    **_ollama_stats(final, started),
                }
            )
        except Exception as e:
            yield _ndjson({"error": str(e)})

    return StreamingResponse(gen(), media_type=_NDJSON_MEDIA_TYPE)


def ollama_generate_collect(
    chunks: Iterator[Completion], *, model: str, think: bool
) -> dict[str, Any]:
    """Drain `chunks` into a single non-streaming Ollama `/api/generate` object."""
    started = time.perf_counter_ns()
    content, reasoning, _tool_calls, final = collect(chunks)
    out: dict[str, Any] = {
        "model": model,
        "created_at": _now_iso(),
        "response": content,
        "done": True,
        "done_reason": _ollama_done_reason(final.finish_reason if final else None),
        **_ollama_stats(final, started),
    }
    if think and reasoning:
        out["thinking"] = reasoning
    return out
