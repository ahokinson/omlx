# AGENTS.md: guidance for AI agents working on omlx

## Quick commands

Run all of these from the repo root. They are the CI checks (.github/workflows/ci.yml):

```sh
uv run ruff check .       # lint
uv run ruff format --check .   # format check (run `ruff format .` to apply)
uv run ty check            # type check
uv run pytest -q           # tests (unit tests only; no model downloads, no Apple Silicon needed)
```

Install deps first with `uv sync --group dev`. `ruff`/`ty`/`pytest` are dev-group only.

## Load-bearing contract: do NOT break

`omlx` is a drop-in for Ollama/OpenAI clients. The wire contract below must stay
byte-for-byte stable; `tests/test_server.py` pins it. If you change any envelope
shape, update those tests intentionally, never by accident.

- **OpenAI `/v1` HTTP API** on `http://127.0.0.1:11434/v1` (Ollama's port):
  - `GET /health` -> `{"status": "ok", "loaded": "<mru model or null>",
    "loaded_models": ["<mru>", ..., "<lru>"]}`. The scalar `loaded` is the
    most-recently-used resident model (or null) and is byte-for-byte stable for
    drop-in Ollama/OpenAI clients; `loaded_models` is the multi-model list
    (MRU first) added for richer observability. Do not remove or rename either.
  - `GET /api/ps` -> Ollama-shaped `{"models": [{name, model, size, size_vram,
    digest, expires_at}]}` listing resident models (MRU first). `expires_at`
    is an RFC-3339 UTC instant (`YYYY-MM-DDTHH:MM:SS.ffffffZ`) of
    `last_used + keepalive_seconds`, or null when `keepalive_seconds < 0`
    ("keep forever"). Informational, not a hard expiry. All Ollama-shaped
    timestamps use the `…Z` suffix shape (Ollama parity); `protocol._now_iso`
    and `server._modified_at` are the only producers.
  - `GET /v1/models` -> OpenAI `{"object": "list", "data": [...]}`
  - `POST /v1/chat/completions` and `POST /v1/completions`, stream + non-stream.
    Streaming uses OpenAI SSE: `data: {chunk}\n\n` frames, a terminal
    `data: [DONE]\n\n`, and — only when the request sets
    `stream_options.include_usage: true` — a trailing usage frame with empty
    choices. Chat streams emit a leading chunk with
    `delta: {"role": "assistant", "content": ""}` (OpenAI parity); text
    completions stream bare `text` deltas and no role chunk. Errors mid-stream
    are emitted as `data: {"error": {"message": ..., "type": ...}}` (a 200
    stream can't switch to an error status late); non-stream generation
    failures return HTTP 500.
  - **Error body shape**: every `/v1/*` and `/api/*` error returns the OpenAI
    envelope `{"error": {"message", "type", "param", "code"}}` (built by
    `protocol.openai_error_body`; raised as `protocol.OpenAIError`). FastAPI's
    `{"detail": ...}` is replaced everywhere — pydantic validation failures
    return 400 `invalid_request_error/invalid_request`, unknown models 404
    `not_found_error/model_not_found`, internal failures 500 `internal_error`.
    Do not re-introduce `HTTPException(detail=...)` for these routes.
  - **Unknown model = 404, not auto-pull**: generation routes
    (`/v1/chat/completions`, `/v1/completions`, `/api/chat`, `/api/generate`,
    `/api/show`, `/api/delete`) call `server._require_known_model` as a
    preflight and raise `OpenAIError(404, model_not_found)` before any 200
    streaming response can start. Auto-pull is restricted to `omlx pull` and
    `/api/pull` — do not reintroduce silent fetching on a typo from a
    generation route. `engine.ModelManager._resolve` raises `OpenAIError` for
    unknown models; the preflight is the optimistic path, the engine re-raise
    is the race-defense path (coverage on both).
  - **Reasoning models (Harmony / gpt-oss)**: the engine parses Harmony output
    (`engine._HarmonyParser`), stripping control tokens and splitting the
    `analysis` channel from `final`. Reasoning rides `reasoning_content` on the
    chat delta (stream) and message (non-stream), present only when non-empty;
    absent for non-reasoning output. Inbound `reasoning_content` on request
    `messages` is dropped before `apply_chat_template`. `/v1/completions` has no
    reasoning field. `tests/test_server.py` pins both presence and absence.
  - **Tool / function calling**: `/v1/chat/completions` accepts OpenAI `tools`
    and `tool_choice`, plus the full OpenAI message shape — `content` as a
    string, a structured parts array, or null; `role: "tool"` results; and
    assistant `tool_calls`. `protocol.ChatMessage.to_template_dict` flattens
    these for `apply_chat_template` (parts joined, null -> `""`, tool-call
    `arguments` decoded to an object). Emitted calls ride `tool_calls` (on the
    delta for stream, with a running `index`; on the message for non-stream,
    without `index`) and flip `finish_reason` to `"tool_calls"`. Two parse
    paths: mainstream models via mlx-lm's per-model `tokenizer.tool_parser` +
    `tool_call_start`/`tool_call_end` (gated on `tokenizer.has_tool_calling`,
    `engine._parse_tool_calls`); gpt-oss/Harmony via the commentary
    `to=functions.NAME` channel (`engine._HarmonyParser`). Absent when no
    tools/calls. `tests/test_server.py` and `tests/test_engine.py` pin it.
  - **Sampling**: `_SamplingRequest` accepts `temperature`, `top_p`, `top_k`,
    `min_p`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, and
    `logit_bias` (OpenAI token-id-string keys, coerced to int in
    `protocol._int_keyed`). `engine._generation_kwargs` maps these onto
    `mlx-lm`'s `make_sampler` / `make_logits_processors`; the penalties/bias are
    no-ops at their defaults so the processor list stays None.
  - Object tags: `chat.completion` / `chat.completion.chunk` for chat;
    `text_completion` for completions.
- **Ollama native API** on the same port:
  - `POST /api/chat` and `POST /api/generate`, stream + non-stream. Streaming
    uses **NDJSON** (one JSON object per line, `{...}\n`, no `data:` prefix and
    no `[DONE]` sentinel); `stream` **defaults to `true`** (opposite of the
    OpenAI routes). Per-token frames carry `done: false`; the terminal frame has
    `done: true`, a `done_reason`, and timing/count stats (`prompt_eval_count` /
    `eval_count` exact; durations best-effort). Chat carries `message: {role,
    content, thinking?, tool_calls?}`; generate carries a flat `response` string.
    Mid-stream errors emit a trailing `{"error": ...}` line; non-stream errors
    return HTTP 500.
  - **Reasoning** rides `message.thinking` (chat) / top-level `thinking`
    (generate), emitted when the request sets `think`. **Tool-call `arguments`
    is a JSON object**, not the OpenAI JSON string, and the call is
    `{"function": {name, arguments}}` with no `id`/`type`/`index`.
  - **Sampling** lives under `options` (`num_predict` → `max_tokens`,
    `repeat_penalty` → `repetition_penalty`, plus `temperature`/`top_p`/`top_k`/
    `min_p`/`frequency_penalty`/`presence_penalty`/`stop`/`seed`);
    `OllamaOptions.to_sampling` maps them. `keep_alive` is accepted but ignored.
    `format` is honored for the strings `"json"` (enables JSON-object logits
    masking via `_json.json_object_processor`) and `"text"` (no-op); a dict
    (schema) or other string is `400 unsupported`.
  - **JSON mode**: `_SamplingRequest.response_format`
    (`{"type": "json_object|text|json_schema"}`) and `OllamaOptions`/the
    request-level `format` field map onto a single JSON-mask logits processor
    built by `engine._json_processor` from `omlx._json.json_object_processor`.
    `json_schema` (and Ollama schema dicts) are `400 unsupported`. `tools` and
    `response_format` are mutually exclusive — sending both is
    `400 response_format_and_tools_mutually_exclusive`. For Harmony models the
    mask is channel-aware (applies only inside `final`), re-derived over the body
    suffix each step so Harmony control tokens never reach the JSON DFA.
  - `POST /api/pull` downloads + registers a model (coarse NDJSON progress: a
    `pulling` frame then `success`; the HF download is blocking).
  - `GET /` returns the literal `Ollama is running` (Ollama liveness probe).
  - Wire shapes and NDJSON envelope builders live in `protocol.py`
    (`ollama_chat_response`/`_collect`, `ollama_generate_response`/`_collect`);
    `tests/test_server.py` pins them.
- **CLI surface**: `omlx pull|list|rm|run|serve|daemon start|stop|status`.
  Don't rename commands or flags.
- **Config**: `OMLX_*` env vars via `pydantic-settings`. `OMLX_KEEPALIVE` is a
  legacy alias for `keepalive_seconds`; keep it working. Multi-model residency
  is governed by `OMLX_MAX_LOADED_MODELS` (legacy alias `OMLX_MAX_LOADED`) and
  `OMLX_MAX_MEM_MB`, with `OMLX_MEM_BUDGET_FRACTION` (default 0.6). When both
  caps are unset (default), the budget = fraction × probed system memory and
  the count cap is derived from it (~1 model per 8 GiB); an explicitly
  requested model is loaded even if it alone exceeds the budget (Ollama parity).
  `OMLX_PROMPT_CACHE` (default on) toggles KV prompt-cache reuse;
  `OMLX_KV_BITS` (+ `OMLX_KV_GROUP_SIZE`, `OMLX_QUANTIZED_KV_START`) enable
  quantized KV caching.
- **Engine**: `ModelManager` keeps multiple models resident in an LRU
  `OrderedDict`; each `LoadedModel` has a per-model `active` counter (not a
  global one) so a generation on model A doesn't protect an idle model B.
  Resident `size_bytes` is backfilled after `mlx_lm.load` from
  `mx.get_active_memory()` (delta vs a baseline captured before the load),
  falling back to the registry entry's `size_bytes`, then to a 1 GiB minimum.
- **Prompt cache**: each `LoadedModel` holds a reusable KV cache guarded by
  `cache_lock`. `stream_chat` prefills only the prompt suffix that diverges from
  the cached prefix (`_prepare_cache`), then trims generated tokens back off so
  the resting invariant `len(cache_tokens) == cache offset` holds
  (`_finalize_cache`). A request that can't take the lock (a concurrent
  generation) runs with a throwaway cache; usage `prompt_tokens` adds the reused
  prefix length back so it reports the full prompt size.
- **Storage**: HF hub cache (`~/.cache/huggingface/hub`) for weights; index at
  `~/.omlx/models.json` mapping friendly name -> repo/path/metadata. The JSON
  shape (the `asdict(ModelEntry)`) is the on-disk format.

## Testing conventions

- `tests/conftest.py` redirects `settings.home` to a tmp dir, so tests never
  touch the real `~/.omlx` or HF cache. New tests should rely on this fixture.
- MLX is lazily imported in `engine.py` / `pull.py`; tests inject fake
  `mlx_lm` / `mlx.core` modules via `sys.modules` (see `test_engine.py`). Don't
  import mlx eagerly or those tests break.
- `ty` can't resolve attributes on those dynamic `ModuleType` fakes, so
  `tests/**` carries an override (`[[tool.ty.overrides]]` in pyproject.toml).

## Conventions

- `from __future__ import annotations` everywhere; PEP 604 unions.
- Ruff line-length 100, target py310. Lint rules: E, F, I, UP, B.
- Lazy-import mlx / huggingface_hub / uvicorn inside functions that need them
  (the CLI is usable without Apple Silicon for `list`/`rm`/`daemon status`).
- **Layering**: HTTP routing lives only in `server.py`; MLX is touched only in
  `engine.py` (and lazily in `pull.py`); `protocol.py` holds the wire shapes and
  envelope helpers and stays free of routing and MLX; `_harmony.py` holds the
  shared Harmony control-token set and channel detection (imported by
  `engine.py` for the streaming parser and by `_json.py` for the channel-aware
  JSON mask); `_json.py` holds the JSON-object logits processor (no routing,
  no MLX — it imports `mlx.core` lazily inside the processor). Don't reach
  across these.
- **Value objects**: `pydantic.BaseModel` for request bodies at the HTTP
  boundary — `@dataclass` for everything internal (`Completion`,
  `SamplingParams`, `LoadedModel`, `ModelEntry`, …). Don't push Pydantic inward.
- **Type hints on every signature** (`ty check` is a CI gate). Full annotations,
  not partial.
- **Docstrings** on every module and public function; module docstrings state
  the file's role. Comments and docs state non-obvious facts — contract
  constraints, edge cases, assumptions — not narration or justification of a
  change. Full-sentence prose.
- **Private surface**: module-private helpers and constants are `_`-prefixed;
  constants are `UPPER_SNAKE` with a short explaining comment.
- **Concurrency** (`ModelManager`): guard shared state with `self._lock`
  (`RLock`); methods that assume the lock is already held carry a `_locked`
  suffix. Do slow work (cold model load) *outside* the lock, then re-check state
  after reacquiring it.
- **Logging**: one named logger, `logging.getLogger("omlx")`. Tolerated failures
  (missing Metal, `clear_cache`) are swallowed with `logger.debug`, never raised.