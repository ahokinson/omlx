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
    is an ISO-8601 UTC instant of `last_used + keepalive_seconds`, or null when
    `keepalive_seconds < 0` ("keep forever"). Informational, not a hard expiry.
  - `GET /v1/models` -> OpenAI `{"object": "list", "data": [...]}`
  - `POST /v1/chat/completions` and `POST /v1/completions`, stream + non-stream.
    Streaming uses OpenAI SSE: `data: {chunk}\n\n` frames, a trailing usage
    frame, and a terminal `data: [DONE]\n\n`. Errors mid-stream are emitted as
    `data: {"error": {"message": ..., "type": ...}}` (a 200 stream can't switch
    to an error status late); non-stream errors return HTTP 500.
  - Object tags: `chat.completion` / `chat.completion.chunk` for chat;
    `text_completion` for completions.
- **CLI surface**: `omlx pull|list|rm|run|serve|daemon start|stop|status`.
  Don't rename commands or flags.
- **Config**: `OMLX_*` env vars via `pydantic-settings`. `OMLX_KEEPALIVE` is a
  legacy alias for `keepalive_seconds`; keep it working. Multi-model residency
  is governed by `OMLX_MAX_LOADED_MODELS` (legacy alias `OMLX_MAX_LOADED`) and
  `OMLX_MAX_MEM_MB`, with `OMLX_MEM_BUDGET_FRACTION` (default 0.6). When both
  caps are unset (default), the budget = fraction × probed system memory and
  the count cap is derived from it (~1 model per 8 GiB); an explicitly
  requested model is loaded even if it alone exceeds the budget (Ollama parity).
- **Engine**: `ModelManager` keeps multiple models resident in an LRU
  `OrderedDict`; each `LoadedModel` has a per-model `active` counter (not a
  global one) so a generation on model A doesn't protect an idle model B.
  Resident `size_bytes` is backfilled after `mlx_lm.load` from
  `mx.get_active_memory()` (delta vs a baseline captured before the load),
  falling back to the registry entry's `size_bytes`, then to a 1 GiB minimum.
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