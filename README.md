# omlx: Ollama, for MLX

Swap Ollama's engine for Apple's: `ollama` is `o` + `llama`; `omlx` is `o` +
`mlx`. A lightweight [Ollama](https://ollama.com) alternative for Apple Silicon,
backed by [`mlx-lm`](https://github.com/ml-explore/mlx-lm). Pull **any** Hugging Face
model (Xet-accelerated), serve it over an **OpenAI-compatible** API, keep it
warm in a background daemon. Runs quantized **safetensors** natively. **No
GGUF**.

## Install

Requires Apple Silicon (MLX). Install the CLI from git with
[uv](https://docs.astral.sh/uv/):

```sh
uv tool install git+https://github.com/ahokinson/omlx
```

This puts an `omlx` command on your PATH. To hack on it from a clone instead:

```sh
uv sync
```

## Use

```sh
# Pull a model (Xet-accelerated download into the HF cache)
uv run omlx pull mlx-community/Llama-3.2-1B-Instruct-4bit

# List / remove
uv run omlx list
uv run omlx rm Llama-3.2-1B-Instruct-4bit

# Interactive chat (auto-starts the daemon, keeps the model warm)
uv run omlx run mlx-community/Llama-3.2-1B-Instruct-4bit

# One-shot
uv run omlx run mlx-community/Llama-3.2-1B-Instruct-4bit --prompt "hi"

# Daemon lifecycle
uv run omlx daemon start|stop|status
uv run omlx serve            # foreground
```

## OpenAI API

The daemon listens on `http://127.0.0.1:11434/v1` (Ollama's port).

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="unused")
client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "hi"}],
)
```

Endpoints: `GET /v1/models`, `POST /v1/chat/completions` (stream + non-stream),
`POST /v1/completions`, `GET /health`.

## Reasoning models

Reasoning models that emit OpenAI **Harmony** format (e.g. `gpt-oss`) are
supported. The chain-of-thought (the `analysis` channel) is split from the
answer (the `final` channel) and the control tokens are stripped:

- `omlx run` prints the thinking **dimmed**, then the answer.
- `/v1/chat/completions` returns the thinking in `reasoning_content` (on the
  message for non-stream, on the delta for stream), alongside `content`. The
  field is present only when the model reasons.
- A prior turn's `reasoning_content` sent back in `messages` is dropped before
  templating.

## Tool calling

`/v1/chat/completions` supports OpenAI **function calling**, so agentic clients
like [opencode](https://github.com/sst/opencode) work as a drop-in. Pass `tools`;
when the model invokes one, the response carries `tool_calls` and
`finish_reason: "tool_calls"`.

```python
client.chat.completions.create(
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    messages=[{"role": "user", "content": "weather in SF?"}],
    tools=[{"type": "function", "function": {"name": "get_weather", ...}}],
)
```

- **Tool-capable models** (Qwen, Mistral, Llama, GLM, …) are parsed via
  `mlx-lm`'s per-model tool parsers.
- **gpt-oss / Harmony** tool calls (the `commentary` channel) are supported too.
- The full OpenAI message shape is accepted on input: `content` as a string or a
  structured parts array, `null` content, and `role: "tool"` results.

## How it works

- **Storage** reuses the HF hub cache (`~/.cache/huggingface/hub`); a small
  registry at `~/.omlx/models.json` maps friendly names → repos.
- **Pull** auto-detects MLX-ready repos; use `--convert` to quantize a
  non-MLX repo on-device via `mlx_lm.convert`.
- **Daemon** lazy-loads models and unloads them after an idle keep-alive TTL,
  freeing Metal memory.

## Development

```sh
uv sync --group dev
uv run pytest          # unit tests (no model downloads)
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check        # type check
```

Tests mock the MLX engine, so they run anywhere; the CLI/daemon still need
Apple Silicon at runtime. CI runs the same checks on macOS.
