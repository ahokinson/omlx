# omlx

> Ollama, for MLX — run any Hugging Face model locally on Apple Silicon.

[![CI](https://github.com/ahokinson/omlx/actions/workflows/ci.yml/badge.svg)](https://github.com/ahokinson/omlx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20–%203.13-blue.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)](#install)

A lightweight [Ollama](https://ollama.com) alternative, running on Apple's
[`mlx-lm`](https://github.com/ml-explore/mlx-lm) engine instead of llama.cpp.

- **Pull any Hugging Face model** — Xet-accelerated downloads, native
  quantized safetensors. No GGUF.
- **OpenAI-compatible `/v1` API** on Ollama's port (`11434`) — drop-in for
  existing clients.
- **Reasoning and tool calling** — Harmony / gpt-oss chain-of-thought and
  OpenAI function calling.
- **Background daemon** — keeps models warm, unloads on idle to free Metal
  memory.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [OpenAI API](#openai-api)
- [Reasoning models](#reasoning-models)
- [Tool calling](#tool-calling)
- [How it works](#how-it-works)
- [Development](#development)

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

## Quickstart

```sh
# Pull a model (Xet-accelerated download into the HF cache)
omlx pull mlx-community/Llama-3.2-1B-Instruct-4bit

# List / remove
omlx list
omlx rm Llama-3.2-1B-Instruct-4bit

# Chat (auto-starts the daemon, keeps the model warm). Type /bye to exit.
omlx run mlx-community/Llama-3.2-1B-Instruct-4bit

# One-shot
omlx run mlx-community/Llama-3.2-1B-Instruct-4bit --prompt "hi"

# Daemon lifecycle
omlx daemon start|stop|status
omlx serve            # foreground
```

## OpenAI API

The daemon listens on `http://127.0.0.1:11434/v1`, so any OpenAI client works:

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

Models that emit OpenAI **Harmony** format (e.g. `gpt-oss`) are supported. The
chain-of-thought (the `analysis` channel) is split from the answer (the `final`
channel) and control tokens are stripped:

- `omlx run` prints the thinking **dimmed**, then the answer.
- `/v1/chat/completions` returns the thinking in `reasoning_content` — on the
  message for non-stream, on the delta for stream — alongside `content`.
  Present only when the model reasons.
- A prior turn's `reasoning_content` sent back in `messages` is dropped before
  templating.

## Tool calling

`/v1/chat/completions` supports OpenAI **function calling**, so agentic clients
like [opencode](https://github.com/sst/opencode) work as a drop-in. Pass
`tools`; when the model invokes one, the response carries `tool_calls` and
`finish_reason: "tool_calls"`.

```python
client.chat.completions.create(
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    messages=[{"role": "user", "content": "weather in SF?"}],
    tools=[{"type": "function", "function": {"name": "get_weather", ...}}],
)
```

- **Tool-capable models** (Qwen, Mistral, Llama, GLM, etc.) parse via `mlx-lm`'s
  per-model tool parsers.
- **gpt-oss / Harmony** tool calls are also supported.
- The full OpenAI message shape is accepted on input: `content` as a string or a
  structured parts array, `null` content, and `role: "tool"` results.

## How it works

- **Storage** reuses the HF hub cache (`~/.cache/huggingface/hub`); a small
  registry at `~/.omlx/models.json` maps friendly names → repos.
- **Pull** auto-detects MLX-ready repos; use `--convert` to quantize a non-MLX
  repo on-device via `mlx_lm.convert`.
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
