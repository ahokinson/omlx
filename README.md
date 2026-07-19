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
- **OpenAI + Ollama APIs** on Ollama's port (`11434`) — OpenAI `/v1` and native
  Ollama `/api/*` (chat, generate, pull, tags), drop-in for existing clients.
- **Reasoning and tool calling** — Harmony / gpt-oss chain-of-thought and
  OpenAI function calling.
- **Background daemon** — keeps models warm, unloads on idle to free Metal
  memory.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [OpenAI API](#openai-api)
- [Ollama API](#ollama-api)
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

## Ollama API

The native Ollama endpoints are served on the same port, so the `ollama` client
works unchanged:

```python
from ollama import Client
client = Client(host="http://127.0.0.1:11434")
client.chat(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "hi"}],
)
```

Endpoints: `POST /api/chat`, `POST /api/generate`, `POST /api/pull`,
`GET /api/tags`, `POST /api/show`, `GET /api/ps`, `DELETE /api/delete`,
`GET /api/version`, and the `GET /` liveness probe. Generation streams **NDJSON**
(one JSON object per line) and `stream` defaults to `true`, per Ollama; sampling
rides the `options` block (`num_predict`, `temperature`, `repeat_penalty`, …).
Reasoning models return the chain-of-thought in `message.thinking` (chat) or the
top-level `thinking` field (generate) when the request sets `think`.

Not implemented: `/api/embed` (mlx-lm has no embedding path), `/api/copy`,
`/api/create`, `/api/push`. `keep_alive` and `format` are accepted but ignored,
and `/api/pull` reports coarse progress (the Hugging Face download is blocking).

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

### Using with opencode

Add omlx as an OpenAI-compatible provider in `opencode.json`. The model ids
under `models` must match the names from `omlx list`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "omlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "omlx (local MLX)",
      "options": { "baseURL": "http://127.0.0.1:11434/v1" },
      "models": {
        "gpt-oss:22b": {
          "name": "gpt-oss 20B",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 131072, "output": 32768 }
        },
        "glm-4.7-flash:31b": {
          "name": "GLM-4.7 Flash",
          "tool_call": true,
          "limit": { "context": 131072, "output": 32768 }
        }
      }
    }
  }
}
```

`limit.output` sets opencode's per-reply token budget; without it, long replies
and tool-call bodies can truncate. Start the daemon (`omlx daemon start` or
`omlx serve`), then select the model in opencode with `/models`.

## How it works

- **Storage** reuses the HF hub cache (`~/.cache/huggingface/hub`); a small
  registry at `~/.omlx/models.json` maps friendly names → repos.
- **Pull** auto-detects MLX-ready repos; use `--convert` to quantize a non-MLX
  repo on-device via `mlx_lm.convert`.
- **Daemon** lazy-loads models and unloads them after an idle keep-alive TTL,
  freeing Metal memory.
- **Prompt cache** keeps a per-model KV cache warm and prefills only the part of
  each prompt that changed from the last turn — the big win for agentic clients
  that resend a large stable prefix (system prompt + tools) every request. Set
  `OMLX_PROMPT_CACHE=0` to disable, or `OMLX_KV_BITS=8` to quantize the KV cache
  for longer contexts.
- **Sampling** honors `top_p`, `top_k`, `min_p`, `frequency_penalty`,
  `presence_penalty`, `repetition_penalty`, and `logit_bias` alongside
  `temperature`.

## Development

```sh
uv sync --group dev
uv run pytest          # unit tests (no model downloads)
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check        # type check
```
