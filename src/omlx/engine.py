"""Model lifecycle: load, cache, stream, and idle-unload MLX models."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from . import config, registry
from ._harmony import HARMONY_CONTROL, REASONING_CHANNELS, harmony_tool_name, is_harmony
from .config import settings
from .protocol import Completion, OpenAIError, SamplingParams

logger = logging.getLogger("omlx")

REAP_INTERVAL_SECONDS = 15
_FALLBACK_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GiB when size can't be measured


@dataclass
class LoadedModel:
    name: str
    model: Any
    tokenizer: Any
    last_used: float
    size_bytes: int = 0
    active: int = 0  # in-flight generations on this model; protects against eviction
    # Reusable KV prompt cache and the token ids it holds. Invariant: at rest
    # `len(cache_tokens) == cache[0].offset` — the cache holds exactly the last
    # prompt (generated tokens are trimmed off after each turn), so the next
    # request need only prefill the diverging suffix. Guarded by `cache_lock`;
    # a request that can't take the lock (a concurrent generation holds it) runs
    # with its own throwaway cache and leaves this one untouched.
    cache: Any = None
    cache_tokens: list[int] = field(default_factory=list)
    cache_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class LoadedModelInfo:
    """Public snapshot of a resident model (for `/health` and `/api/ps`)."""

    name: str
    size_bytes: int
    last_used: float
    expires_at: float | None


def _metal_active_memory() -> int | None:
    """Resident Metal memory in bytes, or None if unavailable.

    Lazily imported; absence (non-Apple-Silicon, stubbed tests) is tolerated.
    """
    try:
        import mlx.core as mx  # ty: ignore[unresolved-import]

        return int(mx.get_active_memory())
    except Exception as e:
        logger.debug("get_active_memory failed: %s", e)
        return None


class ModelManager:
    """Holds up to N warm models within a memory budget; evicts least-recently-used.

    Mirrors Ollama's `OLLAMA_MAX_LOADED_MODELS`: multiple models can stay
    resident concurrently, and an idle TTL reaps each one after `keepalive_seconds`
    of inactivity. The memory budget (default ~60% of system memory) bounds the
    resident set; LRU entries with no in-flight generations are evicted to fit.
    """

    def __init__(
        self,
        keepalive_seconds: int | None = None,
        max_loaded: int | None = None,
        mem_budget_mb: int | None = None,
        start_reaper: bool = True,
    ):
        self.keepalive_seconds = (
            keepalive_seconds if keepalive_seconds is not None else settings.keepalive_seconds
        )
        budget_mb = mem_budget_mb if mem_budget_mb is not None else settings.mem_budget_mb()
        self._mem_budget_bytes = budget_mb * 1024 * 1024
        # Count cap: explicit arg wins, else an explicit OMLX_MAX_LOADED_MODELS,
        # else derive from *this manager's* effective budget (so a custom
        # mem_budget_mb stays self-consistent instead of pulling the cap from
        # global system-memory settings).
        if max_loaded is not None:
            self._max_loaded = max_loaded
        elif settings.max_loaded_models is not None:
            self._max_loaded = max(1, settings.max_loaded_models)
        else:
            self._max_loaded = config._count_cap_for_budget_mb(budget_mb)
        # Insertion-ordered dict: least-recently-used is first, most-recently-used
        # is last. `get()` moves an entry to the end on every touch.
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.RLock()
        if start_reaper:
            threading.Thread(target=self._reap_loop, daemon=True).start()

    def _resolve(self, name: str) -> str:
        """Map a friendly name / repo id to something mlx_lm.load accepts.

        Known models load from their recorded path/repo; unknown names raise
        ``OpenAIError(404, model_not_found)``. Auto-pull is intentionally
        restricted to ``omlx pull`` and ``/api/pull`` so a typo on a generation
        route doesn't silently fetch a multi-GB repo.
        """
        entry = registry.get(name)
        if entry is None:
            raise OpenAIError(
                f"model {name!r} not found",
                status=404,
                type="not_found_error",
                code="model_not_found",
                param="model",
            )
        return entry.repo_id

    def _resident_bytes_locked(self) -> int:
        return sum(lm.size_bytes for lm in self._loaded.values())

    def _evict_to_fit_locked(self, incoming_bytes: int) -> None:
        """Evict idle LRU entries until `incoming_bytes` fits the memory budget.

        Entries with an in-flight generation (`active > 0`) are never evicted.
        If even evicting everything won't fit, we let the newcomer through:
        Ollama loads an explicitly-requested model even if it exceeds budget.
        """
        budget = self._mem_budget_bytes
        while self._resident_bytes_locked() + incoming_bytes > budget and self._loaded:
            for name, lm in list(self._loaded.items()):  # LRU order
                if lm.active > 0:
                    continue
                logger.info("evicting %r to fit budget", name)
                self._unload_locked(name)
                break
            else:
                break  # everything resident is active; can't evict further

    def _evict_to_count_locked(self) -> None:
        """Evict idle LRU entries until the count cap is respected, if reachable."""
        while len(self._loaded) >= self._max_loaded:
            for name, lm in list(self._loaded.items()):
                if lm.active > 0:
                    continue
                logger.info("evicting %r to respect max_loaded", name)
                self._unload_locked(name)
                break
            else:
                break

    def get(self, name: str) -> LoadedModel:
        """Return the warm model for `name`, loading (evicting LRU others) if needed."""
        with self._lock:
            lm = self._loaded.get(name)
            if lm is not None:
                self._loaded.move_to_end(name)
                lm.last_used = time.time()
                return lm

        source = self._resolve(name)
        entry = registry.get(name)

        # Evict to fit the incoming model *before* loading it. Loading first
        # would briefly hold every resident model plus the newcomer in GPU
        # memory at once; on Metal that overflow is an uncatchable out-of-memory
        # abort that takes the whole process down. The registry's recorded
        # on-disk size is the pre-load size estimate.
        with self._lock:
            existing = self._loaded.get(name)
            if existing is not None:
                self._loaded.move_to_end(name)
                existing.last_used = time.time()
                return existing
            self._evict_to_fit_locked(_estimate_size_bytes(None, None, entry))
            self._evict_to_count_locked()

        # Load outside the lock: a cold pull/load can take minutes, and holding
        # the lock would block /health the whole time.
        baseline = _metal_active_memory()
        from mlx_lm import load

        # load() returns (model, tokenizer), plus a config when return_config is
        # set; star-unpack tolerates either arity.
        model, tokenizer, *_ = load(source)
        post = _metal_active_memory()
        size_bytes = _estimate_size_bytes(baseline, post, entry)

        with self._lock:
            # Re-check after reacquiring the lock: another caller may have
            # loaded the same model while we waited on the cold load above.
            # If so, drop ours and reuse theirs to avoid a duplicate resident.
            existing = self._loaded.get(name)
            if existing is not None:
                self._loaded.move_to_end(name)
                existing.last_used = time.time()
                return existing
            # Refine against the measured size in case it exceeded the estimate.
            self._evict_to_fit_locked(size_bytes)
            self._evict_to_count_locked()
            lm = LoadedModel(name, model, tokenizer, time.time(), size_bytes=size_bytes)
            self._loaded[name] = lm
            return lm

    def _unload_locked(self, name: str | None = None) -> None:
        """Unload one model by name, or all when `name` is None."""
        if name is None:
            if not self._loaded:
                return
            self._loaded.clear()
        else:
            if self._loaded.pop(name, None) is None:
                return
        try:
            import mlx.core as mx  # ty: ignore[unresolved-import]

            mx.clear_cache()
        except Exception as e:
            logger.debug("clear_cache failed on unload: %s", e)

    def loaded(self) -> str | None:
        """Name of the most-recently-used resident model, or None if none."""
        with self._lock:
            for lm in reversed(self._loaded.values()):
                return lm.name
            return None

    def loaded_models(self) -> list[LoadedModelInfo]:
        """Snapshots of resident models, most-recently-used first (for /health)."""
        with self._lock:
            out: list[LoadedModelInfo] = []
            for name, lm in reversed(self._loaded.items()):
                expires = (
                    lm.last_used + self.keepalive_seconds if self.keepalive_seconds >= 0 else None
                )
                out.append(
                    LoadedModelInfo(
                        name=name,
                        size_bytes=lm.size_bytes,
                        last_used=lm.last_used,
                        expires_at=expires,
                    )
                )
            return out

    def ps(self) -> list[dict[str, Any]]:
        """Ollama-shaped `/api/ps` listing of resident models."""
        out: list[dict[str, Any]] = []
        for info in self.loaded_models():
            entry: dict[str, Any] = {
                "name": info.name,
                "model": info.name,
                "size": info.size_bytes,
                "size_vram": info.size_bytes,
                "digest": "",
            }
            if info.expires_at is None:
                entry["expires_at"] = None
            else:
                entry["expires_at"] = datetime.fromtimestamp(
                    info.expires_at, timezone.utc
                ).isoformat()
            out.append(entry)
        return out

    def stream_chat(
        self,
        name: str,
        messages: list[dict[str, Any]],
        params: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Iterator[Completion]:
        """Stream generated tokens for `messages` via the model's chat template.

        Output passes through the Harmony parser: the analysis channel is split
        from the final answer and control tokens are stripped. Inert for
        non-reasoning models. When `tools` is given and the tokenizer supports
        tool calling, tools are offered to the template and tool-call spans in
        the output are parsed into OpenAI `tool_calls`. When `response_format`
        carries ``{"type": "json_object"}`` (already validated by the HTTP
        layer) a JSON-mask logits processor is added to constrain output to a
        valid JSON value (Harmony-channel-aware; see :mod:`omlx._json`).
        """
        lm = self.get(name)
        tok = lm.tokenizer
        # gpt-oss reports no `has_tool_calling` (Harmony has no mlx-lm tool parser);
        # its calls are parsed from the commentary channel in `_parse_harmony`.
        generic_tools = bool(tools) and getattr(tok, "has_tool_calling", False)
        offer_tools = bool(tools) and (generic_tools or is_harmony(tok))
        template_kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        if offer_tools:
            template_kwargs["tools"] = tools
        prompt = tok.apply_chat_template(messages, **template_kwargs)
        json_mode = bool(response_format and response_format.get("type") == "json_object")
        stream = _parse_harmony(
            self._stream_prompt(
                lm,
                prompt,
                params or SamplingParams(),
                use_cache=True,
                json_provider=_json_processor(tok) if json_mode else None,
            )
        )
        if generic_tools:
            stream = _parse_tool_calls(
                stream,
                getattr(tok, "tool_call_start", None),
                getattr(tok, "tool_call_end", None),
                getattr(tok, "tool_parser", None),
                tools,
            )
        yield from stream

    def stream_text(
        self,
        name: str,
        prompt: str,
        params: SamplingParams | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Iterator[Completion]:
        """Stream generated tokens for a raw `prompt` (no chat template).

        ``response_format`` mirrors :meth:`stream_chat`: a dict with
        ``"type": "json_object"`` activates the JSON-mask logits processor.
        """
        lm = self.get(name)
        json_mode = bool(response_format and response_format.get("type") == "json_object")
        yield from self._stream_prompt(
            lm,
            prompt,
            params or SamplingParams(),
            json_provider=_json_processor(lm.tokenizer) if json_mode else None,
        )

    def _stream_prompt(
        self,
        lm: LoadedModel,
        prompt: str,
        params: SamplingParams,
        use_cache: bool = False,
        json_provider: Callable[[Any, Any], Any] | None = None,
    ) -> Iterator[Completion]:
        """Stream a prompt, optionally reusing/extending the model's KV prompt cache.

        With `use_cache`, only the suffix of `prompt` that diverges from the
        cached prefix is prefilled; `mlx-lm` reports `prompt_tokens` for that
        suffix only, so the reused-prefix length is added back to keep the usage
        count the full prompt size. ``json_provider``, when supplied, is
        prepended to the logits-processor chain (before the penalty processors)
        to constrain output to valid JSON.
        """
        from mlx_lm import stream_generate

        if params.seed is not None:
            import mlx.core as mx  # ty: ignore[unresolved-import]

            mx.random.seed(params.seed)
        kwargs = _generation_kwargs(params, json_provider=json_provider)
        prompt_arg, offset, cache, tokens = self._prepare_cache(lm, prompt, use_cache)
        if cache is not None:
            kwargs["prompt_cache"] = cache
        gen = stream_generate(
            lm.model,
            lm.tokenizer,
            prompt=prompt_arg,
            max_tokens=params.max_tokens,
            **kwargs,
        )
        try:
            with self._active_generation(lm):
                if not params.stop:
                    for resp in gen:
                        yield _completion(resp, offset)
                else:
                    for comp in _stream_with_stops(gen, params.stop):
                        yield _with_offset(comp, offset)
        finally:
            if tokens is not None:
                self._finalize_cache(lm, cache, tokens)
                lm.cache_lock.release()

    def _prepare_cache(
        self, lm: LoadedModel, prompt: str, use_cache: bool
    ) -> tuple[Any, int, Any, list[int] | None]:
        """Resolve the prefill input, reusing the model's KV cache when possible.

        Returns `(prompt_arg, offset, cache, tokens)`. When caching is off or the
        cache is busy, `prompt_arg` is the original string, `cache`/`tokens` are
        None, and `offset` is 0 — the plain, non-cached path. When caching is
        active, the cache lock is held (released by `_stream_prompt` once the
        stream ends), `prompt_arg` is the diverging suffix (an `mx.array`),
        `offset` is the reused-prefix length, and `tokens` is the full prompt.
        """
        if not (use_cache and settings.prompt_cache):
            return prompt, 0, None, None
        if not lm.cache_lock.acquire(blocking=False):
            return prompt, 0, None, None
        try:
            import mlx.core as mx  # ty: ignore[unresolved-import]
            from mlx_lm.models.cache import (
                can_trim_prompt_cache,
                make_prompt_cache,
                trim_prompt_cache,
            )

            tokens = _encode_prompt(lm.tokenizer, prompt)
            if len(tokens) < 2:
                lm.cache_lock.release()
                return prompt, 0, None, None
            cache = lm.cache
            if cache is None or not can_trim_prompt_cache(cache):
                cache = make_prompt_cache(lm.model)
                lm.cache, lm.cache_tokens = cache, []
            # Reuse the longest common prefix, but keep >=1 token to prefill.
            common = min(_lcp(lm.cache_tokens, tokens), len(tokens) - 1)
            trim_prompt_cache(cache, len(lm.cache_tokens) - common)
            return mx.array(tokens[common:]), common, cache, tokens
        except Exception as e:
            logger.debug("prompt-cache setup failed (%s); prefilling full prompt", e)
            lm.cache_lock.release()
            return prompt, 0, None, None

    def _finalize_cache(self, lm: LoadedModel, cache: Any, tokens: list[int]) -> None:
        """Trim generated tokens so the cache again holds exactly `tokens`.

        Restores the resting invariant (`len(cache_tokens) == cache offset`) so
        the next turn's prefix match is correct. A cache that can't be trimmed
        (e.g. quantized past the KV-quant threshold) is dropped rather than left
        inconsistent.
        """
        try:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

            if not can_trim_prompt_cache(cache):
                lm.cache, lm.cache_tokens = None, []
                return
            extra = cache[0].offset - len(tokens)
            if extra > 0:
                trim_prompt_cache(cache, extra)
            lm.cache, lm.cache_tokens = cache, tokens
        except Exception as e:
            logger.debug("prompt-cache finalize failed (%s); dropping cache", e)
            lm.cache, lm.cache_tokens = None, []

    @contextmanager
    def _active_generation(self, lm: LoadedModel) -> Iterator[None]:
        """Count an in-flight generation so the reaper won't evict this model mid-stream.

        Per-model: bumping A's counter protects only A; an idle B can still be
        reaped. Refreshes `last_used` on exit so the idle TTL restarts after a
        finished generation rather than counting its duration against us.
        """
        with self._lock:
            lm.active += 1
        try:
            yield
        finally:
            with self._lock:
                lm.active -= 1
                lm.last_used = time.time()

    def _reap_loop(self) -> None:
        while True:
            time.sleep(REAP_INTERVAL_SECONDS)
            now = time.time()
            with self._lock:
                for name, lm in list(self._loaded.items()):
                    # keepalive < 0 means "keep forever" (Ollama parity); mirror
                    # loaded_models()/ps() so a negative TTL never reaps.
                    idle = (
                        lm.active == 0
                        and self.keepalive_seconds >= 0
                        and (now - lm.last_used) > self.keepalive_seconds
                    )
                    if idle:
                        logger.info("unloading idle model %r", name)
                        self._unload_locked(name)


MANAGER = ModelManager()


def _encode_prompt(tok: Any, prompt: str) -> list[int]:
    """Tokenize `prompt` the way `stream_generate` would for a string input.

    Mirrors mlx-lm's BOS heuristic so a suffix fed as token ids reconstructs the
    same sequence the string path would have produced.
    """
    bos = getattr(tok, "bos_token", None)
    add_special = bos is None or not prompt.startswith(bos)
    return list(tok.encode(prompt, add_special_tokens=add_special))


def _lcp(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _completion(resp: Any, offset: int) -> Completion:
    """A `stream_generate` step -> `Completion`, adding the reused-prefix length.

    The prefix length is added only on the terminal step, where usage is read;
    intermediate `prompt_tokens` are ignored downstream.
    """
    prompt_tokens = resp.prompt_tokens + (offset if resp.finish_reason is not None else 0)
    return Completion(
        text=resp.text,
        finish_reason=resp.finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=resp.generation_tokens,
    )


def _with_offset(comp: Completion, offset: int) -> Completion:
    """Add the reused-prefix length to a terminal completion's prompt token count."""
    if offset and comp.finish_reason is not None:
        return replace(comp, prompt_tokens=comp.prompt_tokens + offset)
    return comp


def _generation_kwargs(
    params: SamplingParams,
    *,
    json_provider: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Sampler, logits processors, and KV-cache options for `stream_generate`.

    Non-default sampling fields (`top_k`, `min_p`, the penalties, `logit_bias`)
    are honored here; `make_logits_processors` no-ops on zero/None so the
    processor list is empty unless a penalty or bias was requested. When a
    ``json_provider`` is supplied (from a JSON-mode request), it's prepended
    to the processor chain so structural masking runs before estimator
    penalties. KV-cache quantization is server-wide (`OMLX_KV_BITS`),
    applied only when set.
    """
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    sampler = make_sampler(
        temp=params.temperature,
        top_p=params.top_p,
        min_p=params.min_p,
        top_k=params.top_k,
    )
    processors = make_logits_processors(
        logit_bias=params.logit_bias,
        repetition_penalty=params.repetition_penalty,
        presence_penalty=params.presence_penalty,
        frequency_penalty=params.frequency_penalty,
    )
    if json_provider is not None:
        # JSON mask first: structural constraint must dominate over estimator
        # penalties that might otherwise nudge the model off-grammar.
        processors = [json_provider] + list(processors or [])
    kwargs: dict[str, Any] = {"sampler": sampler, "logits_processors": processors or None}
    if settings.kv_bits is not None:
        kwargs["kv_bits"] = settings.kv_bits
        kwargs["kv_group_size"] = settings.kv_group_size
        kwargs["quantized_kv_start"] = settings.quantized_kv_start
    return kwargs


def _json_processor(tokenizer: Any) -> Callable[[Any, Any], Any]:
    """Build a JSON-mode logits processor for a tokenizer.

    Detects whether the tokenizer speaks Harmony (gpt-oss) via the shared vocab
    probe; if so, the processor is channel-aware (applies only inside
    ``final``); otherwise it constrains output from the very first token.
    """
    from ._json import json_object_processor

    return json_object_processor(tokenizer, harmony=is_harmony(tokenizer))


def _earliest_stop(text: str, stops: tuple[str, ...]) -> int | None:
    """Lowest start index at which any stop sequence occurs in `text`, or None."""
    best: int | None = None
    for s in stops:
        i = text.find(s)
        if i != -1 and (best is None or i < best):
            best = i
    return best


def _stream_with_stops(gen: Iterator[Any], stops: tuple[str, ...]) -> Iterator[Completion]:
    """Re-emit `gen`'s steps, stopping at the first `stops` match.

    Stop strings can span token boundaries, so text is buffered and the last
    ``max_len - 1`` chars are withheld before emitting — the longest suffix that
    could still be the start of a match. A match truncates the output at its
    start with finish_reason "stop"; otherwise the stream's own terminal step
    (and its token counts) passes through unchanged.
    """
    max_len = max(len(s) for s in stops)
    acc = ""
    emitted = 0
    last: Any = None
    for resp in gen:
        acc += resp.text
        last = resp
        cut = _earliest_stop(acc, stops)
        if cut is not None:
            yield Completion(
                text=acc[emitted:cut],
                finish_reason="stop",
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.generation_tokens,
            )
            return
        safe = len(acc) - (max_len - 1)
        if safe > emitted:
            yield Completion(text=acc[emitted:safe])
            emitted = safe
    if last is not None:
        yield Completion(
            text=acc[emitted:],
            finish_reason=last.finish_reason,
            prompt_tokens=last.prompt_tokens,
            completion_tokens=last.generation_tokens,
        )


# Harmony control tokens / channel detection live in `_harmony.py` (shared
# with `_json.py` for Harmony-aware JSON-mode masking). The parser below
# references them via the imports at the top of this module.


class _HarmonyParser:
    """Streaming splitter for OpenAI Harmony output (gpt-oss reasoning models).

    Harmony wraps each message in control tokens, e.g.::

        <|channel|>analysis<|message|>THINK<|end|>
        <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>

    Strips the control tokens and routes the ``analysis``/``commentary``
    channels to reasoning and the ``final`` channel to content. A commentary
    header with a ``to=functions.NAME`` recipient is a tool call: its body is
    captured and emitted as an OpenAI ``tool_calls`` entry at the closing
    ``<|call|>`` rather than as content or reasoning.

    Inert for non-Harmony models: with no control tokens the stream stays in the
    initial content state and ``reasoning`` stays empty. Assumes Harmony output
    opens with a control token (gpt-oss does); bare text before the first
    ``<|channel|>`` / ``<|start|>`` is treated as content.

    Feed deltas with :meth:`push`; call :meth:`flush` once the stream ends to
    release any tail withheld while disambiguating a split control token.
    """

    _TEXT, _CHANNEL, _ROLE = range(3)

    def __init__(self) -> None:
        self._buf = ""
        self._state = self._TEXT
        self._channel = ""
        self._to_reasoning = False
        # Tool-call capture: name from the channel header, body accumulated until
        # the closing token; finished calls queue in `_pending` for the next scan.
        self._tool_active = False
        self._tool_name = ""
        self._tool_args: list[str] = []
        self._pending: list[dict[str, Any]] = []

    def push(self, text: str) -> tuple[str, str, list[dict[str, Any]]]:
        """Consume `text`; return (content_delta, reasoning_delta, tool_calls)."""
        self._buf += text
        return self._scan(final=False)

    def flush(self) -> tuple[str, str, list[dict[str, Any]]]:
        """Release any withheld tail at end of stream."""
        return self._scan(final=True)

    def _emit(self, text: str, content: list[str], reasoning: list[str]) -> None:
        if self._state == self._CHANNEL:
            self._channel += text  # collecting the channel name
        elif self._state == self._ROLE:
            pass  # role name / suppressed header text
        elif self._tool_active:
            self._tool_args.append(text)  # tool-call arguments body
        elif self._to_reasoning:
            reasoning.append(text)
        else:
            content.append(text)

    def _close_tool(self) -> None:
        self._pending.append(
            _format_tool_call({"name": self._tool_name, "arguments": "".join(self._tool_args)})
        )
        self._tool_active, self._tool_name, self._tool_args = False, "", []

    def _transition(self, tok: str) -> None:
        if tok == "<|channel|>":
            self._state, self._channel = self._CHANNEL, ""
        elif tok == "<|message|>":
            self._state = self._TEXT
            name = harmony_tool_name(self._channel)
            if name is not None:
                self._tool_active, self._tool_name, self._tool_args = True, name, []
                self._to_reasoning = False
            else:
                self._to_reasoning = self._channel.strip() in REASONING_CHANNELS
        elif tok in ("<|start|>", "<|constrain|>"):
            self._state = self._ROLE  # suppress the role / constraint header
        else:  # <|end|>, <|return|>, <|call|> — close the body, await next header
            if self._tool_active:
                self._close_tool()
            self._state, self._channel = self._ROLE, ""

    def _match(self, i: int) -> str | None:
        for tok in HARMONY_CONTROL:
            if self._buf.startswith(tok, i):
                return tok
        return None

    def _is_partial(self, i: int) -> bool:
        frag = self._buf[i:]
        return any(tok.startswith(frag) for tok in HARMONY_CONTROL)

    def _scan(self, final: bool) -> tuple[str, str, list[dict[str, Any]]]:
        content: list[str] = []
        reasoning: list[str] = []
        buf = self._buf
        i, n = 0, len(buf)
        while i < n:
            j = buf.find("<", i)
            if j == -1:
                self._emit(buf[i:], content, reasoning)
                i = n
                break
            if j > i:
                self._emit(buf[i:j], content, reasoning)
                i = j
            tok = self._match(i)
            if tok is not None:
                self._transition(tok)
                i += len(tok)
                continue
            if not final and self._is_partial(i):
                break  # a control token may be split across chunks; withhold it
            self._emit(buf[i], content, reasoning)  # a literal '<'
            i += 1
        self._buf = buf[i:]
        calls, self._pending = self._pending, []
        return "".join(content), "".join(reasoning), calls


def _parse_harmony(chunks: Iterator[Completion]) -> Iterator[Completion]:
    """Re-emit `chunks` with Harmony channels split into text, reasoning, tool calls.

    Control tokens are stripped; the terminal chunk's `finish_reason` and token
    counts are preserved. A boundary chunk may carry both a reasoning tail and a
    content start — both ride the same emitted `Completion`. A commentary tool
    call becomes a `tool_calls` entry and flips the terminal `finish_reason` to
    ``"tool_calls"`` (OpenAI contract).
    """
    parser = _HarmonyParser()
    saw_tool = False
    for chunk in chunks:
        content, reasoning, calls = parser.push(chunk.text)
        terminal = chunk.finish_reason is not None
        if terminal:
            tail_content, tail_reasoning, tail_calls = parser.flush()
            content += tail_content
            reasoning += tail_reasoning
            calls += tail_calls
        if calls:
            saw_tool = True
        finish = chunk.finish_reason
        if terminal and saw_tool and finish == "stop":
            finish = "tool_calls"
        if content or reasoning or calls or terminal:
            yield Completion(
                text=content,
                reasoning=reasoning,
                finish_reason=finish,
                prompt_tokens=chunk.prompt_tokens,
                completion_tokens=chunk.completion_tokens,
                tool_calls=tuple(calls),
            )


def _tool_partial_suffix(buf: str, delim: str) -> int:
    """Length of the longest tail of `buf` that is a proper prefix of `delim`.

    A delimiter can be split across generation steps; this is how many trailing
    chars to withhold until the next step disambiguates them.
    """
    for k in range(min(len(buf), len(delim) - 1), 0, -1):
        if buf.endswith(delim[:k]):
            return k
    return 0


def _format_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """One parser result -> OpenAI `tool_calls` entry (arguments as a JSON string).

    An empty arguments body (a no-argument call) is emitted as `"{}"`, not `""`:
    the wire contract is a JSON string, and clients (e.g. opencode) `JSON.parse`
    it, which rejects the empty string.
    """
    args = tc.get("arguments", {})
    if isinstance(args, str):
        arguments = args if args.strip() else "{}"
    else:
        arguments = json.dumps(args, ensure_ascii=False)
    return {
        "id": tc.get("id") or f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": tc.get("name", ""),
            "arguments": arguments,
        },
    }


class _ToolCallParser:
    """Streaming splitter for a model's tool-call spans.

    Text between the tokenizer's ``tool_call_start`` and ``tool_call_end`` markers
    is routed to the per-model ``parse`` callable and emitted as OpenAI-shaped
    tool calls; everything else is content. A start marker with no ``end`` runs to
    the end of the stream. A span that fails to parse (truncated mid-generation)
    is dropped.
    """

    def __init__(self, start: str, end: str | None, parse: Callable[..., Any], tools: Any) -> None:
        self._start = start
        self._end = end
        self._parse = parse
        self._tools = tools
        self._buf = ""
        self._in_tool = False

    def push(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        self._buf += text
        return self._scan(final=False)

    def flush(self) -> tuple[str, list[dict[str, Any]]]:
        return self._scan(final=True)

    def _parse_into(self, text: str, out: list[dict[str, Any]]) -> None:
        try:
            parsed = self._parse(text, self._tools)
        except (ValueError, json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("failed to parse tool call (%s); likely truncated", e)
            return
        for tc in parsed if isinstance(parsed, list) else [parsed]:
            out.append(_format_tool_call(tc))

    def _scan(self, final: bool) -> tuple[str, list[dict[str, Any]]]:
        content: list[str] = []
        calls: list[dict[str, Any]] = []
        while True:
            if not self._in_tool:
                i = self._buf.find(self._start)
                if i == -1:
                    keep = 0 if final else _tool_partial_suffix(self._buf, self._start)
                    cut = len(self._buf) - keep
                    content.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                content.append(self._buf[:i])
                self._buf = self._buf[i + len(self._start) :]
                self._in_tool = True
            else:
                j = self._buf.find(self._end) if self._end else -1
                if self._end and j != -1:
                    self._parse_into(self._buf[:j], calls)
                    self._buf = self._buf[j + len(self._end) :]
                    self._in_tool = False
                    continue
                # No end marker yet: hold the span open until it closes or the
                # stream ends (an end-less tool syntax runs to end of stream).
                if final:
                    self._parse_into(self._buf, calls)
                    self._buf = ""
                    self._in_tool = False
                break
        return "".join(content), calls


def _parse_tool_calls(
    chunks: Iterator[Completion],
    start: str | None,
    end: str | None,
    parse: Callable[..., Any] | None,
    tools: Any,
) -> Iterator[Completion]:
    """Re-emit `chunks` with tool-call spans lifted into `Completion.tool_calls`.

    Content outside the spans passes through unchanged; the terminal
    `finish_reason` becomes ``"tool_calls"`` once any call was emitted (OpenAI
    contract). A pass-through when the tokenizer exposes no tool-call markers.
    """
    if not start or parse is None:
        yield from chunks
        return
    parser = _ToolCallParser(start, end, parse, tools)
    saw_tool = False
    for chunk in chunks:
        content, calls = parser.push(chunk.text)
        terminal = chunk.finish_reason is not None
        if terminal:
            tail_content, tail_calls = parser.flush()
            content += tail_content
            calls += tail_calls
        if calls:
            saw_tool = True
        finish = chunk.finish_reason
        if terminal and saw_tool and finish == "stop":
            finish = "tool_calls"
        if content or calls or chunk.reasoning or terminal:
            yield Completion(
                text=content,
                reasoning=chunk.reasoning,
                finish_reason=finish,
                prompt_tokens=chunk.prompt_tokens,
                completion_tokens=chunk.completion_tokens,
                tool_calls=tuple(calls),
            )


def _estimate_size_bytes(
    baseline: int | None, post: int | None, entry: registry.ModelEntry | None
) -> int:
    """Best estimate of a freshly-loaded model's resident size in bytes.

    Prefers the measured Metal delta (`post - baseline`); falls back to the
    registry entry's recorded `size_bytes`; finally a conservative 1 GiB.
    """
    measured = None
    if baseline is not None and post is not None and post >= baseline:
        measured = post - baseline
    recorded = entry.size_bytes if entry is not None else 0
    return max(measured or 0, recorded, _FALLBACK_SIZE_BYTES)
