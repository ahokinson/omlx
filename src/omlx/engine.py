"""Model lifecycle: load, cache, stream, and idle-unload MLX models."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import config, registry
from .config import settings
from .protocol import Completion, SamplingParams

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

        Known models load from their recorded path/repo; unknown names are
        auto-pulled, then loaded.
        """
        entry = registry.get(name)
        if entry is not None:
            return entry.repo_id
        from .pull import pull

        return pull(name).repo_id

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

        # Resolve + load outside the lock: a cold pull/load can take minutes,
        # and holding the lock would block /health the whole time.
        source = self._resolve(name)
        baseline = _metal_active_memory()
        from mlx_lm import load

        # load() returns (model, tokenizer), plus a config when return_config is
        # set; star-unpack tolerates either arity.
        model, tokenizer, *_ = load(source)
        post = _metal_active_memory()
        size_bytes = _estimate_size_bytes(baseline, post, registry.get(name))

        with self._lock:
            # Re-check after reacquiring the lock: another caller may have
            # loaded the same model while we waited on the cold load above.
            # If so, drop ours and reuse theirs to avoid a duplicate resident.
            existing = self._loaded.get(name)
            if existing is not None:
                self._loaded.move_to_end(name)
                existing.last_used = time.time()
                return existing
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
        self, name: str, messages: list[dict[str, str]], params: SamplingParams | None = None
    ) -> Iterator[Completion]:
        """Stream generated tokens for `messages` via the model's chat template."""
        lm = self.get(name)
        prompt = lm.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        yield from self._stream_prompt(lm, prompt, params or SamplingParams())

    def stream_text(
        self, name: str, prompt: str, params: SamplingParams | None = None
    ) -> Iterator[Completion]:
        """Stream generated tokens for a raw `prompt` (no chat template)."""
        lm = self.get(name)
        yield from self._stream_prompt(lm, prompt, params or SamplingParams())

    def _stream_prompt(
        self, lm: LoadedModel, prompt: str, params: SamplingParams
    ) -> Iterator[Completion]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        if params.seed is not None:
            import mlx.core as mx  # ty: ignore[unresolved-import]

            mx.random.seed(params.seed)
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)
        gen = stream_generate(
            lm.model,
            lm.tokenizer,
            prompt=prompt,
            max_tokens=params.max_tokens,
            sampler=sampler,
        )
        with self._active_generation(lm):
            if not params.stop:
                for resp in gen:
                    yield Completion(
                        text=resp.text,
                        finish_reason=resp.finish_reason,
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.generation_tokens,
                    )
            else:
                yield from _stream_with_stops(gen, params.stop)

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
