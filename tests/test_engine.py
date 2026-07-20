from __future__ import annotations

import json
import sys
import threading
import types

import pytest

from omlx import engine, registry
from omlx.engine import Completion, ModelManager, SamplingParams


class _EncTokenizer:
    """Base tokenizer fake: char-code encoding + BOS heuristic for the cache path."""

    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


class FakeCache:
    """Minimal trimmable KV cache: tracks a token `offset` like the real caches."""

    def __init__(self):
        self.offset = 0

    def is_trimmable(self):
        return True


def _fake_cache_module():
    """A stand-in `mlx_lm.models.cache` with the three helpers the engine calls."""
    mod = types.ModuleType("mlx_lm.models.cache")
    mod.make_prompt_cache = lambda model, *a, **k: [FakeCache()]
    mod.can_trim_prompt_cache = lambda cache: all(c.is_trimmable() for c in cache)

    def trim(cache, n):
        for c in cache:
            c.offset = max(0, c.offset - n)
        return n

    mod.trim_prompt_cache = trim
    return mod


def _plen(prompt):
    """Token count of a prefill input (a token list or a raw string)."""
    return len(prompt)


class FakeTokenizer(_EncTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "PROMPT:" + messages[-1]["content"]


@pytest.fixture
def fake_mlx(monkeypatch):
    """Inject fake mlx_lm / mlx.core modules so the engine loads without Metal.

    `get_active_memory()` is a monotonic counter incremented on each `load`,
    so size-backfill tests can drive deterministic deltas without Metal.
    """
    # seed -1 = "not applied"; `threads` records the worker each mlx call ran on.
    calls = {"load": 0, "active_mem": 0, "seed": -1, "threads": []}

    def load(source):
        calls["load"] += 1
        calls["threads"].append(threading.current_thread().name)
        # Simulate weights landing in Metal memory: each load adds 512 MiB.
        calls["active_mem"] += 512 * 1024 * 1024
        return object(), FakeTokenizer()

    def stream_generate(model, tokenizer, prompt, max_tokens, **kwargs):
        # Record the prefilled token count and advance the (optional) cache the
        # way real generation would: prompt tokens, then one per generated token.
        calls["last_prompt_len"] = _plen(prompt)
        calls["threads"].append(threading.current_thread().name)
        cache = kwargs.get("prompt_cache")
        if cache is not None:
            cache[0].offset += _plen(prompt)
        for i, (text, finish) in enumerate((("a", None), ("b", "stop"))):
            if cache is not None:
                cache[0].offset += 1
            calls["threads"].append(threading.current_thread().name)
            yield types.SimpleNamespace(
                text=text,
                finish_reason=finish,
                prompt_tokens=_plen(prompt),
                generation_tokens=i + 1,
            )

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = load
    mlx_lm.stream_generate = stream_generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **k: object()
    sample_utils.make_logits_processors = lambda **k: []
    models = types.ModuleType("mlx_lm.models")
    cache_mod = _fake_cache_module()
    mx = types.ModuleType("mlx.core")
    mx.clear_cache = lambda: None
    mx.get_active_memory = lambda: calls["active_mem"]
    mx.random = types.SimpleNamespace(seed=lambda s: calls.__setitem__("seed", s))
    mx.array = lambda x: list(x)
    mlx = types.ModuleType("mlx")
    mlx.core = mx

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return calls


def test_resolve_uses_registered_repo_id(make_entry):
    registry.add(make_entry(name="Llama", repo_id="org/Llama", path="/p"))
    mgr = ModelManager(start_reaper=False)
    assert mgr._resolve("Llama") == "org/Llama"


def test_resolve_raises_for_unknown_model():
    """Auto-pull is restricted to `omlx pull` / `/api/pull`: an unknown model
    on a generation route surfaces as an OpenAI-shaped 404, not a silent fetch.
    """
    from omlx.protocol import OpenAIError

    mgr = ModelManager(start_reaper=False)
    with pytest.raises(OpenAIError) as exc:
        mgr._resolve("org/new")
    assert exc.value.status == 404
    assert exc.value.type == "not_found_error"
    assert exc.value.code == "model_not_found"
    assert exc.value.param == "model"


def test_count_cap_honors_explicit_settings(monkeypatch):
    """An explicit OMLX_MAX_LOADED_MODELS wins when max_loaded arg is unset."""
    monkeypatch.setattr(engine.settings, "max_loaded_models", 3)
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)
    assert mgr._max_loaded == 3


def test_count_cap_derived_from_injected_budget(monkeypatch):
    """With no explicit cap, the count cap derives from the manager's own budget."""
    monkeypatch.setattr(engine.settings, "max_loaded_models", None)
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)
    assert mgr._max_loaded == 8  # 65536 // 8192


def test_get_caches_and_resident_switch(fake_mlx, make_entry):
    """Loading a second model keeps both resident (LRU), unlike the old single-slot."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536, max_loaded=4)

    a1 = mgr.get("A")
    a2 = mgr.get("A")
    assert a1 is a2
    assert fake_mlx["load"] == 1
    assert mgr.loaded() == "A"

    mgr.get("B")
    assert fake_mlx["load"] == 2
    # Both still resident; B is most-recently-used.
    assert {info.name for info in mgr.loaded_models()} == {"A", "B"}
    assert mgr.loaded() == "B"


def test_get_evicts_lru_when_budget_exceeded(fake_mlx, make_entry):
    """A new model evicts the least-recently-used one when the budget won't fit."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    registry.add(make_entry(name="C", repo_id="org/C", path="/p"))
    # Each load is floored to ~1 GiB; budget 2048 MiB fits two, forces a third
    # to evict the least-recently-used.
    mgr = ModelManager(start_reaper=False, mem_budget_mb=2048, max_loaded=4)

    mgr.get("A")
    mgr.get("B")
    assert {info.name for info in mgr.loaded_models()} == {"A", "B"}

    mgr.get("C")  # would exceed 1024 MiB
    # C is resident; oldest LRU (A) was evicted.
    resident = {info.name for info in mgr.loaded_models()}
    assert "C" in resident
    assert "A" not in resident


def test_get_evicts_lru_when_count_exceeded(fake_mlx, make_entry):
    """A count cap evicts the least-recently-used one regardless of memory."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    registry.add(make_entry(name="C", repo_id="org/C", path="/p"))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536, max_loaded=2)

    mgr.get("A")
    mgr.get("B")
    mgr.get("C")  # exceeds count cap of 2
    assert {info.name for info in mgr.loaded_models()} == {"B", "C"}


def test_stream_chat_yields_completions(fake_mlx, make_entry):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    out = list(
        mgr.stream_chat("A", [{"role": "user", "content": "hi"}], SamplingParams(max_tokens=8))
    )
    assert "".join(c.text for c in out) == "ab"
    assert out[-1].finish_reason == "stop"
    # "PROMPT:hi" encodes to one token per char; first turn reuses nothing.
    assert out[-1].prompt_tokens == len("PROMPT:hi")
    assert out[-1].completion_tokens == 2


def test_stream_text_yields_completions(fake_mlx, make_entry):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    out = list(mgr.stream_text("A", "hello", SamplingParams(max_tokens=8)))
    assert "".join(c.text for c in out) == "ab"
    assert out[-1].finish_reason == "stop"


def test_mlx_work_runs_on_dedicated_worker_thread(fake_mlx, make_entry):
    """Load and every generation step run on the one MLX worker, never the caller.

    mlx-lm's generation stream is thread-local; if a token step ran on a
    different thread than the one that made the stream, real MLX would raise
    "There is no Stream(gpu, N) in current thread." Pinning all mx.* work to a
    single worker is what prevents that, so guard the affinity here.
    """
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)

    caller = threading.current_thread().name
    list(mgr.stream_text("A", "hello", SamplingParams(max_tokens=8)))

    threads = fake_mlx["threads"]
    # load + generator construction + one entry per generated token.
    assert len(threads) >= 3
    assert all(name.startswith("mlx") for name in threads)
    assert caller not in threads
    assert len(set(threads)) == 1  # the same worker across every step


def test_generation_kwargs_forwards_sampling(fake_mlx, monkeypatch):
    """Every sampling field reaches make_sampler / make_logits_processors."""
    seen: dict[str, dict] = {}

    def fake_sampler(**k):
        seen["sampler"] = k
        return "SAMPLER"

    def fake_processors(**k):
        seen["proc"] = k
        return ["PROC"]

    su = sys.modules["mlx_lm.sample_utils"]
    monkeypatch.setattr(su, "make_sampler", fake_sampler)
    monkeypatch.setattr(su, "make_logits_processors", fake_processors)
    params = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        frequency_penalty=0.3,
        presence_penalty=0.2,
        repetition_penalty=1.1,
        logit_bias={123: -5.0},
    )
    kw = engine._generation_kwargs(params)
    assert seen["sampler"] == {"temp": 0.5, "top_p": 0.9, "min_p": 0.05, "top_k": 40}
    assert seen["proc"] == {
        "logit_bias": {123: -5.0},
        "repetition_penalty": 1.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
    }
    assert kw["sampler"] == "SAMPLER" and kw["logits_processors"] == ["PROC"]
    assert "kv_bits" not in kw


def test_generation_kwargs_no_penalties_gives_none(fake_mlx):
    """With default penalties the processor list collapses to None (no overhead)."""
    assert engine._generation_kwargs(SamplingParams())["logits_processors"] is None


def test_generation_kwargs_kv_quant(fake_mlx, monkeypatch):
    """`OMLX_KV_BITS` and friends ride into stream_generate only when set."""
    from omlx.config import settings

    monkeypatch.setattr(settings, "kv_bits", 4)
    monkeypatch.setattr(settings, "kv_group_size", 32)
    monkeypatch.setattr(settings, "quantized_kv_start", 100)
    kw = engine._generation_kwargs(SamplingParams())
    assert kw["kv_bits"] == 4 and kw["kv_group_size"] == 32 and kw["quantized_kv_start"] == 100


def test_prompt_cache_reuses_prefix(fake_mlx, make_entry):
    """A second turn sharing a prefix prefills only the diverging suffix."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    p = SamplingParams(max_tokens=8)
    list(mgr.stream_chat("A", [{"role": "user", "content": "hi"}], p))
    assert fake_mlx["last_prompt_len"] == len("PROMPT:hi")  # full prefill, nothing cached yet
    out = list(mgr.stream_chat("A", [{"role": "user", "content": "hi there"}], p))
    # Only " there" (the divergence from "PROMPT:hi") is prefilled the second turn.
    assert fake_mlx["last_prompt_len"] == len("PROMPT:hi there") - len("PROMPT:hi")
    # Reused prefix is added back, so usage still reports the whole prompt.
    assert out[-1].prompt_tokens == len("PROMPT:hi there")
    lm = mgr.get("A")
    assert lm.cache_tokens == [ord(c) for c in "PROMPT:hi there"]
    assert lm.cache[0].offset == len(lm.cache_tokens)  # invariant: cache holds the prompt


def test_prompt_cache_disabled_prefills_full(fake_mlx, make_entry, monkeypatch):
    """With OMLX_PROMPT_CACHE off, no cache is built and the full prompt is prefilled."""
    from omlx.config import settings

    monkeypatch.setattr(settings, "prompt_cache", False)
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    list(mgr.stream_chat("A", [{"role": "user", "content": "hi"}], SamplingParams(max_tokens=8)))
    lm = mgr.get("A")
    assert lm.cache is None and lm.cache_tokens == []


def test_prompt_cache_concurrent_falls_back(fake_mlx, make_entry):
    """When the cache lock is held (a concurrent generation), the turn skips reuse."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    lm = mgr.get("A")
    lm.cache_lock.acquire()
    try:
        list(
            mgr.stream_chat("A", [{"role": "user", "content": "hi"}], SamplingParams(max_tokens=8))
        )
    finally:
        lm.cache_lock.release()
    assert lm.cache is None and lm.cache_tokens == []
    assert fake_mlx["last_prompt_len"] == len("PROMPT:hi")


def test_with_offset_adds_only_on_terminal():
    """`_with_offset` bumps the prompt count on the terminal completion only."""
    mid = Completion(text="x")
    assert engine._with_offset(mid, 5) is mid  # no finish_reason -> untouched
    terminal = Completion(text="", finish_reason="stop", prompt_tokens=2)
    assert engine._with_offset(terminal, 5).prompt_tokens == 7
    assert engine._with_offset(terminal, 0) is terminal  # zero offset -> untouched


def test_lcp():
    assert engine._lcp([1, 2, 3], [1, 2, 9, 4]) == 2  # diverge mid-sequence
    assert engine._lcp([1, 2], [1, 2, 3]) == 2  # one extends the other
    assert engine._lcp([], [1]) == 0


def test_prompt_cache_skips_tiny_prompt(fake_mlx, make_entry, monkeypatch):
    """A prompt too short to be worth caching takes the plain string path."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    lm = mgr.get("A")
    monkeypatch.setattr(lm.tokenizer, "apply_chat_template", lambda *a, **k: "x")
    list(mgr.stream_chat("A", [{"role": "user", "content": "hi"}], SamplingParams(max_tokens=8)))
    assert lm.cache is None and lm.cache_tokens == []
    assert fake_mlx["last_prompt_len"] == 1  # the raw string "x", not a token suffix


def test_prompt_cache_untrimmable_is_rebuilt_then_dropped(fake_mlx, make_entry, monkeypatch):
    """An existing cache that stops being trimmable is rebuilt, then dropped on finalize."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    p = SamplingParams(max_tokens=8)
    list(mgr.stream_chat("A", [{"role": "user", "content": "hi"}], p))
    lm = mgr.get("A")
    assert lm.cache is not None
    monkeypatch.setattr(
        sys.modules["mlx_lm.models.cache"], "can_trim_prompt_cache", lambda c: False
    )
    list(mgr.stream_chat("A", [{"role": "user", "content": "hi again"}], p))
    assert lm.cache is None and lm.cache_tokens == []


def test_prompt_cache_setup_failure_falls_back(fake_mlx, make_entry, monkeypatch):
    """A cache build error releases the lock and prefills the full string prompt."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    lm = mgr.get("A")

    def boom(model, *a, **k):
        raise RuntimeError("no cache")

    monkeypatch.setattr(sys.modules["mlx_lm.models.cache"], "make_prompt_cache", boom)
    list(mgr.stream_chat("A", [{"role": "user", "content": "hi"}], SamplingParams(max_tokens=8)))
    assert lm.cache is None
    assert not lm.cache_lock.locked()  # lock was released on the fallback path
    assert fake_mlx["last_prompt_len"] == len("PROMPT:hi")


def test_prompt_cache_finalize_failure_drops_cache(fake_mlx, make_entry, monkeypatch):
    """A trim error while finalizing drops the cache rather than leaving it inconsistent."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    lm = mgr.get("A")

    def trim(cache, n):
        if n > 0:  # prepare trims 0 (fine); finalize trims the generated tokens
            raise RuntimeError("trim failed")
        return 0

    monkeypatch.setattr(sys.modules["mlx_lm.models.cache"], "trim_prompt_cache", trim)
    list(mgr.stream_chat("A", [{"role": "user", "content": "hi"}], SamplingParams(max_tokens=8)))
    assert lm.cache is None and lm.cache_tokens == []


def _char_stream(text, finish="stop"):
    """A stream_generate stand-in that emits `text` one character per step."""

    def gen(model, tokenizer, prompt, max_tokens, **kwargs):
        last = len(text) - 1
        for i, ch in enumerate(text):
            yield types.SimpleNamespace(
                text=ch,
                finish_reason=finish if i == last else None,
                prompt_tokens=3,
                generation_tokens=i + 1,
            )

    return gen


def _chunk_stream(chunks, finish="stop"):
    """A stream_generate stand-in that emits `chunks` verbatim, terminal last."""

    def gen(model, tokenizer, prompt, max_tokens, **kwargs):
        last = len(chunks) - 1
        for i, c in enumerate(chunks):
            yield types.SimpleNamespace(
                text=c,
                finish_reason=finish if i == last else None,
                prompt_tokens=3,
                generation_tokens=i + 1,
            )

    return gen


_HARMONY = (
    "<|channel|>analysis<|message|>THINK<|end|>"
    "<|start|>assistant<|channel|>final<|message|>ANSWER<|return|>"
)


def _chat(mgr):
    return list(
        mgr.stream_chat("A", [{"role": "user", "content": "hi"}], SamplingParams(max_tokens=64))
    )


def test_harmony_splits_reasoning_from_content(fake_mlx, make_entry, monkeypatch):
    """gpt-oss channels: analysis -> reasoning, final -> content, tokens stripped."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    # One char per step: every control token is split across boundaries.
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream(_HARMONY))
    out = _chat(ModelManager(start_reaper=False))
    assert "".join(c.text for c in out) == "ANSWER"
    assert "".join(c.reasoning for c in out) == "THINK"
    assert "<|" not in "".join(c.text + c.reasoning for c in out)  # no leaked control tokens
    assert out[-1].finish_reason == "stop"


def test_harmony_control_token_split_across_chunks(fake_mlx, make_entry, monkeypatch):
    """Multi-char chunks that bisect control tokens still parse cleanly."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    chunks = [
        "<|chan",
        "nel|>analysis<|mess",
        "age|>TH",
        "INK<|end|><|start|>assistant<|channel|>fin",
        "al<|message|>ANS",
        "WER<|return|>",
    ]
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _chunk_stream(chunks))
    out = _chat(ModelManager(start_reaper=False))
    assert "".join(c.text for c in out) == "ANSWER"
    assert "".join(c.reasoning for c in out) == "THINK"


def test_harmony_parser_inert_for_plain_text(fake_mlx, make_entry, monkeypatch):
    """A non-reasoning model's output is unchanged and produces no reasoning."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream("hello"))
    out = _chat(ModelManager(start_reaper=False))
    assert "".join(c.text for c in out) == "hello"
    assert all(c.reasoning == "" for c in out)


def test_harmony_preserves_literal_angle_brackets(fake_mlx, make_entry, monkeypatch):
    """A bare `<|foo` in content (not a control token) survives intact."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream("a <| b < c"))
    out = _chat(ModelManager(start_reaper=False))
    assert "".join(c.text for c in out) == "a <| b < c"


class FakeToolTokenizer(_EncTokenizer):
    """A tool-capable tokenizer: `<tool_call>`-delimited JSON, echoing tools passed."""

    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    def __init__(self):
        self.seen_tools = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, tools=None):
        self.seen_tools = tools
        return "PROMPT"

    @staticmethod
    def tool_parser(text, tools=None):
        return json.loads(text.strip())


def _tool_chat(mgr, tools):
    return list(
        mgr.stream_chat(
            "A", [{"role": "user", "content": "hi"}], SamplingParams(max_tokens=64), tools=tools
        )
    )


def _load_tool_tokenizer(monkeypatch):
    tok = FakeToolTokenizer()
    monkeypatch.setattr(sys.modules["mlx_lm"], "load", lambda source: (object(), tok))
    return tok


def test_stream_chat_parses_tool_call(fake_mlx, make_entry, monkeypatch):
    """A `<tool_call>` span becomes a tool_call; surrounding text stays content."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    tok = _load_tool_tokenizer(monkeypatch)
    body = 'pre<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>post'
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream(body))
    tools = [{"type": "function", "function": {"name": "f"}}]
    out = _tool_chat(ModelManager(start_reaper=False), tools)

    assert tok.seen_tools == tools  # tools reached the template
    assert "".join(c.text for c in out) == "prepost"
    calls = [tc for c in out for tc in c.tool_calls]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "f"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
    assert out[-1].finish_reason == "tool_calls"


def test_stream_chat_tool_call_split_across_chunks(fake_mlx, make_entry, monkeypatch):
    """Delimiters bisected across generation steps still parse to one tool call."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    _load_tool_tokenizer(monkeypatch)
    chunks = ["<tool_", 'call>{"name": "f", ', '"arguments": {}}</tool', "_call>"]
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _chunk_stream(chunks))
    out = _tool_chat(
        ModelManager(start_reaper=False), [{"type": "function", "function": {"name": "f"}}]
    )
    calls = [tc for c in out for tc in c.tool_calls]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "f"
    assert "".join(c.text for c in out) == ""  # no delimiter fragments leaked into content


def test_stream_chat_drops_truncated_tool_call(fake_mlx, make_entry, monkeypatch):
    """An unterminated / unparseable tool span is dropped; finish stays stop."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    _load_tool_tokenizer(monkeypatch)
    monkeypatch.setattr(
        sys.modules["mlx_lm"], "stream_generate", _char_stream('<tool_call>{"name": bro')
    )
    out = _tool_chat(
        ModelManager(start_reaper=False), [{"type": "function", "function": {"name": "f"}}]
    )
    assert [tc for c in out for tc in c.tool_calls] == []
    assert out[-1].finish_reason == "stop"


def test_stream_chat_without_tool_support_ignores_tools(fake_mlx, make_entry, monkeypatch):
    """A tokenizer with no tool support leaves output untouched even when tools are passed."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    # Default FakeTokenizer has no has_tool_calling; its <tool_call> text passes through.
    monkeypatch.setattr(
        sys.modules["mlx_lm"], "stream_generate", _char_stream("<tool_call>x</tool_call>")
    )
    out = _tool_chat(
        ModelManager(start_reaper=False), [{"type": "function", "function": {"name": "f"}}]
    )
    assert "".join(c.text for c in out) == "<tool_call>x</tool_call>"
    assert [tc for c in out for tc in c.tool_calls] == []


def test_parse_tool_calls_passthrough_without_markers():
    """No start marker / parser => chunks pass through untouched (defensive guard)."""
    chunks = [Completion(text="hi", finish_reason="stop")]
    out = list(engine._parse_tool_calls(iter(chunks), None, None, None, None))
    assert out == chunks


class FakeHarmonyTokenizer(_EncTokenizer):
    """gpt-oss: reports no has_tool_calling; Harmony support inferred from vocab."""

    has_tool_calling = False

    def __init__(self):
        self.seen_tools = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, tools=None):
        self.seen_tools = tools
        return "PROMPT"

    def get_vocab(self):
        return {"<|call|>": 1}


def _load_harmony_tokenizer(monkeypatch):
    tok = FakeHarmonyTokenizer()
    monkeypatch.setattr(sys.modules["mlx_lm"], "load", lambda source: (object(), tok))
    return tok


_HARMONY_TOOL = (
    "<|channel|>commentary to=functions.get_weather<|constrain|>json"
    '<|message|>{"city": "SF"}<|call|>'
)


def test_harmony_offers_tools_via_vocab_probe(fake_mlx, make_entry, monkeypatch):
    """gpt-oss has no has_tool_calling, yet tools still reach the template."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    tok = _load_harmony_tokenizer(monkeypatch)
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream("hi"))
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    _tool_chat(ModelManager(start_reaper=False), tools)
    assert tok.seen_tools == tools


def test_harmony_parses_tool_call(fake_mlx, make_entry, monkeypatch):
    """A commentary to=functions call becomes a tool_call; args don't leak to content.

    `_char_stream` emits one char per step, so every control token is split — this
    also guards the streaming path.
    """
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    _load_harmony_tokenizer(monkeypatch)
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream(_HARMONY_TOOL))
    out = _tool_chat(
        ModelManager(start_reaper=False),
        [{"type": "function", "function": {"name": "get_weather"}}],
    )
    calls = [tc for c in out for tc in c.tool_calls]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "SF"}
    assert "".join(c.text for c in out) == ""
    assert out[-1].finish_reason == "tool_calls"


def test_harmony_reasoning_then_tool_call(fake_mlx, make_entry, monkeypatch):
    """An analysis block still routes to reasoning while a following call is captured."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    _load_harmony_tokenizer(monkeypatch)
    body = (
        "<|channel|>analysis<|message|>THINK<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.f<|message|>{}<|call|>"
    )
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream(body))
    out = _tool_chat(
        ModelManager(start_reaper=False), [{"type": "function", "function": {"name": "f"}}]
    )
    assert "".join(c.reasoning for c in out) == "THINK"
    assert "".join(c.text for c in out) == ""
    calls = [tc for c in out for tc in c.tool_calls]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "f"
    assert out[-1].finish_reason == "tool_calls"


def test_harmony_multiple_tool_calls(fake_mlx, make_entry, monkeypatch):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    _load_harmony_tokenizer(monkeypatch)
    body = (
        "<|channel|>commentary to=functions.a<|message|>{}<|call|>"
        "<|start|>assistant<|channel|>commentary to=functions.b<|message|>{}<|call|>"
    )
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream(body))
    out = _tool_chat(
        ModelManager(start_reaper=False), [{"type": "function", "function": {"name": "a"}}]
    )
    names = [tc["function"]["name"] for c in out for tc in c.tool_calls]
    assert names == ["a", "b"]


def test_harmony_tool_call_empty_args(fake_mlx, make_entry, monkeypatch):
    """A no-argument commentary call emits `"{}"`, not `""` (clients JSON.parse it)."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    _load_harmony_tokenizer(monkeypatch)
    body = "<|channel|>commentary to=functions.f<|message|><|call|>"
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream(body))
    out = _tool_chat(
        ModelManager(start_reaper=False), [{"type": "function", "function": {"name": "f"}}]
    )
    calls = [tc for c in out for tc in c.tool_calls]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "f"
    assert calls[0]["function"]["arguments"] == "{}"
    assert json.loads(calls[0]["function"]["arguments"]) == {}


def test_format_tool_call_empty_arguments():
    """Empty / whitespace argument bodies normalize to `"{}"`; real payloads pass through."""
    assert engine._format_tool_call({"name": "f", "arguments": ""})["function"]["arguments"] == "{}"
    assert (
        engine._format_tool_call({"name": "f", "arguments": "  "})["function"]["arguments"] == "{}"
    )
    assert engine._format_tool_call({"name": "f", "arguments": {}})["function"]["arguments"] == "{}"
    assert (
        engine._format_tool_call({"name": "f", "arguments": {"x": 1}})["function"]["arguments"]
        == '{"x": 1}'
    )
    assert (
        engine._format_tool_call({"name": "f", "arguments": '{"x": 1}'})["function"]["arguments"]
        == '{"x": 1}'
    )


def test_harmony_plain_answer_yields_no_tool_calls(fake_mlx, make_entry, monkeypatch):
    """A normal Harmony answer (no commentary call) produces no tool_calls, finish stop."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    _load_harmony_tokenizer(monkeypatch)
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream(_HARMONY))
    out = _tool_chat(
        ModelManager(start_reaper=False), [{"type": "function", "function": {"name": "f"}}]
    )
    assert [tc for c in out for tc in c.tool_calls] == []
    assert "".join(c.text for c in out) == "ANSWER"
    assert out[-1].finish_reason == "stop"


def test_stop_sequence_truncates_output(fake_mlx, make_entry, monkeypatch):
    """A stop sequence halts generation and truncates the text at its start."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream("hello\n\nworld"))
    mgr = ModelManager(start_reaper=False)
    out = list(mgr.stream_text("A", "hi", SamplingParams(max_tokens=64, stop=("\n\n",))))
    assert "".join(c.text for c in out) == "hello"
    assert out[-1].finish_reason == "stop"


def test_stop_sequence_spanning_token_boundary(fake_mlx, make_entry, monkeypatch):
    """A multi-char stop split across single-char steps is still caught."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream("abENDcd"))
    mgr = ModelManager(start_reaper=False)
    out = list(mgr.stream_text("A", "hi", SamplingParams(max_tokens=64, stop=("END",))))
    assert "".join(c.text for c in out) == "ab"
    assert out[-1].finish_reason == "stop"


def test_stop_sequence_absent_passes_through(fake_mlx, make_entry, monkeypatch):
    """When no stop sequence matches, the full text and terminal reason survive."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _char_stream("hello"))
    mgr = ModelManager(start_reaper=False)
    out = list(mgr.stream_text("A", "hi", SamplingParams(max_tokens=64, stop=("XYZ",))))
    assert "".join(c.text for c in out) == "hello"
    assert out[-1].finish_reason == "stop"


def test_seed_is_applied(fake_mlx, make_entry):
    """A request seed reaches mx.random.seed before generation."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    list(mgr.stream_text("A", "hi", SamplingParams(max_tokens=8, seed=42)))
    assert fake_mlx["seed"] == 42


def test_stream_reports_length_finish(fake_mlx, make_entry, monkeypatch):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))

    def truncated(model, tokenizer, prompt, max_tokens, **kwargs):
        yield types.SimpleNamespace(
            text="a", finish_reason=None, prompt_tokens=1, generation_tokens=1
        )
        yield types.SimpleNamespace(
            text="", finish_reason="length", prompt_tokens=1, generation_tokens=2
        )

    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", truncated)
    mgr = ModelManager(start_reaper=False)
    out = list(mgr.stream_text("A", "hello", SamplingParams(max_tokens=2)))
    assert out[-1].finish_reason == "length"


def test_reaper_disabled_leaves_no_thread():
    before = engine.threading.active_count()
    ModelManager(start_reaper=False)
    assert engine.threading.active_count() == before


class _StopLoop(Exception):
    """Breaks out of the otherwise-infinite reaper loop after one iteration."""


def _run_one_reap(mgr, monkeypatch):
    """Drive `_reap_loop` through exactly one check, then abort the loop."""
    n = {"i": 0}

    def fake_sleep(_):
        n["i"] += 1
        if n["i"] > 1:
            raise _StopLoop

    monkeypatch.setattr(engine.time, "sleep", fake_sleep)
    with pytest.raises(_StopLoop):
        mgr._reap_loop()


def test_reaper_evicts_idle_model(fake_mlx, make_entry, monkeypatch):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(keepalive_seconds=0, start_reaper=False)
    mgr.get("A")
    assert mgr.loaded() == "A"

    _run_one_reap(mgr, monkeypatch)
    assert mgr.loaded() is None


def test_reaper_keeps_model_during_active_stream(fake_mlx, make_entry, monkeypatch):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(keepalive_seconds=0, start_reaper=False, mem_budget_mb=65536)
    mgr.get("A")
    # Bump the per-model active counter directly (mirrors _active_generation).
    lm = next(iter(mgr._loaded.values()))
    lm.active = 1

    _run_one_reap(mgr, monkeypatch)
    assert mgr.loaded() == "A"


def test_reaper_keeps_forever_when_keepalive_negative(fake_mlx, make_entry, monkeypatch):
    """keepalive_seconds < 0 means "keep forever": an idle model survives a reap."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(keepalive_seconds=-1, start_reaper=False, mem_budget_mb=65536)
    mgr.get("A")
    assert mgr.loaded() == "A"

    _run_one_reap(mgr, monkeypatch)
    assert mgr.loaded() == "A"


def test_active_generation_tracks_in_flight_and_refreshes_last_used(
    fake_mlx, make_entry, monkeypatch
):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)
    lm = mgr.get("A")
    assert lm.active == 0

    before = lm.last_used
    frozen = before + 1000
    monkeypatch.setattr(engine.time, "time", lambda: frozen)

    with mgr._active_generation(lm):
        assert lm.active == 1
    assert lm.active == 0
    assert lm.last_used == frozen


def test_active_generation_decrements_and_refreshes_on_exception(fake_mlx, make_entry, monkeypatch):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)
    lm = mgr.get("A")

    before = lm.last_used
    frozen = before + 1000
    monkeypatch.setattr(engine.time, "time", lambda: frozen)

    with pytest.raises(RuntimeError, match="boom"):
        with mgr._active_generation(lm):
            raise RuntimeError("boom")
    assert lm.active == 0
    assert lm.last_used == frozen


def test_active_generation_is_per_model(fake_mlx, make_entry, monkeypatch):
    """Bumping A's counter protects only A; an idle B can still be reaped."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    mgr = ModelManager(keepalive_seconds=0, start_reaper=False, mem_budget_mb=65536)
    a = mgr.get("A")
    mgr.get("B")

    with mgr._active_generation(a):
        _run_one_reap(mgr, monkeypatch)
        # A is active; B is idle past TTL → only B should be evicted.
        resident = {info.name for info in mgr.loaded_models()}
        assert resident == {"A"}


def test_get_reuses_concurrent_load_when_racer_wins(fake_mlx, make_entry, monkeypatch):
    """If another caller loaded the same model while we waited on the cold
    load (so the fast-path early return didn't fire because the dict didn't
    yet contain it when we entered), drop our freshly-loaded model and reuse
    the racer's.
    """
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)

    mgr.get("B")
    assert mgr.loaded() == "B"

    racer_model, racer_tok = object(), object()
    real_load = sys.modules["mlx_lm"].load

    def slow_load(source):
        if "A" in mgr._loaded:
            return real_load(source)
        racer = engine.LoadedModel("A", racer_model, racer_tok, engine.time.time())
        mgr._loaded["A"] = racer
        return object(), object()

    monkeypatch.setattr(sys.modules["mlx_lm"], "load", slow_load)
    got = mgr.get("A")
    assert got.model is racer_model
    assert mgr.loaded() == "A"


def test_unload_logged_when_clear_cache_raises(fake_mlx, make_entry, monkeypatch):
    """clear_cache failure during unload is swallowed."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(keepalive_seconds=0, start_reaper=False)
    mgr.get("A")
    assert mgr.loaded() == "A"

    sys.modules["mlx.core"].clear_cache = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    _run_one_reap(mgr, monkeypatch)
    assert mgr.loaded() is None


def test_unload_locked_unknown_name_is_noop(fake_mlx, make_entry):
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)
    mgr._unload_locked("nonexistent")  # must not raise
    assert mgr.loaded_models() == []


def test_unload_all_when_empty_is_noop(fake_mlx):
    """`_unload_locked(None)` with an empty resident set returns without touching mlx."""
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)
    mgr._unload_locked(None)  # covers the early-return branch
    assert mgr.loaded_models() == []


def test_generation_kwargs_prepends_json_provider(fake_mlx, monkeypatch):
    """A `json_provider` is prepended to the logits-processor chain (before penalties)."""
    su = sys.modules["mlx_lm.sample_utils"]
    monkeypatch.setattr(su, "make_logits_processors", lambda **k: ["PENALTY"])

    def json_proc(_tokens: object, _logits: object) -> object:
        return _logits

    kw = engine._generation_kwargs(SamplingParams(), json_provider=json_proc)
    # The same object the caller supplied sits first in the processor chain.
    assert kw["logits_processors"][0] is json_proc
    assert kw["logits_processors"][1] == "PENALTY"


def test_json_processor_builds_harmony_aware_for_gpt_oss():
    """For a Harmony tokenizer the processor defers to the final channel."""
    from omlx._harmony import is_harmony

    class _HarmonyTok:
        def get_vocab(self):
            return {"<|call|>": 1}

        def decode(self, ids):
            return ""

    # Smoke: `_json_processor` returns a callable; harmony behavior is gated
    # by `is_harmony` and exercised in `tests/test_json.py`.
    proc = engine._json_processor(_HarmonyTok())
    assert callable(proc)
    assert is_harmony(_HarmonyTok())


def test_json_processor_builds_non_harmony_for_plain_tokenizer():
    class _PlainTok:
        def get_vocab(self):
            return {"not_harmony": 1}

        def decode(self, ids):
            return ""

    proc = engine._json_processor(_PlainTok())
    assert callable(proc)


def test_unload_all_clears_resident_set(fake_mlx, make_entry):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)
    mgr.get("A")
    mgr.get("B")
    mgr._unload_locked(None)
    assert mgr.loaded_models() == []


def test_ps_returns_ollama_shape(fake_mlx, make_entry):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(keepalive_seconds=120, start_reaper=False, mem_budget_mb=65536)
    mgr.get("A")
    listing = mgr.ps()
    assert len(listing) == 1
    entry = listing[0]
    assert entry["name"] == "A"
    assert entry["model"] == "A"
    assert entry["digest"] == ""
    assert entry["size"] == entry["size_vram"]
    # ISO-8601 expiry string is set for finite keepalive_seconds
    assert isinstance(entry["expires_at"], str) and "T" in entry["expires_at"]


def test_ps_expires_at_none_when_keepalive_disabled(fake_mlx, make_entry):
    """keepalive_seconds < 0 means "keep forever" -> expires_at is None."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(keepalive_seconds=-1, start_reaper=False, mem_budget_mb=65536)
    mgr.get("A")
    assert mgr.ps()[0]["expires_at"] is None


def test_size_falls_back_to_recorded_when_metal_unavailable(fake_mlx, make_entry, monkeypatch):
    """When mx.get_active_memory() raises, size is taken from the registry entry."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p", size_bytes=700 * 1024 * 1024))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)

    # Force the probe to fail outright (no Metal, ImportError inside helper).
    def _boom():
        raise RuntimeError("x")

    monkeypatch.setattr(sys.modules["mlx.core"], "get_active_memory", _boom)
    mgr.get("A")
    info = mgr.loaded_models()[0]
    # Recorded (700 MiB) is floored up to the 1 GiB minimum.
    assert info.size_bytes == 1024 * 1024 * 1024


def test_evict_to_fit_skipped_when_all_resident_is_active(fake_mlx, make_entry, monkeypatch):
    """If every resident model is mid-generation, we can't evict; the newcomer still loads."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    registry.add(make_entry(name="C", repo_id="org/C", path="/p"))
    # Budget that fits at most one model; the active A should block eviction
    # via the "continue" path in _evict_to_fit_locked.
    mgr = ModelManager(start_reaper=False, mem_budget_mb=1024, max_loaded=4)
    a = mgr.get("A")
    a.active = 1  # mark as in-flight; reaper/evictor must spare it

    # Loading B will exceed budget, but A is active → eviction is skipped and B loads anyway.
    assert mgr.get("B").name == "B"
    # A survives because it shielded itself; B joins despite the over-budget state.
    resident = {info.name for info in mgr.loaded_models()}
    assert resident == {"A", "B"}


def test_get_reuses_model_that_lands_before_preevict(fake_mlx, make_entry, monkeypatch):
    """A racer that finishes loading between the fast-path miss and the
    pre-eviction lock is reused, not reloaded (the pre-evict dup re-check).
    """
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536)

    racer = engine.LoadedModel("A", object(), object(), engine.time.time())
    real_resolve = mgr._resolve

    def resolve_then_race(name):
        source = real_resolve(name)
        # Another caller lands "A" resident before we reach the pre-evict lock.
        mgr._loaded["A"] = racer
        return source

    monkeypatch.setattr(mgr, "_resolve", resolve_then_race)

    got = mgr.get("A")
    assert got is racer
    assert fake_mlx["load"] == 0  # our own load never ran


def test_evicts_before_loading_new_model(fake_mlx, make_entry, monkeypatch):
    """Over-budget eviction must happen *before* the new model is loaded, else
    the old and new models are briefly co-resident in GPU memory — which on
    Metal is an uncatchable OOM abort that crashes the daemon.
    """
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    # Budget fits one model (each floors to 1 GiB); loading B must evict A first.
    mgr = ModelManager(start_reaper=False, mem_budget_mb=1024, max_loaded=4)
    mgr.get("A")

    resident_at_load = {}
    real_load = sys.modules["mlx_lm"].load

    def spy_load(source):
        resident_at_load["names"] = set(mgr._loaded)
        return real_load(source)

    monkeypatch.setattr(sys.modules["mlx_lm"], "load", spy_load)
    mgr.get("B")

    # A was already gone when B's weights were loaded — never co-resident.
    assert resident_at_load["names"] == set()
    assert {info.name for info in mgr.loaded_models()} == {"B"}


def test_evict_to_count_skipped_when_all_resident_is_active(fake_mlx, make_entry):
    """The "everything resident is active" branch of _evict_to_count_locked."""
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    registry.add(make_entry(name="B", repo_id="org/B", path="/p"))
    registry.add(make_entry(name="C", repo_id="org/C", path="/p"))
    mgr = ModelManager(start_reaper=False, mem_budget_mb=65536, max_loaded=2)

    a = mgr.get("A")
    a.active = 1
    mgr.get("B")  # at cap; A is active → eviction of A is impossible
    # We're now at the cap. Mark B active too and load C; neither A nor B can be evicted.
    mgr.get("B").active = 1
    c = mgr.get("C")
    # count cap should have been respected (no eviction) because no eviction was possible
    # for a guest that we explicitly requested → resident grows past the count cap.
    assert c.name == "C"
    assert len(mgr._loaded) == 3
