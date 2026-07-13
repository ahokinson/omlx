from __future__ import annotations

import json
import sys
import types

import pytest

from omlx import engine, registry
from omlx.engine import ModelManager, SamplingParams


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "PROMPT:" + messages[-1]["content"]


@pytest.fixture
def fake_mlx(monkeypatch):
    """Inject fake mlx_lm / mlx.core modules so the engine loads without Metal.

    `get_active_memory()` is a monotonic counter incremented on each `load`,
    so size-backfill tests can drive deterministic deltas without Metal.
    """
    calls = {"load": 0, "active_mem": 0, "seed": -1}  # seed -1 = "not applied"

    def load(source):
        calls["load"] += 1
        # Simulate weights landing in Metal memory: each load adds 512 MiB.
        calls["active_mem"] += 512 * 1024 * 1024
        return object(), FakeTokenizer()

    def stream_generate(model, tokenizer, prompt, max_tokens, sampler):
        yield types.SimpleNamespace(
            text="a", finish_reason=None, prompt_tokens=3, generation_tokens=1
        )
        yield types.SimpleNamespace(
            text="b", finish_reason="stop", prompt_tokens=3, generation_tokens=2
        )

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = load
    mlx_lm.stream_generate = stream_generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **k: object()
    mx = types.ModuleType("mlx.core")
    mx.clear_cache = lambda: None
    mx.get_active_memory = lambda: calls["active_mem"]
    mx.random = types.SimpleNamespace(seed=lambda s: calls.__setitem__("seed", s))
    mlx = types.ModuleType("mlx")
    mlx.core = mx

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return calls


def test_resolve_uses_registered_repo_id(make_entry):
    registry.add(make_entry(name="Llama", repo_id="org/Llama", path="/p"))
    mgr = ModelManager(start_reaper=False)
    assert mgr._resolve("Llama") == "org/Llama"


def test_resolve_auto_pulls_unknown(monkeypatch, make_entry):
    pulled = make_entry(name="new", repo_id="org/new")
    monkeypatch.setattr("omlx.pull.pull", lambda name: pulled)
    mgr = ModelManager(start_reaper=False)
    assert mgr._resolve("org/new") == "org/new"


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
    assert out[-1].prompt_tokens == 3
    assert out[-1].completion_tokens == 2


def test_stream_text_yields_completions(fake_mlx, make_entry):
    registry.add(make_entry(name="A", repo_id="org/A", path="/p"))
    mgr = ModelManager(start_reaper=False)
    out = list(mgr.stream_text("A", "hello", SamplingParams(max_tokens=8)))
    assert "".join(c.text for c in out) == "ab"
    assert out[-1].finish_reason == "stop"


def _char_stream(text, finish="stop"):
    """A stream_generate stand-in that emits `text` one character per step."""

    def gen(model, tokenizer, prompt, max_tokens, sampler):
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

    def gen(model, tokenizer, prompt, max_tokens, sampler):
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


class FakeToolTokenizer:
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

    def truncated(model, tokenizer, prompt, max_tokens, sampler):
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
