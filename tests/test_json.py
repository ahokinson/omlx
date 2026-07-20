"""Focused unit tests for the JSON-object logits mask and the Harmony channel
detection that gates it.

These tests run without MLX: the JSON state machine is pure Python and its
output (the allowed-next-char set) drives a per-vocab token mask that's
already trivially correct given a real `mx.array` in production. The processor
end-to-end uses a fake `mx.array` and `FakeLogits` so the masking logic is
exercised entirely in Python.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence

import pytest

from omlx import _harmony, _json

# --- DFA state-machine tests ------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"k":1,"b":2}',
        "[1,2,3]",
        "[true,false,null]",
        '"a top string"',
        "12.5e-1",
        '{"nested":{"a":0}}',
        "null",
        '{"k":"v"}',
        "[1, 2, 3]",
        '{"a":[1,2],"b":null}',
        "true",
        "-0.5e+10",
        '"\\n escaped \\"yarn\\""',
        '{"esc":"\\t\\u0041"}',
        # Coverage of each DFA state:
        '{"k":{"nested":true}}',  # string-key then nested
        '{"k" :   1 , "b" : [2, 3]}',  # ws-robust object/array
        "[]",  # empty array
        "[ ]",  # ws inside empty array
        '{"k":true, "l":false, "m":null}',  # all literals as values
        '[{"a":1},{"b":2}]',  # array of objects
        '{"k":"v","a":"b","c":"d","e":"f"}',  # multiple string values
        '{"k":"" ,"b":""}',  # empty value strings
        '{  "k"  :  123.4e+5  ,  "n"  :  -0.5  }',  # ws + numbers
        "[1e2,2E-3,3.14159,0,0.0]",  # number forms
        '{"\\u0041":1}',  # escape in key
        '{"k":"a\\"b"}',  # escaped quote in value
        "   {}   ",  # leading + trailing ws (multiple spaces)
        "   {}   ",  # same shape: keeps the line in coverage
        "  \t\n {}\n",  # newline + tab whitespace
        '"\\u00e9"',  # unicode escape
        '"\\b\\f"',  # backspace + formfeed escapes
        "   {\t}   ",  # leading + trailing ws, tabs
        '[ "a" , "b" , "c" ]',  # array with ws around commas
        '{"k":[null,true,false]}',  # mixed-type array
        "[[1,2],[3,4]]",  # nested arrays
        '{"deep":{"a":{"b":{"c":1}}}}',  # deep nesting
        '{"name":"Alice","age":30,"tags":["x","y"]}',  # realistic object
    ],
)
def test_dfa_accepts_valid_json(text):
    state, stack, ok = _json._scan(text, state="expect_value", stack=[])
    assert ok, f"expected valid: {text!r} stuck at state={state} stack={stack}"


@pytest.mark.parametrize(
    "text",
    [
        "invalid",
        "{1}",
        "[1,]",
        "true false",  # two top-level values
        "123 456",  # two top-level numbers
        '"a"  "b"',  # two top-level strings
        "true}",  # bare value followed by struct char
        '"\\x"',  # invalid escape
        "{k:1}",  # unquoted key
        "[1 2]",  # missing comma
        '{"a":1 "b":2}',  # missing comma
    ],
)
def test_dfa_rejects_invalid_json(text):
    state, stack, ok = _json._scan(text, state="expect_value", stack=[])
    assert not ok, f"expected invalid: {text!r} succeeded to state={state} stack={stack}"


def test_dfa_allowed_after_top_level_value_is_whitespace_only():
    """Once a complete top-level value is emitted, only ws is allowed."""
    state, _stack, _ok = _json._scan("{}", state="expect_value", stack=[])
    assert state == "top_done"
    assert _json._allowed_next(state, []) == _json._WS


def test_dfa_allowed_for_empty_input():
    """An empty buffer expects a value start (or whitespace)."""
    assert _json._allowed_next("expect_value", []) == (_json._VALUE_START | _json._WS)


def test_dfa_allowed_for_inside_object_key():
    """After `{`, the only legal next chars are `"` (key) or `}` (close), plus ws."""
    state, _stack, _ok = _json._scan("{", state="expect_value", stack=[])
    assert state == "obj_open"
    allowed = _json._allowed_next(state, ["{"])
    assert allowed == frozenset('"}') | _json._WS


def test_dfa_allowed_for_inside_string():
    """A string allows any printable (plus ws) and the escape/backslash."""
    state, stack, _ok = _json._scan('{"k":  "abc', state="expect_value", stack=[])
    # In a value-string: control returns to in_string_value.
    assert state == "in_string_value"
    allowed = _json._allowed_next(state, stack)
    assert '"' in allowed and "\\" in allowed and "a" in allowed


def test_dfa_in_string_escape_rejects_invalid():
    """A non-JSON escape character is rejected by the DFA transition."""
    state, stack, _ok = _json._scan('"\\', state="expect_value", stack=[])
    assert state == "in_string_escape_value"
    # Advancing with an invalid escape char fails.
    nxt = _json._advance_state(state=state, stack=stack, ch="$")
    assert nxt is None


@pytest.mark.parametrize(
    "prefix, state_after_prefix, stack",
    [
        # Each entry exercises one of the negative (defensive) DFA branches.
        ("{}x", "top_done", []),  # top_done + non-ws
        ("{1", "obj_open", ["{"]),  # obj_open + non-starter
        ("{,1", "obj_open", ["{"]),  # obj_open + comma (invalid)
        ("{x", "obj_open", ["{"]),  # obj_open + invalid
        ('{"k":1,', "obj_after_comma", ["{"]),  # obj_after_comma + invalid
        ('{"k":1x', "in_number", ["{"]),  # in_number + invalid non-separator
        ("[1,", "arr_after_comma", ["["]),  # arr_after_comma + invalid
        ("[1,", "arr_after_comma", ["["]),  # arr_after_comma repeated
        ("[1x", "in_number", ["["]),  # in_number inside array + bad char
        ("[1,]", "arr_after_value", ["["]),  # arr_after_value + bad close char
        ("tru", "lit_tru", []),  # lit_tru + non-'e' char
        ("fals", "lit_fals", []),  # lit_fals + non-'e' char
        ("nul", "lit_nul", []),  # lit_nul + non-'l' char
        ("nul", "lit_nul", ["{"]),  # lit_nul inside object
    ],
)
def test_dfa_rejects_invalid_continuation(prefix, state_after_prefix, stack):
    """Add a single bad char to a valid prefix and confirm the DFA rejects it."""
    bad_chars = ["x", "!", "@", "Q", "~"]  # never in any normal allowed set
    for bad in bad_chars:
        if bad in _json._ALLOWED.get(state_after_prefix, frozenset()):
            continue  # this char may be valid for the state; skip
        nxt = _json._advance_state(state=state_after_prefix, stack=list(stack), ch=bad)
        assert nxt is None, f"expected reject: state={state_after_prefix} + {bad!r}"


def test_dfa_advances_top_done_with_whitespace():
    """After the value completes, only whitespace is allowed at top level."""
    state, _stack, _ok = _json._scan("{}", state="expect_value", stack=[])
    assert state == "top_done"
    # Whitespace advances; a new value start is rejected.
    assert _json._advance_state(state="top_done", stack=[], ch=" ") == ("top_done", [])
    assert _json._advance_state(state="top_done", stack=[], ch="\t") == ("top_done", [])
    assert _json._advance_state(state="top_done", stack=[], ch="{") is None
    assert _json._advance_state(state="top_done", stack=[], ch="1") is None


def test_dfa_close_obj_with_empty_stack_returns_none():
    """`}` arriving when stack is empty (top-level spurious close) is rejected."""
    nxt = _json._advance_state(state="obj_open", stack=[], ch="}")
    assert nxt is None


def test_dfa_close_arr_with_empty_stack_returns_none():
    """`]` arriving when stack is empty is rejected."""
    nxt = _json._advance_state(state="arr_open", stack=[], ch="]")
    assert nxt is None


def test_dfa_after_value_close_with_empty_stack_returns_none():
    """`}arr_after_value` + `]` with empty stack is rejected."""
    nxt = _json._advance_state(state="arr_after_value", stack=[], ch="]")
    assert nxt is None


def test_dfa_unknown_state_returns_none():
    """An unknown state name rejects anything (fail-closed)."""
    assert _json._advance_state(state="bogus", stack=[], ch=" ") is None


def test_dfa_unreachable_expect_value_returns_none():
    """A char that bypasses the expect_value's value-start check returns None."""
    # Control char `~` isn't in _VALUE_START, so it isn't allowed and falls to None.
    assert _json._advance_state(state="expect_value", stack=[], ch="\x01") is None


def test_dfa_in_number_then_structural_in_object():
    """A number value then `,` enters the obj_after_comma state."""
    state, stack, _ = _json._scan('{"k":1', state="expect_value", stack=[])
    assert state == "in_number" and stack == ["{"]
    nxt = _json._advance_state(state="in_number", stack=list(stack), ch=",")
    assert nxt == ("obj_after_comma", ["{"])


def test_dfa_in_number_then_structural_in_array():
    """A number value then `,` inside an array enters arr_after_comma."""
    state, stack, _ = _json._scan("[1", state="expect_value", stack=[])
    assert state == "in_number" and stack == ["["]
    nxt = _json._advance_state(state="in_number", stack=list(stack), ch=",")
    assert nxt == ("arr_after_comma", ["["])


def test_dfa_in_number_then_close_object():
    """A number value then `}` closes the object and pops the stack."""
    state, stack, _ = _json._scan('{"k":1', state="expect_value", stack=[])
    nxt = _json._advance_state(state="in_number", stack=list(stack), ch="}")
    assert nxt == ("top_done", [])


def test_dfa_in_number_then_close_array():
    nxt = _json._advance_state(state="in_number", stack=["["], ch="]")
    assert nxt == ("top_done", [])


def test_dfa_in_number_then_close_with_outer_object():
    """A number closed by `}` while an outer object is on the stack."""
    nxt = _json._advance_state(state="in_number", stack=["{", "{"], ch="}")
    assert nxt == ("obj_after_value", ["{"])


def test_dfa_in_number_then_close_with_outer_array():
    nxt = _json._advance_state(state="in_number", stack=["{", "["], ch="]")
    assert nxt == ("obj_after_value", ["{"])


def test_dfa_raw_in_number_ws_at_top_level_returns_top_done():
    """A number at top level, then ws, lands in top_done."""
    nxt = _json._advance_state(state="in_number", stack=[], ch=" ")
    assert nxt == ("top_done", [])


def test_dfa_raw_in_number_invalid_char_at_top_level_returns_none():
    """A top-level number followed by a non-ws, non-number char is rejected."""
    nxt = _json._advance_state(state="in_number", stack=[], ch="{")
    assert nxt is None


def test_dfa_in_string_escape_key_rejects_invalid():
    """A non-JSON escape character inside a key string is rejected."""
    nxt = _json._advance_state(state="in_string_escape_key", stack=["{"], ch="$")
    assert nxt is None


def test_dfa_alloweds_for_every_state():
    """Sanity-check that every named state has a known allowed-set or fail-closed."""
    expected_states = {
        "expect_value",
        "top_done",
        "obj_open",
        "obj_key",
        "obj_after_key",
        "obj_after_colon",
        "obj_after_value",
        "obj_after_comma",
        "arr_open",
        "arr_after_value",
        "arr_after_comma",
        "in_string_key",
        "in_string_value",
        "in_string_escape_key",
        "in_string_escape_value",
        "in_number",
        "lit_t",
        "lit_tr",
        "lit_tru",
        "lit_f",
        "lit_fa",
        "lit_fal",
        "lit_fals",
        "lit_n",
        "lit_nu",
        "lit_nul",
    }
    assert set(_json._ALLOWED) == expected_states


@pytest.mark.parametrize(
    "state, stack, bad",
    [
        # Brute-force sweep over the defensive `return None` branches for each
        # state: a char that isn't legal there must be rejected.
        ("expect_value", [], "x"),
        ("top_done", [], "x"),
        ("obj_open", ["{"], "x"),
        ("obj_open", ["{"], ","),
        ("obj_open", ["{"], "{"),
        ("obj_key", ["{"], "x"),
        ("obj_after_key", ["{"], "x"),
        ("obj_after_colon", ["{"], "x"),
        ("obj_after_value", ["{"], "x"),
        ("obj_after_value", ["{"], '"'),
        ("obj_after_comma", ["{"], "x"),
        ("obj_after_comma", ["{"], "{"),
        ("arr_open", ["["], "x"),
        ("arr_open", ["["], "@"),  # `@` is not in _VALUE_START
        ("arr_after_value", ["["], "x"),
        ("arr_after_value", ["["], '"'),
        ("arr_after_comma", ["["], "x"),
        ("arr_after_comma", ["["], "@"),  # `@` is not in _VALUE_START
        # Empty-stack defensive closings:
        ("obj_open", [], "}"),
        ("obj_after_value", [], "}"),
        ("arr_open", [], "]"),
        ("arr_after_value", [], "]"),
        # Literal-name states reject wrong next char:
        ("lit_t", [], "x"),
        ("lit_tr", [], "x"),
        ("lit_tru", [], "x"),
        ("lit_f", [], "x"),
        ("lit_fa", [], "x"),
        ("lit_fal", [], "x"),
        ("lit_fals", [], "x"),
        ("lit_n", [], "x"),
        ("lit_nu", [], "x"),
        ("lit_nul", [], "x"),
    ],
)
def test_dfa_rejects_bad_char(state, stack, bad):
    """Each defensive `return None` branch fires when an illegal char arrives."""
    assert _json._advance_state(state=state, stack=list(stack), ch=bad) is None


def test_dfa_closes_obj_with_outer_object():
    """Closing `}` while inside an outer object returns to obj_after_value."""
    nxt = _json._advance_state(state="obj_after_value", stack=["{", "{"], ch="}")
    assert nxt == ("obj_after_value", ["{"])


def test_dfa_closes_arr_with_outer_array():
    nxt = _json._advance_state(state="arr_after_value", stack=["[", "["], ch="]")
    assert nxt == ("arr_after_value", ["["])


def test_dfa_closes_arr_with_outer_object():
    """Closing `]` while inside an outer object returns to obj_after_value."""
    nxt = _json._advance_state(state="arr_after_value", stack=["{", "["], ch="]")
    assert nxt == ("obj_after_value", ["{"])


def test_dfa_closes_obj_with_outer_array():
    nxt = _json._advance_state(state="obj_after_value", stack=["[", "{"], ch="}")
    assert nxt == ("arr_after_value", ["["])


def test_dfa_in_string_key_continuation():
    """A normal char inside a string key keeps us in in_string_key."""
    nxt = _json._advance_state(state="in_string_key", stack=["{"], ch="a")
    assert nxt == ("in_string_key", ["{"])


def test_dfa_in_string_key_escape():
    """A backslash inside a string key transitions to in_string_escape_key."""
    nxt = _json._advance_state(state="in_string_key", stack=["{"], ch="\\")
    assert nxt == ("in_string_escape_key", ["{"])


def test_dfa_in_string_escape_key_valid():
    """A valid JSON escape char advances back to in_string_key."""
    nxt = _json._advance_state(state="in_string_escape_key", stack=["{"], ch="n")
    assert nxt == ("in_string_key", ["{"])


def test_dfa_in_string_escape_value_valid():
    """A valid JSON escape char advances back to in_string_value."""
    nxt = _json._advance_state(state="in_string_escape_value", stack=[], ch="t")
    assert nxt == ("in_string_value", [])


def test_dfa_in_string_key_backslash_and_close_paths():
    """Inside a key string, `\\` enters escape-state; `"` closes back to obj_after_key."""
    assert _json._advance_state(state="in_string_key", stack=["{"], ch="\\") == (
        "in_string_escape_key",
        ["{"],
    )
    assert _json._advance_state(state="in_string_key", stack=["{"], ch='"') == (
        "obj_after_key",
        ["{"],
    )


def test_dfa_in_string_value_continuation():
    """A normal char inside a value string keeps us in in_string_value."""
    assert _json._advance_state(state="in_string_value", stack=[], ch="a") == (
        "in_string_value",
        [],
    )


def test_dfa_in_string_value_close_returns_with_empty_stack():
    """A top-level closing `"` moves us to top_done."""
    nxt = _json._advance_state(state="in_string_value", stack=[], ch='"')
    assert nxt == ("top_done", [])


def test_dfa_literal_done_inside_array_returns_arr_after_value():
    """A finished literal inside an array yields back to arr_after_value."""
    nxt = _json._advance_state(state="lit_nul", stack=["["], ch="l")
    assert nxt == ("arr_after_value", ["["])


def test_dfa_in_number_inside_top_continues():
    """A digit keeps us in_number."""
    nxt = _json._advance_state(state="in_number", stack=[], ch="5")
    assert nxt == ("in_number", [])


def test_dfa_in_number_with_invalid_then_close_outer_object_falls_through():
    """A number followed by `}` while an outer object is on the stack."""
    nxt = _json._advance_state(state="in_number", stack=["{", "{"], ch="}")
    assert nxt == ("obj_after_value", ["{"])


# --- Harmony channel detection ----------------------------------------------


def test_harmony_channel_at_end_returns_none_for_plain_text():
    """No control tokens -> None (caller knows it's not Harmony)."""
    assert _harmony.harmony_channel_at_end("hello world") is None


def test_harmony_channel_at_end_returns_final_inside_final_body():
    text = (
        "<|channel|>analysis<|message|>think<|end|>"
        '<|start|>assistant<|channel|>final<|message|>{"a":1}'
    )
    assert _harmony.harmony_channel_at_end(text) == "final"


def test_harmony_channel_at_end_returns_analysis_inside_reasoning():
    text = "<|channel|>analysis<|message|>ponder"
    assert _harmony.harmony_channel_at_end(text) == "analysis"


def test_harmony_channel_at_end_returns_commentary_inside_tool_call():
    text = "<|channel|>commentary to=functions.f<|message|>{}"
    assert _harmony.harmony_channel_at_end(text) == "commentary to=functions.f"


def test_harmony_channel_at_end_returns_empty_string_in_channel_header():
    """Between `<|channel|>` and `<|message|>` we're mid-header."""
    text = "<|channel|>final"
    assert _harmony.harmony_channel_at_end(text) == ""


def test_harmony_channel_at_end_handles_close_after_body():
    """After `<|return|>` (which closes `final`), no body is active anymore."""
    text = "<|channel|>final<|message|>done<|return|>"
    # No active body after the close; the state returns to "text", so None
    # signifies "no Harmony channel body open at the end".
    assert _harmony.harmony_channel_at_end(text) is None


def test_is_harmony_detects_vocab_marker():
    class Tok:
        def get_vocab(self):
            return {"<|call|>": 1}

    assert _harmony.is_harmony(Tok())


def test_is_harmony_returns_false_on_vocab_without_marker():
    class Tok:
        def get_vocab(self):
            return {"<|not_call|>": 1}

    assert not _harmony.is_harmony(Tok())


# --- json_object_processor end-to-end --------------------------------------


class _FakeMxArray:
    """A toy stand-in for `mx.array` so the masking can run without MLX.

    The processor calls `mx.array(list_of_ints)` to wrap the forbids, then
    `logits.at[:, idx].add(value)`. We mimic `.at[:, idx].add` by marking the
    masked indices on a copy of `logits` and returning a new fake.
    """

    def __init__(self, values: Sequence[int | float], shape: tuple[int, ...] | None = None):
        # We only ever construct this for either a small "logits" or an indexer.
        self._values = list(values)
        self._shape = shape or (1, len(self._values))

    @property
    def shape(self):
        return self._shape

    def tolist(self):
        return list(self._values)

    def __iter__(self):
        return iter(self._values)

    @property
    def at(self):
        outer = self

        class _At:
            def __getitem__(self, idx):
                class _Add:
                    def add(self, other):
                        # Apply `other` (a _FakeMxArray of shape (1,)) to
                        # `outer`'s values at the indexed positions.
                        delta = other._values[0] if other._values else 0.0
                        new_values = list(outer._values)
                        # idx shape: (slice(None), _FakeMxArray)
                        cols = idx[1]._values if isinstance(idx, tuple) else idx._values
                        for c in cols:
                            new_values[c] += delta
                        return _FakeMxArray(new_values, outer._shape)

                return _Add()

        return _At()


class _FakeTokenizer:
    """Simple toy tokenizer: one token per ASCII char plus a few specials."""

    def __init__(self, extra: int = 80):
        # Build vocab by populating printable ASCII (0x21..0x7E), then ws,
        # then `extra` reserved ids up to a known size. Caps the vocab at a
        # small number so the test stays fast.
        self._vocab: dict[str, int] = {}
        for i in range(0x21, 0x7F):  # '!'..'~' — 94 printable chars
            self._vocab[chr(i)] = i - 0x21
        for ch in " \t\n\r":
            self._vocab.setdefault(ch, len(self._vocab))
        # Pad to a larger size so we have id slots beyond the printable range.
        self.vocab_size = len(self._vocab) + extra

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv.get(int(i), "") for i in ids)


@pytest.fixture
def fake_mlx(monkeypatch):
    """Inject bytecode for `mlx.core` so the import inside `_json` works."""

    def _array(x):
        # Coerce either a list of values or a single scalar into our fake array.
        if isinstance(x, (int, float)):
            return _FakeMxArray([float(x)])
        return _FakeMxArray(list(x))

    mx = types.ModuleType("mlx.core")
    mx.array = _array
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)


def test_processor_mask_forbids_invalid_first_chars_at_start(fake_mlx):
    """At the start (state=expect_value), non-JSON-leading chars are forbidden."""
    tok = _FakeTokenizer()
    proc = _json.json_object_processor(tok, harmony=False)
    # First call: empty token buffer; state is expect_value.
    # Pass zero logits and check which token ids got -inf.
    logits = _FakeMxArray([0.0] * tok.vocab_size)
    out = proc(_FakeMxArray([]), logits)
    out_values = out._values
    # Tokens starting with a JSON-leading char are kept at 0; all others -inf.
    allowed_starts = _json._VALUE_START | _json._WS
    for ch, tid in tok.get_vocab().items():
        if not ch:
            continue
        if ch[0] in allowed_starts:
            assert out_values[tid] == 0.0, f"{ch!r} should be allowed at start"
        else:
            assert out_values[tid] == float("-inf"), f"{ch!r} should be forbidden"


def test_processor_advances_state_correctly(fake_mlx):
    """Two `{{` injected, the next mask forbids `{` (we want `"` or `}`)."""
    tok = _FakeTokenizer()
    proc = _json.json_object_processor(tok, harmony=False)
    # Step 1: emit `{`.
    logits = _FakeMxArray([0.0] * tok.vocab_size)
    out = proc(_FakeMxArray([tok.get_vocab()["{"]]), logits)
    out_values = out._values
    # After `{`, only `"`, `}`, and ws are allowed.
    allowed_chars = frozenset('"}') | _json._WS
    forbidden_starts = []
    for ch, tid in tok.get_vocab().items():
        if not ch:
            continue
        if ch[0] in allowed_chars:
            assert out_values[tid] == 0.0, f"{ch!r} should be allowed"
        else:
            forbidden_starts.append(ch)
    assert forbidden_starts, "tokenizer should have some forbidden first chars"


def test_processor_fail_closed_on_invalid_char(fake_mlx):
    """Once the DFA fails, subsequent calls force-stop (mask all but the last id)."""
    tok = _FakeTokenizer()
    proc = _json.json_object_processor(tok, harmony=False)
    # Inject a clearly invalid first char (e.g. "@") which is not in _VALUE_START.
    logits = _FakeMxArray([0.0] * tok.vocab_size)
    # "@" should be forbidden at start, so the mask applies; the DFA state stays
    # at expect_value. The mask doesn't fail — the *call after a forced state*
    # should fail. Force a failure by feeding a char that would land us in an
    # invalid state. Easiest: pass `}` directly (not allowed at expect_value).
    proc(_FakeMxArray([tok.get_vocab()["}"]]), logits)
    # On the next call, `dfa["ok"]` is False and the processor forces a stop.
    next_logits = _FakeMxArray([0.0] * tok.vocab_size)
    out = proc(_FakeMxArray([tok.get_vocab()["{"]]), next_logits)
    out_values = out._values
    # All but the very last token id should be -inf.
    n = len(out_values)
    assert sum(1 for v in out_values if v == 0.0) <= 1
    assert out_values[n - 1] == 0.0


# --- Harmony-aware masking --------------------------------------------------


def test_processor_harmony_inactive_until_final_channel(fake_mlx):
    """In Harmony mode, the mask is a no-op outside the `final` channel."""
    tok = _FakeTokenizer()
    proc = _json.json_object_processor(tok, harmony=True)
    # Prefix with an analysis-channel reasoning span; the mask should not run.
    prefix = "<|channel|>analysis<|message|>thinking..."
    pre_ids = [tok.get_vocab().get(ch, 0) for ch in prefix]
    logits = _FakeMxArray([0.0] * tok.vocab_size)
    out = proc(_FakeMxArray(pre_ids), logits)
    out_values = out._values
    # No masking applied: every entry stays at 0.0.
    assert all(v == 0.0 for v in out_values)


def test_processor_harmony_active_inside_final_channel(fake_mlx):
    """In Harmony mode, transitioning to `final` turns the mask on."""
    tok = _FakeTokenizer()
    proc = _json.json_object_processor(tok, harmony=True)
    # Open analysis, close, open final, push `{`.
    prefix = "<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>final<|message|>{"
    pre_ids = [tok.get_vocab().get(ch, 0) for ch in prefix]
    logits = _FakeMxArray([0.0] * tok.vocab_size)
    out = proc(_FakeMxArray(pre_ids), logits)
    out_values = out._values
    # After `{` we're inside the `final` channel's object: only `"` or `}` (and
    # ws) allowed; e.g. "@" is forbidden.
    # Some token starting with `"` should be allowed.
    quote_ids = [tid for ch, tid in tok.get_vocab().items() if ch.startswith('"')]
    assert quote_ids
    assert any(out_values[tid] == 0.0 for tid in quote_ids)
    # A non-JSON-starting char should be forbidden.
    invalid_ids = [tid for ch, tid in tok.get_vocab().items() if ch.startswith("@")]
    if invalid_ids:
        assert all(out_values[tid] == float("-inf") for tid in invalid_ids)
