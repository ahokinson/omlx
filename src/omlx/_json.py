"""Logits processor that constrains generation to a valid JSON top-level value.

Returned by :func:`json_object_processor`; plugs into mlx-lm's
``logits_processors`` list in :func:`omlx.engine._generation_kwargs` when the
request opts into ``response_format={"type": "json_object"}`` (OpenAI) or
``format="json"`` (Ollama). The processor masks the next-token logits so the
emitted text — *for the ``final`` channel only on Harmony/gpt-oss models* —
stays a prefix of a valid JSON value (object, array, string, number, ``true``,
``false``, ``null``).

The mask checks each vocab token's **first decoded character** against the
set of legal next characters given the current DFA state. Multi-character
tokens may therefore contribute tail characters that violate the grammar;
the next-step DFA re-derives over the full decoded text and re-masks the
following token, so violations are caught at most one step late. On a
non-recoverable state (an invalid char in a non-permissive position), the
mask clamps all logits except EOS so the model stops cleanly rather than
emitting malformed JSON.

For Harmony models (:func:`omlx._harmony.is_harmony`) the processor is
channel-aware: it's a no-op while the model is reasoning in the ``analysis``
channel or emitting a ``commentary`` tool call, and applies only inside the
``final`` (answer) channel — where the model writes the JSON answer.

Schema-typed JSON output (``response_format.type == "json_schema"`` and
Ollama ``format`` given as a dict) is **not supported** in this round; the
HTTP layer rejects those requests with ``400 unsupported`` (see
``server._require_response_format``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

from ._harmony import HARMONY_CONTROL

# JSON value-start characters accepted at any "expect-a-value" position.
_VALUE_START: frozenset[str] = frozenset('"{[0123456789-tfn')
_NUMBER_CHARS: frozenset[str] = frozenset("0123456789.eE+-")
# Inside a JSON string any printable plus space/tab/newline is allowed; these
# are the *byte* chars (Python strings are unicode — a single `c` per char),
# so the set covers ASCII 0x20..0x7e plus \t\n\r (whitespace inside strings
# is allowed by the JSON spec).
_IN_STRING: frozenset[str] = (
    frozenset(chr(c) for c in range(0x20, 0x7F)) | frozenset("\t\n\r") | frozenset('"\\')
)
# Valid second-character set after a JSON string escape backslash.
_IN_ESCAPE: frozenset[str] = frozenset('"\\/bfnrtu')
# Whitespace tolerated between JSON tokens (structural separators), per spec.
_WS: frozenset[str] = frozenset(" \t\n\r")

# A DFA over JSON prefixes. State names mirror the parser's expectation at the
# point just BEFORE the next character is consumed. Transitions are a
# `(state, char) -> state` mapping; we elide illegal chars by leaving them out
# (the mask forbids any char outside the allowed set for the current state).
_ALLOWED: dict[str, frozenset[str]] = {
    "expect_value": _VALUE_START | _WS,
    # A complete top-level value has been emitted; only whitespace (or stream
    # end) is tolerated for the rest of the input. A new value start here is
    # a JSON syntax error (multiple top-level values aren't permitted).
    "top_done": _WS,
    "obj_open": frozenset('"}') | _WS,  # `{`: expect key or `}`
    "obj_key": frozenset('"') | _WS,  # after `,`: expect a key
    "obj_after_key": frozenset(":") | _WS,  # after `"key"`: expect colon
    "obj_after_colon": _VALUE_START | _WS,  # after `:`: expect value
    "obj_after_value": frozenset(",}") | _WS,  # after value: separator or close
    "obj_after_comma": frozenset('"') | _WS,  # after `,`: expect a key
    "arr_open": _VALUE_START | frozenset("]") | _WS,  # `[`: expect value or `]`
    "arr_after_value": frozenset(",]") | _WS,  # after value: separator or close
    "arr_after_comma": _VALUE_START | _WS,  # after `,`: expect value
    "in_string_key": _IN_STRING,  # inside an object-key string
    "in_string_value": _IN_STRING,  # inside a value-side string
    "in_string_escape_key": _IN_ESCAPE,  # backslash inside a key string
    "in_string_escape_value": _IN_ESCAPE,  # backslash inside a value string
    # The four literal names advance one allowed char at a time.
    "lit_t": frozenset("r"),
    "lit_tr": frozenset("u"),
    "lit_tru": frozenset("e"),
    "lit_f": frozenset("a"),
    "lit_fa": frozenset("l"),
    "lit_fal": frozenset("s"),
    "lit_fals": frozenset("e"),
    "lit_n": frozenset("u"),
    "lit_nu": frozenset("l"),
    "lit_nul": frozenset("l"),
    # Numbers: first char (digit or `-`) sets us on the number track; any of
    # the whole number alphabet is then accepted (the relaxed grammar trades
    # strict JSON int/frac/exp shape for cheap masking — the model rarely
    # emits `1.+` once digits are bounded by `[0-9.eE+-]`). Structural
    # separators that end the number are accepted via the in_number branch
    # itself (not the early `_ALLOWED` check).
    "in_number": _NUMBER_CHARS,
}


def _advance_state(*, state: str, stack: list[str], ch: str) -> tuple[str, list[str]] | None:
    """One DFA transition. Returns the new (state, stack) or None on a failure.

    Stack entries are container opens (`{` or `[`); empty stack means we're at
    the top level of the JSON value. The DFA never emits errors itself — a
    failed transition is what the mask tries to prevent by ruling out the
    char before the model samples it.

    The early ``ch not in allowed`` rejection at the top means the per-state
    ``return None`` lines tagged ``# pragma: no cover`` below are truly
    unreachable — they're kept as defensive documentation of the state's
    rejection set. The lit_chain inner ``return None`` is reachable through
    the early check too.
    """
    allowed = _ALLOWED.get(state)
    if allowed is None:  # pragma: no cover - unknown state is unreachable here
        return None
    # `in_number` accepts the number alphabet but also has container-driven
    # structural separators (`,`, `}`, `]`, whitespace) once the number ends —
    # those depend on the stack and aren't in `_ALLOWED['in_number']`, so skip
    # the early rejection for that state and let the in_number branch decide.
    if state != "in_number" and ch not in allowed:
        return None
    if state == "expect_value":
        if ch in _WS:
            return state, stack
        if ch == '"':
            return "in_string_value", stack
        if ch in _VALUE_START - frozenset('"'):
            return _advance_value_start(ch, stack)
        return None  # pragma: no cover - unreachable given the allowed set

    if state == "top_done":
        # A complete top-level value has been emitted. Only whitespace (already
        # checked against `_ALLOWED['top_done']`) is tolerated; the only other
        # possibility is a new value start, which falls through to the early
        # `ch not in allowed` check above.
        if ch in _WS:
            return state, stack
        return None  # pragma: no cover - multiple top-level values not permitted

    if state == "obj_open":
        if ch in _WS:
            return state, stack
        if ch == "}":
            if not stack:  # pragma: no cover - top-level close handled elsewhere
                return None
            return _after_value(stack[:-1], stack[-1])
        if ch == '"':
            return "in_string_key", stack
        return None  # pragma: no cover - early rejection covers this

    if state == "obj_key":  # same as obj_open minus the `}` close
        if ch in _WS:
            return state, stack
        if ch == '"':
            return "in_string_key", stack
        return None  # pragma: no cover - early rejection covers this

    if state == "obj_after_key":
        if ch in _WS:
            return state, stack
        if ch == ":":
            return "obj_after_colon", stack
        return None  # pragma: no cover - early rejection covers this

    if state == "obj_after_colon":
        if ch in _WS:
            return state, stack
        if ch in _VALUE_START - frozenset('"'):
            return _advance_value_start(ch, stack)
        if ch == '"':
            return "in_string_value", stack
        return None  # pragma: no cover - early rejection covers this

    if state == "obj_after_value":
        if ch in _WS:
            return state, stack
        if ch == ",":
            return "obj_after_comma", stack
        if ch == "}":
            if not stack:  # pragma: no cover - top-level close handled elsewhere
                return None
            return _after_value(stack[:-1], stack[-1])
        return None  # pragma: no cover - early rejection covers this

    if state == "obj_after_comma":
        if ch in _WS:
            return state, stack
        if ch == '"':
            return "in_string_key", stack
        return None  # pragma: no cover - early rejection covers this

    if state == "arr_open":
        if ch in _WS:
            return state, stack
        if ch == "]":
            if not stack:  # pragma: no cover - top-level close handled elsewhere
                return None
            return _after_value(stack[:-1], stack[-1])
        if ch in _VALUE_START - frozenset('"'):
            return _advance_value_start(ch, stack)
        if ch == '"':
            return "in_string_value", stack
        return None  # pragma: no cover - early rejection covers this

    if state == "arr_after_value":
        if ch in _WS:
            return state, stack
        if ch == ",":
            return "arr_after_comma", stack
        if ch == "]":
            if not stack:  # pragma: no cover - top-level close handled elsewhere
                return None
            return _after_value(stack[:-1], stack[-1])
        return None  # pragma: no cover - early rejection covers this

    if state == "arr_after_comma":
        if ch in _WS:
            return state, stack
        if ch in _VALUE_START - frozenset('"'):
            return _advance_value_start(ch, stack)
        if ch == '"':
            return "in_string_value", stack
        return None  # pragma: no cover - early rejection covers this

    def _close_string_key() -> tuple[str, list[str]]:
        # A closed object-key string yields back to "expecting the colon".
        return "obj_after_key", stack

    def _close_string_value() -> tuple[str, list[str]] | None:
        # A closed value string yields to the enclosing container's after-value
        # state (or the top-level "done" state when stack is empty).
        if not stack:
            return "top_done", stack
        outer = stack[-1]
        return _after_value_inner(outer, stack)

    if state == "in_string_key":
        if ch == "\\":
            return "in_string_escape_key", stack
        if ch == '"':
            return _close_string_key()
        return "in_string_key", stack

    if state == "in_string_value":
        if ch == "\\":
            return "in_string_escape_value", stack
        if ch == '"':
            return _close_string_value()
        return "in_string_value", stack

    if state == "in_string_escape_key":
        if ch not in _IN_ESCAPE:
            return None
        return "in_string_key", stack

    if state == "in_string_escape_value":
        if ch not in _IN_ESCAPE:
            return None
        return "in_string_value", stack

    if state == "in_number":
        if ch in _NUMBER_CHARS:
            return "in_number", stack
        # A non-number char terminates the number; re-feed it as a structural
        # separator into the enclosing container's after-value state.
        if not stack:
            if ch in _WS:
                return "top_done", stack  # top-level number done; ws-only after
            return None
        return _after_value_char(stack, ch)

    # Literal-name states advance one char apiece.
    lit_chain = {
        "lit_t": ("r", "lit_tr"),
        "lit_tr": ("u", "lit_tru"),
        "lit_tru": ("e", "LITERAL_DONE"),
        "lit_f": ("a", "lit_fa"),
        "lit_fa": ("l", "lit_fal"),
        "lit_fal": ("s", "lit_fals"),
        "lit_fals": ("e", "LITERAL_DONE"),
        "lit_n": ("u", "lit_nu"),
        "lit_nu": ("l", "lit_nul"),
        "lit_nul": ("l", "LITERAL_DONE"),
    }
    if state in lit_chain:
        want, next_state = lit_chain[state]
        if ch != want:
            return None  # pragma: no cover - early rejection covers this
        if next_state == "LITERAL_DONE":
            if not stack:
                return "top_done", stack  # top-level literal done
            outer = stack[-1]
            return _after_value_inner(outer, stack)
        return next_state, stack

    return None  # pragma: no cover - unknown state handled by early rejection


def _advance_value_start(ch: str, stack: list[str]) -> tuple[str, list[str]] | None:
    """Open the sub-state for a container / number / literal / value string
    where the enclosing container expects a value."""
    if ch == "{":
        return "obj_open", stack + ["{"]
    if ch == "[":
        return "arr_open", stack + ["["]
    if ch == '"':
        return "in_string_value", stack
    if ch == "-":
        return "in_number", stack
    if ch.isdigit():
        return "in_number", stack
    if ch == "t":
        return "lit_t", stack
    if ch == "f":
        return "lit_f", stack
    if ch == "n":
        return "lit_n", stack
    return None  # pragma: no cover - early rejection covers this


def _after_value(stack: list[str], container: str) -> tuple[str, list[str]]:
    """After a value (or close) returns control to the surrounding container."""
    if not stack:
        return "top_done", stack  # top-level value done: only ws (or EOF) after
    outer = stack[-1]
    if outer == "{":
        return "obj_after_value", stack
    return "arr_after_value", stack


def _after_value_inner(container: str, stack: list[str]) -> tuple[str, list[str]]:
    """After a literal/number/closed string value inside the named container,
    choose the outer container's "after value" state."""
    if not stack:
        return "top_done", stack  # top-level literal/number closed
    if container == "{":
        return "obj_after_value", stack
    return "arr_after_value", stack


def _after_value_char(stack: list[str], ch: str) -> tuple[str, list[str]] | None:
    """Re-feed `ch` after a number's tail terminator re-enters the enclosing
    container's after-value state, then advances it."""
    if not stack:  # pragma: no cover - top-level in_number handles ws separately
        return None
    outer = stack[-1]
    state = "obj_after_value" if outer == "{" else "arr_after_value"
    return _advance_state(state=state, stack=stack, ch=ch)


def _allowed_next(state: str, stack: list[str]) -> frozenset[str]:
    """Allowed next-char set for the current DFA position.

    Most states have a constant allowed set (looked up from `_ALLOWED`);
    numbers are special-cased: a top-level ``in_number`` ends the JSON value
    when the stack is empty (only whitespace is parseable as "nothing more" —
    i.e. the model should stop). Failing-closed: if the DFA is in an unknown
    state, no char is allowed (the model is forced to stop).
    """
    if state == "in_number" and not stack:
        return _WS
    if state == "top_done":
        return _WS
    return _ALLOWED.get(state, frozenset())


def _scan(text: str, *, state: str, stack: list[str]) -> tuple[str, list[str], bool]:
    """Drive the DFA over `text`. Returns final ``state, stack, ok`` (ok is
    False on a transition failure, in which case `state` is left where it
    failed)."""
    for ch in text:
        nxt = _advance_state(state=state, stack=stack, ch=ch)
        if nxt is None:
            return state, stack, False
        state, stack = nxt
    return state, stack, True


@cache
def _vocab_prefix_index(
    tokenizer_id: int,
    make_pairs: Callable[[], list[tuple[str, int]]],
) -> tuple[dict[str, list[int]], list[int]]:
    """One-time prefix index over a tokenizer's vocab: ``{first_char: [token_ids]}``.

    ``make_pairs`` is a closure that returns ``(decoded_string, token_id)`` for
    every token in the vocab; we accept a callable (rather than the tokenizer
    itself) so the cache key stays hashable (``id(tokenizer)``) and the work
    runs only on a miss. The returned pair additionally includes every
    token id that decodes to an empty string under ``EMPTY_KEY`` ("") so a
    caller can decide what to do with empty-decode tokens (currently: keep them
    allowed).
    """
    pairs = make_pairs()
    by_first: dict[str, list[int]] = {}
    empties: list[int] = []
    for text, tid in pairs:
        if not text:
            empties.append(tid)
            continue
        by_first.setdefault(text[0], []).append(tid)
    return by_first, empties


def _decode_token_set(tokenizer: Any) -> list[tuple[str, int]]:
    """``(decoded_text, token_id)`` pairs for a tokenizer's full vocab.

    Uses ``tokenizer.id_to_token`` / ``convert_tokens_to_string`` when present
    (mlx-lm's wrapping of HF tokenizers exposes these) and falls back to a
    per-id ``tokenizer.decode([id])`` call. The latter is slower but works on
    both real mlx-lm tokenizers and the toy fakes used in tests.
    """
    pairs: list[tuple[str, int]] = []
    vocab: dict[str, int] = {}
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}
    if vocab:
        for tok_text, tid in vocab.items():
            if hasattr(tokenizer, "convert_tokens_to_string"):
                decoded = tokenizer.convert_tokens_to_string([tok_text])
            else:
                decoded = tok_text
            pairs.append((decoded, tid))
        return pairs
    # Fallback: probe ids 0..N until decode stops yielding new tokens; used by
    # toy tokenizers in tests where there's no vocab map.
    for tid in range(0, getattr(tokenizer, "vocab_size", 1024)):
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            continue
        pairs.append((decoded, tid))
    return pairs


def _final_body_start(text: str) -> int | None:
    """Index right after the ``<|message|>`` that opened the active ``final`` body.

    Returns None when no ``final`` body is currently open at the end of `text`.
    Scans left-to-right over Harmony control tokens and tracks the most recent
    ``<|channel|>final<|message|>`` opening whose body hasn't been closed by a
    ``<|end|>`` / ``<|return|>`` / ``<|call|>``. Reopening ``final`` after a
    close resets the index forward.
    """
    i = 0
    state = "text"  # one of: "text", "channel", "role"
    channel = ""
    in_body = False
    body_start: int | None = None
    final_active = False
    while i < len(text):
        j = text.find("<", i)
        if j == -1:
            break
        if state == "channel":
            channel += text[i:j]
        matched = None
        for tok in HARMONY_CONTROL:
            if text.startswith(tok, j):
                matched = tok
                break
        if matched is None:
            i = j + 1
            continue
        if matched == "<|channel|>":
            state = "channel"
            channel = ""
            in_body = False
        elif matched == "<|message|>":
            state = "text"
            in_body = True
            if channel.strip() == "final":
                final_active = True
                body_start = j + len(matched)
            else:
                final_active = False
        elif matched in ("<|start|>", "<|constrain|>"):
            state = "role"
            in_body = False
            final_active = False
        else:  # <|end|>, <|return|>, <|call|>
            in_body = False
            final_active = False
            state = "text"
            channel = ""
        i = j + len(matched)
    return body_start if (in_body and final_active) else None


def json_object_processor(tokenizer: Any, *, harmony: bool = False) -> Callable[[Any, Any], Any]:
    """A logits processor that constrains output to a valid JSON value.

    Parameters
    ----------
    tokenizer:
        An mlx-lm tokenizer (real production tokenizer or a test fake exposing
        ``decode`` and a vocab source — either ``get_vocab()`` or
        ``vocab_size``; see :func:`_decode_token_set` for the dispatch).
    harmony:
        When True, the processor applies the mask only inside the active
        ``final`` channel's body (re-derived over the body suffix each step);
        while the model is reasoning in ``analysis`` / ``commentary`` or
        mid-header, logits pass through untouched. This keeps control tokens
        outside the JSON state machine. When False (the non-Harmony default),
        the mask applies to the full text from the first token and the DFA
        state carries across calls (linear in body length).

    Returns a ``(tokens, logits) -> logits`` callable suitable for
    ``mlx_lm.stream_generate``'s ``logits_processors`` argument. The processor
    mutates ``logits`` in place via ``logits.at[:, forbids].add(-inf)``; the
    return value is the same ``logits`` array.
    """
    # Build (and cache by id) the per-tokenizer first-char -> [token_ids] index.
    by_first, _empties = _vocab_prefix_index(id(tokenizer), lambda: _decode_token_set(tokenizer))

    # Mutable per-generation DFA state captured in a dict so the closure sees
    # updates across processor calls. Carried across calls in the non-Harmony
    # path; re-derived from scratch on every call in the Harmony path.
    dfa: dict[str, Any] = {"state": "expect_value", "stack": [], "ok": True}

    def processor(tokens: Any, logits: Any) -> Any:
        try:
            ids = tokens.tolist()
        except AttributeError:
            ids = list(tokens)
        text = tokenizer.decode(ids)
        if not dfa["ok"]:
            return _force_stop(logits)
        if harmony:
            body_start = _final_body_start(text)
            if body_start is None:
                # Reasoning / commentary / channel header — pass through untouched.
                return logits
            # Re-derive the DFA over only the final-channel body so control
            # tokens outside this body never reach the JSON state machine.
            body = text[body_start:]
            state, stack, ok = _scan(body, state="expect_value", stack=[])
            # Drop the running DFA: we recompute from the body every step.
            dfa["state"] = state
            dfa["stack"] = stack
        else:
            state, stack, ok = _scan(text, state=dfa["state"], stack=list(dfa["stack"]))
        if not ok:
            dfa["ok"] = False
            return _force_stop(logits)
        dfa["state"] = state
        dfa["stack"] = stack
        allowed_chars = _allowed_next(state, stack)
        if not allowed_chars:
            return _force_stop(logits)
        forbids: list[int] = []
        for first_char, token_ids in by_first.items():
            if first_char in allowed_chars:
                continue
            forbids.extend(token_ids)
        if not forbids:
            return logits
        import mlx.core as mx  # ty: ignore[unresolved-import]

        idx = mx.array(forbids)
        return logits.at[:, idx].add(mx.array(-float("inf")))

    return processor


def _force_stop(logits: Any) -> Any:
    """Mask all but the EOS token so the sampling step ends the stream."""
    import mlx.core as mx  # ty: ignore[unresolved-import]

    shape = logits.shape
    vocab = shape[-1]
    # Allow only the very last token id as a stand-in EOS; mlx-lm tokenizers
    # conventionally place EOS at the end of the vocab, but the precise EOS id
    # varies by model. As a stopgap we forbid all but the last id and let the
    # engine's own finish check (`stream_generate`'s EOS set) end the stream.
    forbids = mx.array(list(range(0, vocab - 1)))
    return logits.at[:, forbids].add(mx.array(-float("inf")))
