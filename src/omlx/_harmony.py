"""Shared OpenAI Harmony helpers (gpt-oss and friends).

The control-token set, channel detection helpers, and tokenizer probe are
shared between :mod:`omlx.engine` (which runs the streaming parser) and
:mod:`omlx._json` (which masks logits during JSON-mode generation — for
Harmony models the mask applies only inside the ``final`` channel).
"""

from __future__ import annotations

import re
from typing import Any

# OpenAI Harmony control tokens (gpt-oss and friends). Stripped from output.
HARMONY_CONTROL: tuple[str, ...] = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)

# Channels whose body is chain-of-thought (routed to `reasoning`); anything
# else (`final`, or an unlabeled body) is the answer (routed to `text`).
REASONING_CHANNELS: tuple[str, ...] = ("analysis", "commentary")

# The Harmony body-closing tokens that end a channel's content span. After one
# of these the parser is back in a role/header state until the next
# ``<|channel|>`` opens a new channel.
HARMONY_CLOSE: frozenset[str] = frozenset({"<|end|>", "<|return|>", "<|call|>"})

# A Harmony tool call rides the commentary channel with a `to=functions.NAME`
# recipient, e.g. `<|channel|>commentary to=functions.get_weather<|message|>`.
HARMONY_TOOL_RE = re.compile(r"to=functions\.([\w.-]+)")


def harmony_tool_name(channel: str) -> str | None:
    """The function name if `channel` is a `to=functions.NAME` tool header, else None."""
    m = HARMONY_TOOL_RE.search(channel)
    return m.group(1) if m else None


def is_harmony(tok: Any) -> bool:
    """True when the tokenizer speaks Harmony (gpt-oss), by probing its vocab.

    gpt-oss reports no `has_tool_calling`, so this is the pre-generation signal
    for whether to offer tools to the chat template.
    """
    try:
        return "<|call|>" in tok.get_vocab()
    except Exception:
        return False


def harmony_channel_at_end(text: str) -> str | None:
    """Name of the active Harmony channel at the end of `text`, or ``None``.

    Scans ``text`` left-to-right for Harmony control tokens and tracks the
    running channel: the last opened ``<|channel|>NAME<|message|>`` whose body
    hasn't been closed by ``<|end|>`` / ``<|return|>`` / ``<|call|>`` is the
    active one. Returns the bare channel name (e.g. ``"final"``,
    ``"analysis"``, ``"commentary to=functions.foo"``). Returns an empty
    string when we're inside a ``<|channel|>`` header (between the channel
    open and ``<|message|>``) but no channel name has been fully received,
    which lets callers distinguish "header-in-progress" (don't mask) from
    "no Harmony at all" (mask per the caller's knowledge of the model).

    The scan is O(text length) and never returns to earlier matches; Harmony's
    grammar is regular enough that no nested channel spans exist.
    """
    i = 0
    state = "text"  # one of: "text", "channel", "role"
    channel = ""
    in_body = False  # True once a `<|message|>` opened the body
    while i < len(text):
        # Find the next control token starting at or after `i`.
        j = text.find("<", i)
        if j == -1:
            # Tail with no more control tokens. If we're mid-channel name
            # collection, the channel header is still open — capture the tail
            # as the partial channel name.
            if state == "channel":
                channel += text[i:]
            break
        # If we're accumulating a channel name between `<|channel|>` and
        # `<|message|>`, capture the body chars up to this control token.
        if state == "channel":
            channel += text[i:j]
        # Try each control token at j; the set is small so linear is fine.
        matched = None
        for tok in HARMONY_CONTROL:
            if text.startswith(tok, j):
                matched = tok
                break
        if matched is None:
            # A literal '<' that isn't the start of a control token: skip it.
            # (We're past the channel-name accumulation already handled above.)
            i = j + 1
            continue
        # Apply the transition just like `_HarmonyParser._transition` does.
        if matched == "<|channel|>":
            state = "channel"
            channel = ""
            in_body = False
        elif matched == "<|message|>":
            state = "text"
            in_body = True
        elif matched in ("<|start|>", "<|constrain|>"):
            state = "role"
            in_body = False
        else:  # <|end|>, <|return|>, <|call|>
            in_body = False
            state = "text"
            channel = ""
        i = j + len(matched)
    if not in_body:
        # Header between `<|channel|>` and `<|message|>` → return "" so the
        # JSON mask knows we're mid-header. `None` means "no Harmony seen".
        return "" if state == "channel" else None
    return channel
