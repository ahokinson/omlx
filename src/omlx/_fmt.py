"""Shared presentation helpers (byte sizing, etc.) used by the CLI and pull."""

from __future__ import annotations

_UNITS = ("B", "KB", "MB", "GB")


def human_size(n: int) -> str:
    """Format a byte count as a short human-readable string (e.g. ``"1.5GB"``).

    Saturates at TB so absurdly large values stay bounded rather than rolling
    into MiB units beyond the table.
    """
    x = float(n)
    for unit in _UNITS:
        if x < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}TB"


def human_params(total: int) -> str:
    """Format a parameter count as an Ollama-style tag (e.g. ``"21b"``, ``"700m"``)."""
    if total >= 1_000_000_000:
        return f"{round(total / 1e9)}b"
    if total >= 1_000_000:
        return f"{round(total / 1e6)}m"
    return str(total)
