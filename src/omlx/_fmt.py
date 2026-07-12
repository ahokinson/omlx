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
