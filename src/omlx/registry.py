"""Local model registry: friendly name -> repo/path/metadata.

Weights live in the HF hub cache (or ~/.omlx/converted for on-device
conversions); this registry is just a thin index over what has been pulled.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ensure_dirs, settings

logger = logging.getLogger("omlx")

Row = dict[str, Any]


@dataclass
class ModelEntry:
    name: str  # Ollama-style "base:paramtag", e.g. "gpt-oss:21b"
    repo_id: str  # HF repo id or local path used to load
    path: str  # resolved local snapshot / converted dir
    quant: str | None = None  # e.g. "4bit", "8bit", or None
    mlx_ready: bool = True  # True if used directly (no conversion)
    size_bytes: int = 0


def _load() -> dict[str, Row]:
    if not settings.registry_path.exists():
        return {}
    try:
        return json.loads(settings.registry_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("ignoring unreadable registry at %s: %s", settings.registry_path, e)
        return {}


def _save(data: dict[str, Row]) -> None:
    ensure_dirs()
    # Write-and-rename so a crash mid-write can't corrupt the registry.
    tmp = settings.registry_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(settings.registry_path)


# Trailing name tokens that a param tag or the `quant` field records instead:
# size (20b, 3b), quant bits (4bit), float/int precision, and a bare mlx tag.
_TAG_SUFFIX = re.compile(r"^(?:\d+(?:\.\d+)?[bm]|\d+bit|(?:bf|fp|f)16|int[48]|mxfp4|mlx)$")


def name_for(repo_id: str) -> str:
    """Derive a base name from a repo id: drop the org prefix, lowercase, and
    strip trailing size/quant/precision tokens. The param size becomes the
    ``:tag`` built in pull; the quant is kept in ``ModelEntry.quant``.
    """
    parts = repo_id.split("/")[-1].lower().split("-")
    while len(parts) > 1 and _TAG_SUFFIX.match(parts[-1]):
        parts.pop()
    return "-".join(parts)


def _resolve_key(data: dict[str, Row], name: str) -> str | None:
    """Resolve ``name`` to its registry key, or ``None`` if not registered.

    Lookup precedence (first hit wins), so a stored friendly name is never
    shadowed by a coincidentally-matching repo id:

    1. Exact key match: names are stored under their friendly name, so this
       handles ``get("Llama-…")`` directly.
    2. Friendly-tail match: drops the org prefix, so ``get("Llama-…")`` also
       resolves when the caller passes the full ``org/Llama-…`` id.
    3. Tagless base match: an untagged base resolves to its sole tagged entry,
       so ``get("gpt-oss")`` finds ``gpt-oss:21b`` (Ollama ``:latest`` style).
    4. Full repo_id scan: finds an entry whose recorded ``repo_id`` equals
       ``name``, so ``get("org/Llama")`` works even when the friendly name
       differs from the repo's last path segment.
    """
    if name in data:
        return name
    friendly = name_for(name)
    if friendly in data:
        return friendly
    for base in dict.fromkeys((name, friendly)):
        if ":" in base:
            continue
        matches = [k for k in data if k.startswith(base + ":")]
        if len(matches) == 1:
            return matches[0]
    for key, row in data.items():
        if row.get("repo_id") == name:
            return key
    return None


def add(entry: ModelEntry) -> None:
    data = _load()
    data[entry.name] = asdict(entry)
    _save(data)


def get(name: str) -> ModelEntry | None:
    data = _load()
    key = _resolve_key(data, name)
    return ModelEntry(**data[key]) if key else None


def entries() -> list[ModelEntry]:
    return [ModelEntry(**row) for row in _load().values()]


def remove(name: str, purge: bool = True) -> ModelEntry | None:
    """Drop from the registry. Purges the HF cache snapshot unless ``purge`` is False.
    Returns the entry, or None if not registered.
    """
    data = _load()
    key = _resolve_key(data, name)
    if key is None:
        return None
    entry = ModelEntry(**data.pop(key))
    _save(data)
    if purge:
        _purge_weights(entry)
    return entry


def _purge_weights(entry: ModelEntry) -> None:
    """Delete an entry's on-disk weights.

    On-device conversions live in ``~/.omlx/converted/<name>`` (a local dir, not
    an HF repo) and are removed outright; everything else is a snapshot in the
    shared HF hub cache, purged by repo id.
    """
    path = Path(entry.path)
    if path.is_relative_to(settings.converted_dir):
        shutil.rmtree(path, ignore_errors=True)
        return
    _purge_cache(entry.repo_id)


def _purge_cache(repo_id: str) -> None:
    """Delete the model's snapshot(s) from the HF hub cache."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return
    try:
        info = scan_cache_dir()
    except (OSError, ValueError) as e:
        # OSError: cache dir missing/unreadable; ValueError: corrupt metadata.
        logger.warning("could not scan HF cache to purge %s: %s", repo_id, e)
        return
    hashes: list[str] = []
    for repo in info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "model":
            hashes.extend(rev.commit_hash for rev in repo.revisions)
    if hashes:
        info.delete_revisions(*hashes).execute()
