"""Pull a model from Hugging Face (Xet-accelerated) and register it.

Auto-detects whether a repo is MLX-ready (quantized safetensors usable
directly by mlx-lm) or needs on-device conversion via mlx_lm.convert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._fmt import human_params
from .config import ensure_dirs, settings
from .registry import ModelEntry, add, name_for


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        # is_file() follows symlinks, so HF-cache blob links are counted once.
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _detect_quant(config_path: Path) -> str | None:
    """Return a quant label (e.g. "4bit") from config.json, or None.

    mlx-lm quantized repos carry a top-level "quantization" block; unquantized
    MLX/HF-format weights are still loadable but carry no label.
    """
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    quant = cfg.get("quantization")
    if isinstance(quant, dict) and "bits" in quant:
        return f"{quant['bits']}bit"
    return None


# Weight + tokenizer + config patterns fetched from a repo. *.bin covers repos
# that ship only PyTorch weights (read by mlx_lm.convert on the convert path).
_ALLOW_PATTERNS = ["*.safetensors", "*.bin", "*.json", "*.txt", "*.model", "tokenizer*", "*.py"]


def _cached_snapshot(repo_id: str, revision: str | None) -> Path | None:
    """Return the snapshot path if the repo is already fully in the HF cache, else None."""
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id,
                revision=revision,
                allow_patterns=_ALLOW_PATTERNS,
                local_files_only=True,
            )
        )
    except FileNotFoundError:
        # LocalEntryNotFoundError (a FileNotFoundError) when not fully cached.
        return None


def _repo_meta(repo_id: str, revision: str | None):
    """Return ``(filenames, safetensors)`` from the HF ``model_info``.

    ``safetensors`` is a ``SafeTensorsInfo`` (``.total`` is the param count) or
    ``None`` when the Hub hasn't computed it for the repo.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, expand=["siblings", "safetensors"])
    return [s.rfilename for s in (info.siblings or [])], info.safetensors


def _tagged_name(repo_id: str, safetensors: Any) -> str:
    """Build the Ollama-style ``base:paramtag`` name, or bare base if params are unknown."""
    base = name_for(repo_id)
    if safetensors is not None:
        return f"{base}:{human_params(safetensors.total)}"
    return base


def pull(
    repo_id: str,
    revision: str | None = None,
    convert: bool = False,
    bits: int = 4,
    name: str | None = None,
) -> ModelEntry:
    """Download `repo_id` and add it to the registry. Returns the entry."""
    from huggingface_hub import snapshot_download

    ensure_dirs()
    files, safetensors = _repo_meta(repo_id, revision)
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_config = "config.json" in files

    if not has_config:
        raise ValueError(f"{repo_id!r} has no config.json; not a loadable model repo")

    # Reuse the cached snapshot when the weights are already present; only hit the
    # network (hf_xet Xet-accelerated for Xet-backed repos) on a cache miss.
    snapshot = _cached_snapshot(repo_id, revision) or Path(
        snapshot_download(repo_id, revision=revision, allow_patterns=_ALLOW_PATTERNS)
    )

    if convert or not has_safetensors:
        entry = _convert(repo_id, snapshot, bits, name=name)
    else:
        entry = ModelEntry(
            name=name or _tagged_name(repo_id, safetensors),
            repo_id=repo_id,
            path=str(snapshot),
            quant=_detect_quant(snapshot / "config.json"),
            mlx_ready=True,
            size_bytes=_dir_size(snapshot),
        )

    add(entry)
    return entry


def _convert(repo_id: str, snapshot: Path, bits: int, name: str | None = None) -> ModelEntry:
    """Quantize a non-MLX repo on-device into ~/.omlx/converted/<name>."""
    from mlx_lm.convert import convert as mlx_convert

    name = name or (name_for(repo_id) + f"-{bits}bit-mlx")
    out = settings.converted_dir / name
    mlx_convert(
        str(snapshot),
        mlx_path=str(out),
        quantize=True,
        q_bits=bits,
    )
    return ModelEntry(
        name=name,
        repo_id=str(out),  # load directly from the converted local dir
        path=str(out),
        quant=f"{bits}bit",
        mlx_ready=True,
        size_bytes=_dir_size(out),
    )
