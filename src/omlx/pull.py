"""Pull a model from Hugging Face (Xet-accelerated) and register it.

Auto-detects whether a repo is MLX-ready (quantized safetensors usable
directly by mlx-lm) or needs on-device conversion via mlx_lm.convert.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def _list_repo_files(repo_id: str, revision: str | None) -> list[str]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=False)
    return [s.rfilename for s in (info.siblings or [])]


def pull(
    repo_id: str,
    revision: str | None = None,
    convert: bool = False,
    bits: int = 4,
) -> ModelEntry:
    """Download `repo_id` and add it to the registry. Returns the entry."""
    from huggingface_hub import snapshot_download

    ensure_dirs()
    files = _list_repo_files(repo_id, revision)
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_config = "config.json" in files

    if not has_config:
        raise ValueError(f"{repo_id!r} has no config.json; not a loadable model repo")

    # hf_xet (installed via huggingface_hub[hf_xet]) makes this Xet-accelerated
    # automatically for Xet-backed repos; progress bars are built in.
    snapshot = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=[
                "*.safetensors",
                # PyTorch weights: needed for the convert path on repos that
                # ship no safetensors (mlx_lm.convert reads them from the snapshot).
                "*.bin",
                "*.json",
                "*.txt",
                "*.model",
                "tokenizer*",
                "*.py",
            ],
        )
    )

    if convert or not has_safetensors:
        entry = _convert(repo_id, snapshot, bits)
    else:
        entry = ModelEntry(
            name=name_for(repo_id),
            repo_id=repo_id,
            path=str(snapshot),
            quant=_detect_quant(snapshot / "config.json"),
            mlx_ready=True,
            size_bytes=_dir_size(snapshot),
        )

    add(entry)
    return entry


def _convert(repo_id: str, snapshot: Path, bits: int) -> ModelEntry:
    """Quantize a non-MLX repo on-device into ~/.omlx/converted/<name>."""
    from mlx_lm.convert import convert as mlx_convert

    name = name_for(repo_id) + f"-{bits}bit-mlx"
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
