from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import huggingface_hub
import pytest

from omlx import pull as pull_mod
from omlx import registry
from omlx.config import settings
from omlx.pull import _detect_quant, _dir_size, _list_repo_files
from omlx.registry import ModelEntry


def _snapshot(tmp_path, files: dict[str, bytes]):
    """Build a fake HF snapshot dir and stub the repo-listing + download."""
    snap = tmp_path / "snap"
    snap.mkdir()
    for name, data in files.items():
        (snap / name).write_bytes(data)
    return snap


def test_dir_size_sums_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 5)
    assert _dir_size(tmp_path) == 15


def test_dir_size_follows_symlink_blob(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"z" * 7)
    link_dir = tmp_path / "snap"
    link_dir.mkdir()
    (link_dir / "weight.safetensors").symlink_to(blob)
    assert _dir_size(link_dir) == 7


def test_detect_quant_reads_bits(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
    assert _detect_quant(cfg) == "4bit"


def test_detect_quant_none_when_unquantized(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"hidden_size": 2048}))
    assert _detect_quant(cfg) is None


def test_detect_quant_none_when_missing(tmp_path):
    assert _detect_quant(tmp_path / "nope.json") is None


def test_pull_registers_mlx_ready_repo(tmp_path, monkeypatch):
    snap = _snapshot(
        tmp_path,
        {
            "config.json": json.dumps({"quantization": {"bits": 4}}).encode(),
            "model.safetensors": b"x" * 100,
        },
    )
    monkeypatch.setattr(
        pull_mod, "_list_repo_files", lambda repo, rev: ["config.json", "model.safetensors"]
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))

    entry = pull_mod.pull("org/Model")

    assert entry.name == "Model"
    assert entry.quant == "4bit"
    assert entry.mlx_ready is True
    assert registry.get("Model").repo_id == "org/Model"


def test_pull_raises_without_config(monkeypatch):
    monkeypatch.setattr(pull_mod, "_list_repo_files", lambda repo, rev: ["model.safetensors"])
    with pytest.raises(ValueError, match="config.json"):
        pull_mod.pull("org/NoConfig")


def test_pull_convert_path_chosen_by_flag(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, {"config.json": b"{}", "model.safetensors": b"x"})
    monkeypatch.setattr(
        pull_mod, "_list_repo_files", lambda repo, rev: ["config.json", "model.safetensors"]
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))

    sentinel = ModelEntry(name="X-8bit-mlx", repo_id="/local", path="/local", quant="8bit")
    seen = {}
    monkeypatch.setattr(
        pull_mod, "_convert", lambda repo, snapshot, bits: seen.update(bits=bits) or sentinel
    )

    entry = pull_mod.pull("org/X", convert=True, bits=8)

    assert entry is sentinel
    assert seen["bits"] == 8
    assert registry.get("X-8bit-mlx") is not None


def test_pull_converts_when_no_safetensors(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, {"config.json": b"{}", "pytorch_model.bin": b"x"})
    monkeypatch.setattr(
        pull_mod, "_list_repo_files", lambda repo, rev: ["config.json", "pytorch_model.bin"]
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))

    sentinel = ModelEntry(name="Y-4bit-mlx", repo_id="/local", path="/local")
    monkeypatch.setattr(pull_mod, "_convert", lambda *a, **k: sentinel)

    assert pull_mod.pull("org/Y") is sentinel


def test_pull_downloads_pytorch_weights_for_convert(tmp_path, monkeypatch):
    """The convert path (no safetensors) must request *.bin weights, else the
    snapshot has no weights for mlx_lm.convert to read."""
    snap = _snapshot(tmp_path, {"config.json": b"{}", "pytorch_model.bin": b"x"})
    monkeypatch.setattr(
        pull_mod, "_list_repo_files", lambda repo, rev: ["config.json", "pytorch_model.bin"]
    )
    captured = {}

    def fake_download(*a, **k):
        captured.update(k)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    sentinel = ModelEntry(name="Y", repo_id="/l", path="/l")
    monkeypatch.setattr(pull_mod, "_convert", lambda *a, **k: sentinel)

    pull_mod.pull("org/Y")
    assert "*.bin" in captured["allow_patterns"]


def test_dir_size_ignores_stat_errors(tmp_path, monkeypatch):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 5)

    real_is_file = Path.is_file
    real_stat = Path.stat
    boom = tmp_path / "sub" / "b.bin"

    def fake_is_file(self):
        if self == boom:
            return True
        return real_is_file(self)

    def fake_stat(self, *a, **k):
        if self == boom:
            raise OSError("permission denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "stat", fake_stat)
    assert _dir_size(tmp_path) == 10


def test_list_repo_files_calls_hf_api(monkeypatch):
    sibling = types.SimpleNamespace(rfilename="config.json")
    info = types.SimpleNamespace(siblings=[sibling])

    fake_api_cls = []

    class FakeApi:
        def __init__(self):
            fake_api_cls.append(self)

        def model_info(self, repo_id, revision=None, files_metadata=False):
            return info

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert _list_repo_files("org/M", None) == ["config.json"]


def test_convert_invokes_mlx_convert_and_builds_entry(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, {"config.json": b"{}"})

    out = settings.converted_dir / "X-8bit-mlx"
    out.mkdir(parents=True)
    (out / "model.safetensors").write_bytes(b"q" * 40)

    convert_module = types.ModuleType("mlx_lm.convert")
    seen = {}

    def fake_convert(src, mlx_path: str = "", quantize=False, q_bits=4):
        seen["src"] = src
        seen["mlx_path"] = mlx_path
        seen["q_bits"] = q_bits
        assert mlx_path
        Path(mlx_path).mkdir(parents=True, exist_ok=True)
        (Path(mlx_path) / "model.safetensors").write_bytes(b"q" * 40)

    convert_module.convert = fake_convert
    monkeypatch.setitem(sys.modules, "mlx_lm.convert", convert_module)

    entry = pull_mod._convert("org/X", snap, bits=8)

    assert entry.name == "X-8bit-mlx"
    assert entry.quant == "8bit"
    assert entry.repo_id == str(out)
    assert entry.path == str(out)
    assert entry.size_bytes == 40
    assert seen["q_bits"] == 8
    assert seen["mlx_path"] == str(out)
