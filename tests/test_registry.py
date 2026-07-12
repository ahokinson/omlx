from __future__ import annotations

import sys
import types

from omlx import registry
from omlx.config import settings


def test_name_for_drops_org_prefix():
    assert (
        registry.name_for("mlx-community/Llama-3.2-1B-Instruct-4bit")
        == "Llama-3.2-1B-Instruct-4bit"
    )
    assert registry.name_for("bare-name") == "bare-name"


def test_add_get_roundtrip(make_entry):
    e = make_entry(name="Llama", repo_id="org/Llama", quant="4bit", size_bytes=123)
    registry.add(e)
    assert registry.get("Llama") == e


def test_get_matches_friendly_name_and_repo_id(make_entry):
    registry.add(make_entry(name="my-llama", repo_id="org/Llama"))
    by_name = registry.get("my-llama")
    by_repo = registry.get("org/Llama")
    assert by_name is not None and by_name.name == "my-llama"
    assert by_repo is not None and by_repo.name == "my-llama"


def test_resolve_prefers_exact_key_over_friendly_tail(make_entry):
    registry.add(make_entry(name="org/X", repo_id="org/X"))
    registry.add(make_entry(name="X", repo_id="other/X"))
    got = registry.get("org/X")
    assert got is not None and got.name == "org/X"


def test_resolve_prefers_friendly_tail_over_repo_id_scan(make_entry):
    registry.add(make_entry(name="Llama", repo_id="OTHER/Llama"))
    registry.add(make_entry(name="other", repo_id="org/Llama"))
    got = registry.get("org/Llama")
    assert got is not None and got.name == "Llama"


def test_get_missing_returns_none():
    assert registry.get("nope") is None


def test_entries_lists_all(make_entry):
    registry.add(make_entry(name="a", repo_id="org/a"))
    registry.add(make_entry(name="b", repo_id="org/b"))
    assert {e.name for e in registry.entries()} == {"a", "b"}


def test_remove_returns_entry_and_drops_it(no_purge, make_entry):
    registry.add(make_entry(name="a", repo_id="org/a"))
    removed = registry.remove("a")
    assert removed is not None and removed.name == "a"
    assert registry.get("a") is None
    assert registry.remove("a") is None  # already gone


def test_remove_by_repo_id(no_purge, make_entry):
    registry.add(make_entry(name="my-llama", repo_id="org/Llama"))
    removed = registry.remove("org/Llama")
    assert removed is not None and removed.name == "my-llama"
    assert registry.get("my-llama") is None


def test_remove_deletes_converted_dir(make_entry):
    """A converted model's local dir under ~/.omlx/converted is deleted on rm."""
    conv = settings.converted_dir / "Y-4bit-mlx"
    conv.mkdir(parents=True)
    (conv / "weights.safetensors").write_bytes(b"x")
    registry.add(make_entry(name="Y-4bit-mlx", repo_id=str(conv), path=str(conv)))

    removed = registry.remove("Y-4bit-mlx")

    assert removed is not None
    assert not conv.exists()


def test_load_ignores_corrupt_registry():
    registry.ensure_dirs()
    settings.registry_path.write_text("{ not valid json")
    assert registry.entries() == []


def test_purge_cache_tolerates_corrupt_cache(monkeypatch):
    import huggingface_hub

    def _boom():
        raise ValueError("corrupt cache metadata")

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", _boom)
    registry._purge_cache("org/M")


def test_purge_cache_returns_when_hf_unimportable(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    registry._purge_cache("org/M")


def test_purge_cache_deletes_matching_revisions(monkeypatch):
    import huggingface_hub

    rev = types.SimpleNamespace(commit_hash="abc123")
    repo = types.SimpleNamespace(repo_id="org/M", repo_type="model", revisions=[rev])
    info = types.SimpleNamespace(
        repos=[repo],
        delete_revisions=lambda *hashes: types.SimpleNamespace(
            execute=lambda: hashes,
        ),
    )
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: info)
    assert registry._purge_cache("org/M") is None
