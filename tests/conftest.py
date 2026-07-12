"""Shared fixtures: redirect omlx state into a temp dir so tests never touch
the real ~/.omlx registry or HF cache."""

from __future__ import annotations

import pytest

from omlx import registry
from omlx.config import settings
from omlx.registry import ModelEntry


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    home = tmp_path / ".omlx"
    monkeypatch.setattr(settings, "home", home)
    return home


@pytest.fixture
def make_entry():
    """Factory for ModelEntry with sensible defaults, overridable per call."""

    def _make(name="m", repo_id="org/m", path="/tmp/m", **over):
        return ModelEntry(name=name, repo_id=repo_id, path=path, **over)

    return _make


@pytest.fixture
def no_purge(monkeypatch):
    """Stub the HF-cache purge so removal tests never touch the real cache."""
    monkeypatch.setattr(registry, "_purge_cache", lambda repo_id: None)
