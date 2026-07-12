from __future__ import annotations

from pathlib import Path

from omlx import config
from omlx.config import Settings, settings


def test_base_url_format(monkeypatch):
    monkeypatch.setattr(settings, "host", "127.0.0.1")
    monkeypatch.setattr(settings, "port", 11434)
    assert settings.base_url == "http://127.0.0.1:11434"


def test_ensure_dirs_creates_home_and_converted(isolated_state):
    config.ensure_dirs()
    assert settings.home.is_dir()
    assert settings.converted_dir.is_dir()


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("OMLX_HOST", "0.0.0.0")
    monkeypatch.setenv("OMLX_PORT", "9999")
    monkeypatch.setenv("OMLX_KEEPALIVE", "42")
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 9999
    assert s.keepalive_seconds == 42


def test_keepalive_seconds_env_var_preferred_over_legacy(monkeypatch):
    monkeypatch.setenv("OMLX_KEEPALIVE_SECONDS", "7")
    monkeypatch.setenv("OMLX_KEEPALIVE", "42")
    assert Settings().keepalive_seconds == 7


def test_keepalive_seconds_default_when_unset():
    assert Settings().keepalive_seconds == 5 * 60


def test_paths_derive_from_home():
    s = Settings(home=Path("/somewhere/omlx"))
    assert s.registry_path == Path("/somewhere/omlx/models.json")
    assert s.converted_dir == Path("/somewhere/omlx/converted")
    assert s.pid_path == Path("/somewhere/omlx/daemon.pid")
    assert s.log_path == Path("/somewhere/omlx/daemon.log")


def test_mem_budget_and_count_can_be_pinned(monkeypatch):
    monkeypatch.setenv("OMLX_MAX_MEM_MB", "2048")
    monkeypatch.setenv("OMLX_MAX_LOADED_MODELS", "3")
    s = Settings()
    assert s.mem_budget_mb() == 2048
    assert s.max_loaded() == 3


def test_explicit_mem_budget_caps_loaded_count(monkeypatch):
    """When count is unset, it's derived from the budget (~one model per 8 GiB)."""
    monkeypatch.setenv("OMLX_MAX_MEM_MB", "8192")  # 8 GiB → ~1 resident
    monkeypatch.delenv("OMLX_MAX_LOADED_MODELS", raising=False)
    assert Settings().max_loaded() == 1


def test_max_loaded_clamps_to_at_least_one():
    s = Settings(max_loaded_models=0)
    assert s.max_loaded() == 1


def test_system_mem_probe_falls_back_on_unsupported_os(monkeypatch):
    """An OS we don't probe (`platform.system()` lying) returns the conservative default."""
    import platform

    monkeypatch.setattr(platform, "system", lambda: "WindowsNT")
    assert config._system_mem_mb() == config._FALLBACK_BUDGET_MB


def test_system_mem_probe_darwin_sysctl(monkeypatch):
    import platform
    import subprocess

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    # 8 GiB total: 8 * 1024 MiB
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"8589934592\n")
    assert config._system_mem_mb() == 8192


def test_system_mem_probe_linux_proc_meminfo(monkeypatch):
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    import omlx.config as cfg

    class FakePath:
        def __init__(self, target):
            self.target = str(target)

        def read_text(self):
            assert self.target == "/proc/meminfo"
            return "MemTotal:       16384000 kB\nSwapTotal:      0 kB\n"

    monkeypatch.setattr(cfg, "Path", FakePath)
    assert cfg._system_mem_mb() == 16000


def test_system_mem_probe_handles_probe_exception(monkeypatch):
    """A failed probe (subprocess timing out) falls back to the conservative default."""
    import platform
    import subprocess

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(["sysctl"], 2)),
    )
    assert config._system_mem_mb() == config._FALLBACK_BUDGET_MB


def test_max_loaded_models_legacy_alias(monkeypatch):
    monkeypatch.setenv("OMLX_MAX_LOADED", "5")
    assert Settings().max_loaded_models == 5


def test_keepalive_legacy_alias_still_works(monkeypatch):
    monkeypatch.setenv("OMLX_KEEPALIVE", "42")
    assert Settings().keepalive_seconds == 42
