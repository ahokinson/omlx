"""Paths and defaults for omlx, driven by ``OMLX_*`` environment variables."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("omlx")

_KEEPALIVE_DEFAULT = 5 * 60
_MEM_BUDGET_FRACTION_DEFAULT = 0.6
_FALLBACK_BUDGET_MB = 4096


def _count_cap_for_budget_mb(budget_mb: int) -> int:
    """Resident-model count cap derived from a memory budget (~1 model / 8 GiB).

    Clamped to ``[1, 16]``. Shared by :meth:`Settings.max_loaded` and the engine
    so a manager built with a custom budget derives a consistent cap.
    """
    return max(1, min(16, budget_mb // 8192))


def _system_mem_mb() -> int:
    """Total system memory in MiB, or a conservative fallback if probing fails.

    Probing uses cheap, side-effect-free reads (sysctl on Darwin, /proc/meminfo
    on Linux); no MLX/Metal dependency. Tests never call this directly.
    """
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL, timeout=2
            ).strip()
            return int(out) // (1024 * 1024)
        if platform.system() == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception as e:
        logger.debug("system mem probe failed: %s", e)
    return _FALLBACK_BUDGET_MB


class Settings(BaseSettings):
    """Runtime configuration, overridable via ``OMLX_*`` env vars.

    Path fields derive from ``home``, so tests (and callers) only need to
    override that one value to relocate all omlx state.
    """

    model_config = SettingsConfigDict(env_prefix="OMLX_", env_prefix_target="all")

    home: Path = Path.home() / ".omlx"
    # 11434 mirrors Ollama for drop-in clients.
    host: str = "127.0.0.1"
    port: int = 11434
    keepalive_seconds: int = Field(
        _KEEPALIVE_DEFAULT, validation_alias=AliasChoices("keepalive_seconds", "keepalive")
    )
    # Multi-model residency. When both are None (default), the budget is
    # `mem_budget_fraction` of total system memory and the count cap is derived
    # from it. Set OMLX_MAX_LOADED_MODELS / OMLX_MAX_MEM_MB to pin explicitly.
    max_loaded_models: int | None = Field(
        None, validation_alias=AliasChoices("max_loaded_models", "max_loaded")
    )
    max_mem_mb: int | None = None
    mem_budget_fraction: float = _MEM_BUDGET_FRACTION_DEFAULT
    # Reuse a per-model KV cache across turns, prefilling only the diverging
    # suffix of the prompt. Disable to prefill the full prompt every request.
    prompt_cache: bool = True
    # Quantized KV cache. `kv_bits` None keeps the cache in full precision;
    # otherwise entries past `quantized_kv_start` tokens are quantized to that
    # bit width, trading a little quality for less KV memory on long contexts.
    kv_bits: int | None = None
    kv_group_size: int = 64
    quantized_kv_start: int = 5000

    @property
    def registry_path(self) -> Path:
        return self.home / "models.json"

    @property
    def converted_dir(self) -> Path:
        return self.home / "converted"

    @property
    def pid_path(self) -> Path:
        return self.home / "daemon.pid"

    @property
    def log_path(self) -> Path:
        return self.home / "daemon.log"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def mem_budget_mb(self) -> int:
        """Max resident set budget in MiB.

        Explicit `OMLX_MAX_MEM_MB` wins; otherwise derive from a fraction of
        total system memory (probed lazily, no Metal dependency).
        """
        if self.max_mem_mb is not None:
            return self.max_mem_mb
        return max(1, int(self.mem_budget_fraction * _system_mem_mb()))

    def max_loaded(self) -> int:
        """Count cap on resident models, derived from the memory budget if unset."""
        if self.max_loaded_models is not None:
            return max(1, self.max_loaded_models)
        # Reasonable default: ~one model per 8 GiB of budget. Clamped to [1, 16].
        return _count_cap_for_budget_mb(self.mem_budget_mb())


settings = Settings()


def ensure_dirs() -> None:
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.converted_dir.mkdir(parents=True, exist_ok=True)
