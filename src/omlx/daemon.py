"""Background uvicorn daemon lifecycle: start / stop / status / ensure."""

from __future__ import annotations

import errno
import logging
import os
import signal
import subprocess
import sys
import time

import httpx2

from .config import ensure_dirs, settings

logger = logging.getLogger("omlx")

HEALTH_TIMEOUT = 1.0
STATUS_TIMEOUT = 2.0
STARTUP_TIMEOUT = 30.0
POLL_INTERVAL = 0.4


def _echo(msg: str) -> None:
    from ._ui import info

    info(f"[omlx] {msg}")


def _read_pid() -> int | None:
    if not settings.pid_path.exists():
        return None
    try:
        return int(settings.pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _healthy(timeout: float) -> httpx2.Response | None:
    """GET /health, returning the response only on HTTP 200, else None."""
    try:
        r = httpx2.get(settings.base_url + "/health", timeout=timeout)
    except httpx2.HTTPError:
        return None
    return r if r.status_code == 200 else None


def is_running() -> bool:
    """True if the pid file points at a live, health-responsive daemon."""
    pid = _read_pid()
    if pid is None or not _alive(pid):
        return False
    return _healthy(HEALTH_TIMEOUT) is not None


def start() -> None:
    """Spawn the detached uvicorn daemon and wait for it to become healthy."""
    if is_running():
        _echo(f"daemon already running at {settings.base_url}")
        return
    ensure_dirs()
    with settings.log_path.open("a") as log:
        # The child dup()s the log fd on POSIX, so closing our copy here is
        # correct: the daemon keeps writing to the file after we return.
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "omlx.server:app",
                "--host",
                settings.host,
                "--port",
                str(settings.port),
                "--log-level",
                "info",
            ],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    settings.pid_path.write_text(str(proc.pid))
    try:
        from ._ui import status

        with status(f"starting daemon at {settings.base_url}…"):
            _wait_healthy(proc.pid)
    except RuntimeError:
        # Startup timed out; the child may still be alive (slow import/model
        # load). Kill it before dropping the pidfile, else it orphans, holds the
        # port, and can no longer be reached via stop().
        if _alive(proc.pid):
            _terminate(proc.pid)
        settings.pid_path.unlink(missing_ok=True)
        raise


def _wait_healthy(pid: int, timeout: float = STARTUP_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            raise RuntimeError(f"daemon exited during startup; see {settings.log_path}")
        if _healthy(HEALTH_TIMEOUT) is not None:
            _echo(f"daemon running at {settings.base_url} (pid {pid})")
            return
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(f"daemon did not become healthy; see {settings.log_path}")


def stop() -> None:
    """Terminate the daemon's process group (SIGTERM) and clear the pid file."""
    pid = _read_pid()
    if pid is None:
        _echo("daemon not running")
        return
    if _alive(pid):
        _terminate(pid)
        _echo(f"stopped daemon (pid {pid})")
    settings.pid_path.unlink(missing_ok=True)


def _terminate(pid: int) -> None:
    """SIGTERM the daemon's process group, falling back to the process itself.

    ``ESRCH`` (no such process) between our ``_alive`` check and the signal is
    silent (the daemon already exited), but ``EPERM`` and other failures are
    logged, since the bare ``except OSError: pass`` this replaces could mask a
    real permission problem. The pid file is cleared either way by the caller.
    """
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError as e:
        if e.errno == errno.ESRCH:
            return
        logger.warning("killpg(%d) failed: %s; trying kill(%d) instead", pid, e, pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e2:
            if e2.errno != errno.ESRCH:
                logger.warning("kill(%d) failed: %s", pid, e2)


def status() -> None:
    """Print whether the daemon is running and which model it has loaded."""
    pid = _read_pid()
    r = _healthy(STATUS_TIMEOUT) if pid is not None and _alive(pid) else None
    if r is not None:
        body = r.json()
        loaded = body.get("loaded") or "none"
        models = body.get("loaded_models") or ([loaded] if loaded != "none" else [])
        tail = f", models={','.join(models)}" if models and models != [loaded] else ""
        _echo(f"running at {settings.base_url} (pid {pid}), loaded={loaded}{tail}")
    else:
        _echo("daemon not running")


def ensure_running() -> None:
    """Start the daemon if it is not already running."""
    if not is_running():
        start()
