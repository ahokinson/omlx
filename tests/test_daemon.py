from __future__ import annotations

import errno
import os

import pytest

from omlx import daemon
from omlx.config import settings


def test_read_pid_none_when_absent():
    assert daemon._read_pid() is None


def test_read_pid_roundtrip():
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text("4321\n")
    assert daemon._read_pid() == 4321


def test_read_pid_none_when_garbage():
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text("not-a-pid")
    assert daemon._read_pid() is None


def test_alive_true_for_self_false_for_bogus():
    assert daemon._alive(os.getpid()) is True
    assert daemon._alive(2**31 - 1) is False


def test_is_running_false_without_pid():
    assert daemon.is_running() is False


def test_is_running_checks_health(monkeypatch):
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(str(os.getpid()))

    class Resp:
        status_code = 200

    monkeypatch.setattr(daemon, "_healthy", lambda timeout: Resp())
    assert daemon.is_running() is True

    monkeypatch.setattr(daemon, "_healthy", lambda timeout: None)
    assert daemon.is_running() is False


class _FakeProc:
    pid = os.getpid()


def test_start_writes_pid_and_reports_healthy(monkeypatch, capsys):
    monkeypatch.setattr(daemon.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(daemon, "_healthy", lambda timeout: type("R", (), {"status_code": 200})())

    daemon.start()

    assert daemon._read_pid() == os.getpid()
    out = capsys.readouterr().out
    assert "[omlx] daemon running" in out


def test_start_cleans_up_pid_on_startup_failure(monkeypatch):
    monkeypatch.setattr(daemon.subprocess, "Popen", lambda *a, **k: _FakeProc())

    def boom(pid, timeout=daemon.STARTUP_TIMEOUT):
        raise RuntimeError("did not become healthy")

    monkeypatch.setattr(daemon, "_wait_healthy", boom)
    # The child (_FakeProc.pid == our pid) is still alive on timeout, so start()
    # must terminate it before dropping the pidfile — never orphan it.
    terminated = []
    monkeypatch.setattr(daemon, "_terminate", lambda pid: terminated.append(pid))

    with pytest.raises(RuntimeError):
        daemon.start()

    assert terminated == [os.getpid()]
    assert not settings.pid_path.exists()


def test_stop_no_pid_reports_not_running(capsys):
    daemon.stop()
    assert "[omlx] daemon not running" in capsys.readouterr().out


def test_stop_signals_and_clears_pid(monkeypatch, capsys):
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(str(os.getpid()))

    killed = []
    monkeypatch.setattr(daemon.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(daemon.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))

    daemon.stop()

    assert killed
    assert not settings.pid_path.exists()
    assert "[omlx] stopped daemon" in capsys.readouterr().out


def test_healthy_returns_none_on_httpx_error(monkeypatch):
    def _boom(*a, **k):
        raise daemon.httpx2.HTTPError("nope")

    monkeypatch.setattr(daemon.httpx2, "get", _boom)
    assert daemon._healthy(1.0) is None


def test_healthy_returns_none_on_non_200(monkeypatch):
    class Resp:
        status_code = 503

    monkeypatch.setattr(daemon.httpx2, "get", lambda *a, **k: Resp())
    assert daemon._healthy(1.0) is None


def test_start_when_already_running(monkeypatch, capsys):
    monkeypatch.setattr(daemon, "is_running", lambda: True)
    daemon.start()
    out = capsys.readouterr().out
    assert "already running" in out
    assert not settings.pid_path.exists()


def test_wait_healthy_raises_if_proc_dies(monkeypatch):
    monkeypatch.setattr(daemon, "_alive", lambda pid: False)
    with pytest.raises(RuntimeError, match="daemon exited during startup"):
        daemon._wait_healthy(os.getpid())


def test_wait_healthy_raises_on_timeout(monkeypatch):
    monkeypatch.setattr(daemon, "_alive", lambda pid: True)
    monkeypatch.setattr(daemon, "_healthy", lambda timeout: None)

    t = {"v": 0.0}

    def fake_time():
        t["v"] += 100.0
        return t["v"]

    monkeypatch.setattr(daemon.time, "time", fake_time)
    with pytest.raises(RuntimeError, match="did not become healthy"):
        daemon._wait_healthy(os.getpid())


def test_terminate_silent_on_esrch(monkeypatch):
    def _boom(pgid, sig):
        raise OSError(errno.ESRCH, "no such process")

    monkeypatch.setattr(daemon.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(daemon.os, "killpg", _boom)
    daemon._terminate(os.getpid())


def test_terminate_falls_back_to_kill_on_oserror(monkeypatch):
    killed = []

    def _killpg(pgid, sig):
        raise OSError(errno.EPERM, "no perm")

    monkeypatch.setattr(daemon.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(daemon.os, "killpg", _killpg)
    monkeypatch.setattr(daemon.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    daemon._terminate(os.getpid())
    assert killed


def test_terminate_kill_esrch_is_silent(monkeypatch):
    monkeypatch.setattr(daemon.os, "getpgid", lambda pid: pid)

    def _killpg(pgid, sig):
        raise OSError(errno.EPERM, "no perm")

    def _kill(pid, sig):
        raise OSError(errno.ESRCH, "gone")

    monkeypatch.setattr(daemon.os, "killpg", _killpg)
    monkeypatch.setattr(daemon.os, "kill", _kill)
    daemon._terminate(os.getpid())


def test_terminate_logs_when_kill_also_fails(monkeypatch):
    monkeypatch.setattr(daemon.os, "getpgid", lambda pid: pid)

    def _killpg(pgid, sig):
        raise OSError(errno.EPERM, "no perm")

    def _kill(pid, sig):
        raise OSError(errno.EPERM, "still no perm")

    monkeypatch.setattr(daemon.os, "killpg", _killpg)
    monkeypatch.setattr(daemon.os, "kill", _kill)
    daemon._terminate(os.getpid())  # logged, not raised


def test_wait_healthy_polls_then_succeeds(monkeypatch):
    """Exercise the poll-loop path that sleeps between health checks."""
    monkeypatch.setattr(daemon, "_alive", lambda pid: True)
    polls = {"n": 0}

    class Resp:
        status_code = 200

    def _healthy(timeout):
        polls["n"] += 1
        return Resp() if polls["n"] >= 2 else None

    monkeypatch.setattr(daemon, "_healthy", _healthy)
    monkeypatch.setattr(daemon.time, "sleep", lambda s: None)
    daemon._wait_healthy(os.getpid())
    assert polls["n"] == 2


def test_ensure_running_starts_when_not_running(monkeypatch):
    started = []
    monkeypatch.setattr(daemon, "is_running", lambda: False)
    monkeypatch.setattr(daemon, "start", lambda: started.append(1))
    daemon.ensure_running()
    assert started == [1]


def test_ensure_running_noop_when_running(monkeypatch):
    started = []
    monkeypatch.setattr(daemon, "is_running", lambda: True)
    monkeypatch.setattr(daemon, "start", lambda: started.append(1))
    daemon.ensure_running()
    assert started == []


def test_status_no_pid_reports_not_running(capsys):
    daemon.status()
    assert "[omlx] daemon not running" in capsys.readouterr().out


def test_status_running_reports_loaded(monkeypatch, capsys):
    settings.home.mkdir(parents=True, exist_ok=True)
    settings.pid_path.write_text(str(os.getpid()))

    class Resp:
        status_code = 200

        def json(self):
            return {"loaded": "Llama"}

    monkeypatch.setattr(daemon, "_healthy", lambda timeout: Resp())
    daemon.status()
    out = capsys.readouterr().out
    assert "running" in out
    assert "loaded=Llama" in out
