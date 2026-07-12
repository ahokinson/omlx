from __future__ import annotations

import pytest
import uvicorn
from typer.testing import CliRunner

from omlx import daemon, registry
from omlx import pull as pull_mod
from omlx._fmt import human_size
from omlx.cli import _stream_once, app
from omlx.client import StreamError
from omlx.config import settings


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "0.0B"),
        (1536, "1.5KB"),
        (5 * 1024**2, "5.0MB"),
        (5 * 1024**3, "5.0GB"),
        (3 * 1024**4, "3.0TB"),
        (1024**5, "1024.0TB"),
    ],
)
def test_human_formats_bytes(n, expected):
    assert human_size(n) == expected


def test_list_empty_hint(runner):
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "no models" in result.stdout


def test_list_shows_registered_model(runner, make_entry):
    registry.add(make_entry(name="Llama", repo_id="org/Llama", quant="4bit"))
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "Llama" in result.stdout
    assert "org/Llama" in result.stdout


def test_rm_known_model(runner, no_purge, make_entry):
    registry.add(make_entry(name="Llama", repo_id="org/Llama"))
    result = runner.invoke(app, ["rm", "Llama"])
    assert result.exit_code == 0
    assert "removed Llama" in result.stdout


def test_rm_unknown_model_exits_1(runner):
    result = runner.invoke(app, ["rm", "ghost"])
    assert result.exit_code == 1
    assert "no such model" in result.stdout


def test_run_one_shot_streams_reply(runner, monkeypatch):
    monkeypatch.setattr("omlx.daemon.ensure_running", lambda: None)
    monkeypatch.setattr("omlx.client.stream_chat", lambda url, body: iter(["Hello", " world"]))
    result = runner.invoke(app, ["run", "M", "-p", "hi"])
    assert result.exit_code == 0
    assert "Hello world" in result.stdout


def test_serve_invokes_uvicorn(runner, monkeypatch):
    called = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: called.update(kw))
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    assert called["host"] == settings.host
    assert called["port"] == settings.port


def test_stream_once_streams_deltas_and_returns_full_reply(monkeypatch, capsys):
    monkeypatch.setattr("omlx.client.stream_chat", lambda url, body: iter(["Hello", " world"]))
    out = _stream_once(
        "http://x/v1/chat/completions",
        "M",
        [{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.5,
    )
    assert out == "Hello world"
    assert "Hello world" in capsys.readouterr().out


def test_stream_once_propagates_stream_error(monkeypatch):
    def _boom(url, body):
        raise StreamError("oops")
        yield  # makes this a generator function; raise fires on first next()

    monkeypatch.setattr("omlx.client.stream_chat", _boom)
    with pytest.raises(StreamError, match="oops"):
        _stream_once(
            "http://x", "M", [{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.5
        )


def test_pull_invokes_do_pull_and_prints(runner, monkeypatch, make_entry):
    entry = make_entry(name="Llama", repo_id="org/Llama", quant="4bit", size_bytes=2048)
    monkeypatch.setattr(pull_mod, "pull", lambda *a, **k: entry)
    result = runner.invoke(app, ["pull", "org/Llama"])
    assert result.exit_code == 0
    assert "pulled Llama" in result.stdout
    assert "2.0KB" in result.stdout
    assert "4bit" in result.stdout


def test_daemon_start_via_cli(runner, monkeypatch):
    called = []
    monkeypatch.setattr(daemon, "start", lambda: called.append(1))
    result = runner.invoke(app, ["daemon", "start"])
    assert result.exit_code == 0
    assert called == [1]


def test_daemon_stop_via_cli(runner, monkeypatch):
    called = []
    monkeypatch.setattr(daemon, "stop", lambda: called.append(1))
    result = runner.invoke(app, ["daemon", "stop"])
    assert result.exit_code == 0
    assert called == [1]


def test_daemon_status_via_cli(runner, monkeypatch):
    called = []
    monkeypatch.setattr(daemon, "status", lambda: called.append(1))
    result = runner.invoke(app, ["daemon", "status"])
    assert result.exit_code == 0
    assert called == [1]


def test_run_one_shot_streamerror_exits_1(runner, monkeypatch):
    monkeypatch.setattr("omlx.daemon.ensure_running", lambda: None)

    def _boom(url, body):
        raise StreamError("oops")
        yield

    monkeypatch.setattr("omlx.client.stream_chat", _boom)
    result = runner.invoke(app, ["run", "M", "-p", "hi"])
    assert result.exit_code == 1


def test_run_interactive_bye_exits_cleanly(runner, monkeypatch):
    monkeypatch.setattr("omlx.daemon.ensure_running", lambda: None)
    monkeypatch.setattr("omlx.client.stream_chat", lambda url, body: iter(["reply"]))
    result = runner.invoke(app, ["run", "M"], input="hi\n/bye\n")
    assert result.exit_code == 0
    assert "reply" in result.stdout


def test_run_interactive_eof_breaks(runner, monkeypatch):
    """click/typer normally converts EOF into Abort (exit 1), so patch
    typer.prompt to raise a genuine EOFError and exercise the handler.
    """
    monkeypatch.setattr("omlx.daemon.ensure_running", lambda: None)
    monkeypatch.setattr("omlx.client.stream_chat", lambda url, body: iter(["reply"]))

    def _eof(*a, **k):
        raise EOFError

    monkeypatch.setattr("omlx.cli.typer.prompt", _eof)
    result = runner.invoke(app, ["run", "M"])
    assert result.exit_code == 0


def test_run_interactive_streamerror_drops_turn(runner, monkeypatch):
    monkeypatch.setattr("omlx.daemon.ensure_running", lambda: None)

    calls = {"n": 0}

    def _stream(url, body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise StreamError("boom")
            yield
        yield "ok"

    monkeypatch.setattr("omlx.client.stream_chat", _stream)
    result = runner.invoke(app, ["run", "M"], input="hi\nagain\n/bye\n")
    assert result.exit_code == 0
    assert "ok" in result.stdout
