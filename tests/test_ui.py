from __future__ import annotations

import types

from omlx import _ui


def _tasks(progress):
    return {t.id: t for t in progress.tasks}


def test_download_progress_tracks_updates():
    """A bar driven through the yielded tqdm_class advances its rich task."""
    with _ui.download_progress() as tqdm_cls:
        progress = _ui._RichTqdm._progress
        bar = tqdm_cls(total=100, desc="model.safetensors")
        assert len(progress.tasks) == 1

        bar.update(40)
        assert _tasks(progress)[bar._task].completed == 40

        bar.update(60)
        assert _tasks(progress)[bar._task].completed == 100

        bar.close()  # snaps to total
        assert _tasks(progress)[bar._task].completed == 100

    # Context exit clears the shared progress reference.
    assert _ui._RichTqdm._progress is None


def test_download_progress_indeterminate_total_none():
    """A bar with unknown total still advances without error."""
    with _ui.download_progress() as tqdm_cls:
        progress = _ui._RichTqdm._progress
        bar = tqdm_cls(total=None, desc="")
        bar.update(5)
        assert _tasks(progress)[bar._task].completed == 5
        bar.close()  # total is None -> no snap, no error


def test_disabled_bar_registers_no_task():
    with _ui.download_progress() as tqdm_cls:
        bar = tqdm_cls(total=100, disable=True)
        assert bar._task is None
        bar.update(10)  # no-op against the progress
        bar.close()


def test_tqdm_without_active_progress_is_silent():
    """Outside a download_progress block the class behaves like plain tqdm."""
    assert _ui._RichTqdm._progress is None
    bar = _ui._RichTqdm(total=10, disable=True)
    assert bar._task is None
    bar.update(3)
    bar.close()


def test_models_table_prints_names_and_repos(capsys):
    rows = [
        types.SimpleNamespace(name="Llama", size_bytes=2048, quant="4bit", repo_id="org/Llama"),
        types.SimpleNamespace(name="Alpha", size_bytes=1024, quant=None, repo_id="org/Alpha"),
    ]
    _ui.models_table(rows)
    out = capsys.readouterr().out
    assert "Llama" in out and "org/Llama" in out
    assert "Alpha" in out and "org/Alpha" in out
    assert "2.0KB" in out
    # None quant renders as a dash.
    assert "-" in out


def test_success_info_error_render_text(capsys):
    _ui.success("pulled X")
    _ui.info("[omlx] hello")
    _ui.error("boom")
    captured = capsys.readouterr()
    assert "pulled X" in captured.out
    assert "[omlx] hello" in captured.out  # brackets kept literal (no markup)
    assert "boom" in captured.err


def test_status_spinner_context():
    with _ui.status("working…"):
        pass


def test_null_sink_swallows_writes():
    sink = _ui._NullSink()
    assert sink.write("anything") is None
    assert sink.flush() is None
    assert sink.isatty() is False
