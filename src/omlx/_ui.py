"""Rich-based terminal styling for the CLI: download progress, tables, messages.

All user-facing rendering funnels through here so the command code stays
declarative. Two consoles: ``console`` writes stdout (tables, success lines,
pipeable output); ``err_console`` writes stderr (progress bars, spinners,
errors), matching huggingface_hub's convention and keeping stdout clean for
piping. Both are built with ``file=None`` so rich resolves ``sys.stdout`` /
``sys.stderr`` at write time, which keeps ``CliRunner`` output capture working.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text
from tqdm.std import tqdm as std_tqdm

console = Console()
err_console = Console(stderr=True)


class _NullSink:
    """A file-like that swallows tqdm's own rendering; rich owns the display."""

    def write(self, *args) -> None:
        pass

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class _RichTqdm(std_tqdm):
    """A ``tqdm`` drop-in that mirrors each bar into a shared rich ``Progress``.

    huggingface_hub creates one ``tqdm`` per file plus an outer "Fetching N
    files" bar and calls them through the ``tqdm_class`` we pass to
    ``snapshot_download``. Each instance registers a rich task and forwards its
    running count on every ``update``. ``_progress`` is set by
    ``download_progress`` for the lifetime of a download; when it is ``None``
    (or the bar is disabled) this behaves like a plain, silent ``tqdm``.
    """

    _progress: Progress | None = None

    def __init__(self, *args, **kwargs):
        # Route tqdm's own output to a sink so it keeps counting (``self.n``)
        # without drawing a second bar over the rich display.
        kwargs["file"] = _NullSink()
        super().__init__(*args, **kwargs)
        self._task = None
        if self.disable or _RichTqdm._progress is None:
            return
        self._task = _RichTqdm._progress.add_task(self.desc or "downloading", total=self.total)

    def update(self, n: float | None = 1) -> bool | None:
        displayed = super().update(n)
        prog = _RichTqdm._progress
        if self._task is not None and prog is not None:
            prog.update(self._task, completed=self.n, total=self.total)
        return displayed

    def close(self) -> None:
        prog = _RichTqdm._progress
        if self._task is not None and prog is not None:
            # Snap the bar to full so a completed file never shows a partial track.
            if self.total is not None:
                prog.update(self._task, completed=self.total)
        super().close()


@contextmanager
def download_progress() -> Iterator[type[_RichTqdm]]:
    """Yield a ``tqdm_class`` for ``snapshot_download`` backed by a live rich bar.

    On exit the shared ``Progress`` is torn down and the class reference is
    cleared so a later download starts clean.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=err_console,
        transient=True,
    )
    _RichTqdm._progress = progress
    with progress:
        try:
            yield _RichTqdm
        finally:
            _RichTqdm._progress = None


def models_table(rows: Iterable) -> None:
    """Print the local-model registry as a table (name / size / quant / repo)."""
    from ._fmt import human_size

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("NAME", style="cyan", no_wrap=True)
    table.add_column("SIZE", justify="right", style="green")
    table.add_column("QUANT", justify="right")
    table.add_column("REPO", style="dim", overflow="fold")
    for r in sorted(rows, key=lambda x: x.name):
        table.add_row(r.name, human_size(r.size_bytes), r.quant or "-", r.repo_id)
    console.print(table)


def success(msg: str) -> None:
    """Green check line on stdout."""
    text = Text("✓ ", style="green")
    text.append(msg)
    console.print(text)


def info(msg: str) -> None:
    """Dim status line on stdout. ``msg`` is treated as literal text (no markup)."""
    console.print(msg, style="dim", highlight=False, markup=False)


def error(msg: str) -> None:
    """Red error line on stderr. ``msg`` is treated as literal text (no markup)."""
    err_console.print(msg, style="red", highlight=False, markup=False)


@contextmanager
def status(msg: str) -> Iterator[None]:
    """Spinner on stderr for the duration of the block."""
    with err_console.status(msg):
        yield
