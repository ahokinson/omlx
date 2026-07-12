"""omlx command-line interface."""

from __future__ import annotations

import typer

from . import daemon, registry
from ._fmt import human_size
from .client import StreamError
from .config import ensure_dirs, settings

app = typer.Typer(
    add_completion=False,
    help="omlx: run Hugging Face models locally on MLX. No GGUF.",
    no_args_is_help=True,
)


def _stream_once(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Stream one chat turn to the daemon, echoing deltas; return the reply.

    Reasoning deltas are printed dimmed; only the final answer is returned.
    Raises ``StreamError`` on a mid-stream error frame so the caller can decide
    how to handle the failed turn (drop it, exit, etc.). Imports are local so
    tests can monkeypatch ``omlx.client.stream_chat`` after the fact.
    """
    from .client import stream_chat

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    out: list[str] = []
    try:
        for kind, delta in stream_chat(url, body):
            if kind == "reasoning":
                typer.echo(typer.style(delta, dim=True), nl=False)
            else:
                typer.echo(delta, nl=False)
                out.append(delta)
    except StreamError as e:
        typer.echo(f"\n[error] {e}", err=True)
        raise
    finally:
        typer.echo("")
    return "".join(out)


@app.command()
def pull(
    repo_id: str = typer.Argument(
        ..., help="HF repo id, e.g. mlx-community/Llama-3.2-1B-Instruct-4bit"
    ),
    revision: str | None = typer.Option(None, help="Git revision / branch / tag."),
    convert: bool = typer.Option(False, "--convert", help="Quantize a non-MLX repo on-device."),
    bits: int = typer.Option(4, help="Quantization bits when converting."),
):
    """Download a model (Xet-accelerated) and register it."""
    from .pull import pull as do_pull

    entry = do_pull(repo_id, revision=revision, convert=convert, bits=bits)
    typer.echo(
        f"pulled {entry.name}  ({human_size(entry.size_bytes)}, quant={entry.quant or 'none'})"
    )


@app.command(name="list")
def list_models():
    """List local models."""
    rows = registry.entries()
    if not rows:
        typer.echo("no models; pull one with `omlx pull <repo>`")
        return
    name_w = max(len(r.name) for r in rows)
    for r in sorted(rows, key=lambda x: x.name):
        typer.echo(
            f"{r.name:<{name_w}}  {human_size(r.size_bytes):>9}  {r.quant or '-':>6}  {r.repo_id}"
        )


@app.command()
def rm(name: str = typer.Argument(..., help="Model name (or repo id) to remove.")):
    """Remove a model from the registry and purge its cache snapshot."""
    entry = registry.remove(name)
    if entry is None:
        typer.echo(f"no such model: {name}")
        raise typer.Exit(1)
    typer.echo(f"removed {entry.name}")


@app.command()
def serve():
    """Run the OpenAI-compatible server in the foreground."""
    import uvicorn

    ensure_dirs()
    uvicorn.run("omlx.server:app", host=settings.host, port=settings.port, log_level="info")


daemon_app = typer.Typer(help="Manage the background daemon.", no_args_is_help=True)
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start():
    daemon.start()


@daemon_app.command("stop")
def daemon_stop():
    daemon.stop()


@daemon_app.command("status")
def daemon_status():
    daemon.status()


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name or HF repo id."),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="One-shot prompt (non-interactive)."
    ),
    max_tokens: int = typer.Option(512, help="Max tokens to generate."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
):
    """Chat with a model. Auto-starts the daemon and keeps the model warm."""
    daemon.ensure_running()
    url = settings.base_url + "/v1/chat/completions"

    if prompt is not None:
        try:
            _stream_once(
                url,
                model,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except StreamError as e:
            raise typer.Exit(1) from e
        return

    typer.echo(f"omlx: chatting with {model}. Type /bye to exit.\n")
    history: list[dict[str, str]] = []
    while True:
        try:
            user = typer.prompt(">>>", prompt_suffix=" ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            break
        if user.strip() in ("/bye", "/exit", "/quit"):
            break
        history.append({"role": "user", "content": user})
        try:
            reply = _stream_once(
                url, model, history, max_tokens=max_tokens, temperature=temperature
            )
        except StreamError:
            history.pop()  # drop the turn we couldn't answer
            continue
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    app()
