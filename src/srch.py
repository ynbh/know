import typer
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
from rich import box
from rich.console import Console
from rich.table import Table

from src.db import ingest, ask, clear, SUPPORTED_EXTENSIONS

app = typer.Typer(help="srch - semantic search CLI")
console = Console()

INDEX_FILE = Path.home() / ".srch_dirs"


def _load_dirs() -> list[str]:
    if INDEX_FILE.exists():
        return [d.strip() for d in INDEX_FILE.read_text().splitlines() if d.strip()]
    return []


def _save_dirs(directories: list[str]) -> None:
    INDEX_FILE.write_text("\n".join(directories))


@app.command()
def add(directory: Path) -> None:
    """Add a directory to the watch list."""
    resolved = directory.resolve()
    if not resolved.is_dir():
        console.print(f"[red]Error:[/] '{directory}' is not a valid directory")
        raise typer.Exit(1)

    dirs = _load_dirs()
    if str(resolved) in dirs:
        console.print(f"[yellow]Already added:[/] {resolved}")
    else:
        dirs.append(str(resolved))
        _save_dirs(dirs)
        console.print(f"[green]Added [cyan]{resolved}[/]")


@app.command()
def index(
    log: Annotated[
        bool, typer.Option("--log", "-l", help="Show detailed logs")
    ] = False,
    extensions: Annotated[
        Optional[list[str]],
        typer.Option("--ext", "-e", help="File extensions to index"),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option("--recursive/--no-recursive", "-r/-R", help="Scan subdirectories"),
    ] = True,
    chunk_size: Annotated[
        int, typer.Option("--chunk-size", "-c", help="Chunk size in tokens")
    ] = 512,
    chunk_overlap: Annotated[
        int, typer.Option("--overlap", "-o", help="Chunk overlap in tokens")
    ] = 50,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Clear and re-index everything")
    ] = False,
) -> None:
    """Index all watched directories."""
    dirs = _load_dirs()

    if not dirs:
        console.print("[yellow]No directories added. Use 'srch add <dir>' first.[/]")
        return

    if force:
        clear()

    processed_exts = []
    if extensions:
        for ext in extensions:
            for e in ext.split(","):
                e = e.strip()
                if e:
                    processed_exts.append(e if e.startswith(".") else f".{e}")
    ext_filter = processed_exts or SUPPORTED_EXTENSIONS

    console.print(f"[bold]Indexing {len(dirs)} directories[/]")
    if log:
        console.print(f"Extensions: {ext_filter}")
        console.print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    total_added, total_skipped = 0, 0
    for d in dirs:
        if log:
            console.rule(d)
        added, skipped = ingest(
            d,
            extensions=ext_filter,
            recursive=recursive,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            log=log,
        )
        total_added += added
        total_skipped += skipped

    console.print(
        f"[green]OK[/] Total: [bold]{total_added}[/] new, [dim]{total_skipped} unchanged[/]"
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of results")] = 5,
) -> None:
    """Search indexed documents."""
    ask(query, limit=limit)


@app.command()
def dirs() -> None:
    """List watched directories."""
    directories = _load_dirs()

    if not directories:
        console.print("[yellow]No directories added[/]")
        return

    table = Table(title="Watched directories", box=box.SIMPLE_HEAVY, pad_edge=False)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Path", style="cyan", overflow="fold")
    for i, d in enumerate(directories, 1):
        table.add_row(str(i), d)
    console.print(table)


@app.command()
def reset() -> None:
    """Clear the entire index."""
    clear()


def main() -> None:
    app()
