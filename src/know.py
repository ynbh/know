import sys
import typer
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from typing_extensions import Annotated
from rich import box
from rich.console import Console
from rich.table import Table

from src.db import ingest, search as run_search, clear, SUPPORTED_EXTENSIONS, has_index, IndexReport
from src.output import render_response

app = typer.Typer(help="know - semantic search CLI")
console = Console()

INDEX_FILE = Path.home() / ".know_dirs"


def _load_dirs() -> list[str]:
    if INDEX_FILE.exists():
        return [d.strip() for d in INDEX_FILE.read_text().splitlines() if d.strip()]
    return []


def _save_dirs(directories: list[str]) -> None:
    INDEX_FILE.write_text("\n".join(directories))


def _parse_since(since: Optional[str]) -> float | None:
    if not since:
        return None
    value = since.strip()
    if not value:
        return None
    unit_map = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = value[-1].lower()
    if unit in unit_map and value[:-1].isdigit():
        delta = int(value[:-1]) * unit_map[unit]
        return (datetime.now() - timedelta(seconds=delta)).timestamp()
    if "T" in value:
        dt = datetime.fromisoformat(value)
    else:
        dt = datetime.strptime(value, "%Y-%m-%d")
    return dt.timestamp()


def _parse_globs(globs: Optional[list[str]]) -> list[str]:
    processed: list[str] = []
    if not globs:
        return processed
    for pattern in globs:
        for g in pattern.split(","):
            g = g.strip()
            if g:
                processed.append(g)
    return processed


def _ensure_index_ready() -> None:
    if has_index():
        return
    console.print("[yellow]No index found.[/]")
    console.print("To get started, run:")
    console.print("  know add <dir>")
    console.print("  know index")
    raise typer.Exit(1)


def _maybe_prefix_search(args: list[str]) -> list[str]:
    if len(args) < 2:
        return args
    first = args[1]
    if first in {"add", "remove", "index", "search", "dirs", "reset", "--help", "-h"}:
        return args
    if first.startswith("-"):
        return args
    return [args[0], "search", *args[1:]]


def _run_search(
    query: str,
    limit: int,
    globs: Optional[list[str]],
    since: Optional[str],
    bm25: bool,
    hybrid: bool,
    benchmark: bool,
    plain: bool,
    json_out: Optional[Path],
    json_stdout: bool,
) -> None:
    processed_globs = _parse_globs(globs)
    try:
        since_ts = _parse_since(since)
    except ValueError:
        console.print("[red]Error:[/] --since must be like 7d, 12h, 30m, or 2024-01-15")
        raise typer.Exit(1)
    if sum([bm25, hybrid]) > 1:
        console.print("[red]Error:[/] Choose only one of --bm25 or --hybrid")
        raise typer.Exit(1)

    mode = "dense"
    if bm25:
        mode = "bm25"
    if hybrid:
        mode = "hybrid"

    if plain and json_stdout:
        console.print("[red]Error:[/] Choose only one of --plain or --json")
        raise typer.Exit(1)

    output = "rich"
    if plain:
        output = "plain"
    if json_stdout:
        output = "json"

    _ensure_index_ready()
    response = run_search(
        query,
        limit=limit,
        include_globs=processed_globs,
        since_timestamp=since_ts,
        mode=mode,
        benchmark=benchmark,
    )
    render_response(response, output=output, json_out=json_out)


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
def remove(directory: Path) -> None:
    """Remove a directory from the watch list."""
    resolved = directory.resolve()
    dirs = _load_dirs()

    if str(resolved) in dirs:
        dirs.remove(str(resolved))
        _save_dirs(dirs)
        console.print(f"[green]Removed [cyan]{resolved}[/]")
    else:
        console.print(f"[yellow]Not in watchlist:[/] {resolved}")


@app.command()
def index(
    log: Annotated[
        bool, typer.Option("--log", "-l", help="Show detailed logs")
    ] = False,
    extensions: Annotated[
        Optional[list[str]],
        typer.Option("--ext", "-e", help="File extensions to index"),
    ] = None,
    globs: Annotated[
        Optional[list[str]],
        typer.Option(
            "--glob",
            "-g",
            help="Include only files matching glob patterns (e.g. **/*.md, notes/**)",
        ),
    ] = None,
    since: Annotated[
        Optional[str],
        typer.Option(
            "--since",
            help="Only index files modified since (e.g. 7d, 12h, 30m, 2024-01-15)",
        ),
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
    report: Annotated[
        Optional[Path], typer.Option("--report", help="Write skip report JSON to file")
    ] = None,
) -> None:
    """Index all watched directories."""
    dirs = _load_dirs()

    if not dirs:
        console.print("[yellow]No directories added. Use 'know add <dir>' first.[/]")
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

    processed_globs = _parse_globs(globs)

    try:
        since_ts = _parse_since(since)
    except ValueError:
        console.print("[red]Error:[/] --since must be like 7d, 12h, 30m, or 2024-01-15")
        raise typer.Exit(1)

    console.print(f"[bold]Indexing {len(dirs)} directories[/]")
    if log:
        console.print(f"Extensions: {ext_filter}")
        if processed_globs:
            console.print(f"Globs: {processed_globs}")
        if since_ts is not None:
            console.print(f"Since: {since}")
        console.print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    total_added, total_skipped = 0, 0
    all_skip_entries: list = []
    for d in dirs:
        if log:
            console.rule(d)
        result = ingest(
            d,
            extensions=ext_filter,
            include_globs=processed_globs,
            since_timestamp=since_ts,
            recursive=recursive,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            log=log,
            report=report is not None,
        )
        if isinstance(result, IndexReport):
            total_added += result.added
            total_skipped += result.skipped
            all_skip_entries.extend(result.skip_entries)
        else:
            total_added += result[0]
            total_skipped += result[1]

    console.print(
        f"[green]OK[/] Total: [bold]{total_added}[/] new, [dim]{total_skipped} unchanged[/]"
    )

    if report is not None:
        combined = IndexReport(
            added=total_added,
            skipped=total_skipped,
            skip_entries=all_skip_entries,
        )
        report.write_text(combined.to_json())
        console.print(f"[dim]Report written to {report}[/]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of results")] = 5,
    globs: Annotated[
        Optional[list[str]],
        typer.Option(
            "--glob",
            "-g",
            help="Include only files matching glob patterns (e.g. **/*.md, notes/**)",
        ),
    ] = None,
    since: Annotated[
        Optional[str],
        typer.Option(
            "--since",
            help="Only include files modified since (e.g. 7d, 12h, 30m, 2024-01-15)",
        ),
    ] = None,
    bm25: Annotated[
        bool, typer.Option("--bm25", help="Use BM25 lexical search")
    ] = False,
    hybrid: Annotated[
        bool, typer.Option("--hybrid", help="Use hybrid BM25 + vector search")
    ] = False,
    benchmark: Annotated[
        bool,
        typer.Option(
            "--benchmark",
            help="Show dense vs BM25 results for comparison",
        ),
    ] = False,
    plain: Annotated[
        bool, typer.Option("--plain", help="Render results as plain text")
    ] = False,
    json_out: Annotated[
        Optional[Path],
        typer.Option("--json-out", help="Write JSON results to a file"),
    ] = None,
    json_stdout: Annotated[
        bool, typer.Option("--json", help="Render results as JSON")
    ] = False,
) -> None:
    """Search indexed documents."""
    _run_search(
        query=query,
        limit=limit,
        globs=globs,
        since=since,
        bm25=bm25,
        hybrid=hybrid,
        benchmark=benchmark,
        plain=plain,
        json_out=json_out,
        json_stdout=json_stdout,
    )


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
    sys.argv = _maybe_prefix_search(sys.argv)
    app()
