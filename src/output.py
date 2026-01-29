import json
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models import BenchmarkResult, ResultSet

console = Console()


def _preview_text(text: str, limit_chars: int) -> str:
    cleaned = text.replace("\n", " ")
    return cleaned[:limit_chars] + ("..." if len(cleaned) > limit_chars else "")


def _shorten_middle(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    keep = max_len - 3
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}...{text[-tail:]}"


def _display_path(path: str, max_len: int = 60) -> str:
    p = Path(path)
    home = Path.home()
    if p.is_absolute() and home in p.parents:
        display = f"~/{p.relative_to(home)}"
    else:
        display = str(p)
    return _shorten_middle(display, max_len)


def _result_set_json(result: ResultSet) -> dict:
    return {
        "query": result.query,
        "mode": result.mode,
        "title": result.title,
        "results": [
            {
                "rank": i,
                "score": f"{item.distance:.4f}" if item.distance is not None else "n/a",
                "filename": item.meta.get("filename"),
                "path": item.meta.get("path"),
                "snippet": _preview_text(item.doc, 200),
                "chunk_index": item.meta.get("chunk_index", 0),
                "size_bytes": item.meta.get("size_bytes", 0),
            }
            for i, item in enumerate(result.items, 1)
        ],
    }


def _write_json_file(payload: dict, json_out: Path | None) -> None:
    if json_out is None:
        return
    json_out.write_text(json.dumps(payload, ensure_ascii=True))


def _render_plain(result: ResultSet) -> None:
    if not result.items:
        print("No results found")
        return
    print(result.title)
    for i, item in enumerate(result.items, 1):
        print(
            f"{i}. {result.score_label}={item.distance:.4f} | "
            f"{item.meta.get('filename')} | {item.meta.get('path')}"
        )
        print(f"    {_preview_text(item.doc, 200)}")
    top = result.items[0]
    print(
        f"Top match: {top.meta.get('filename')} | {top.meta.get('path')} | "
        f"chunk {top.meta.get('chunk_index', 0)} | {top.meta.get('size_bytes', 0)} bytes"
    )
    print(_preview_text(top.doc, 800))


def _render_rich(result: ResultSet) -> None:
    if not result.items:
        console.print("[yellow]No results found[/]")
        return

    table = Table(
        title=result.title,
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        pad_edge=False,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column(result.score_label, style="cyan", width=9, justify="right")
    table.add_column("File", style="green", overflow="fold")
    table.add_column("Path", style="dim", overflow="fold", max_width=40)
    table.add_column("Snippet", style="white", overflow="fold", max_width=60)

    for i, item in enumerate(result.items, 1):
        table.add_row(
            str(i),
            f"{item.distance:.4f}" if item.distance is not None else "n/a",
            item.meta["filename"],
            _display_path(item.meta["path"], 60),
            _preview_text(item.doc, 200),
        )

    console.print(table)

    top_doc = result.items[0].doc
    top_meta = result.items[0].meta
    meta_line = Text(
        f"{_display_path(top_meta['path'], 100)}  ·  "
        f"chunk {top_meta.get('chunk_index', 0)}  ·  "
        f"{top_meta.get('size_bytes', 0)} bytes",
        style="dim",
    )
    console.print(
        Panel(
            Group(meta_line, Text(_preview_text(top_doc, 800))),
            title=f"[bold]Top match:[/] {top_meta['filename']}",
            border_style="green",
        )
    )


def render_response(
    response: ResultSet | BenchmarkResult,
    output: str,
    json_out: Path | None,
) -> None:
    if isinstance(response, BenchmarkResult):
        payload = {
            "query": response.query,
            "dense": _result_set_json(response.dense)["results"],
            "bm25": _result_set_json(response.bm25)["results"],
        }
        _write_json_file(payload, json_out)
        if output == "json":
            print(json.dumps(payload, ensure_ascii=True))
            return
        _render_rich(response.dense) if output == "rich" else _render_plain(
            response.dense
        )
        _render_rich(response.bm25) if output == "rich" else _render_plain(
            response.bm25
        )
        return

    payload = _result_set_json(response)
    _write_json_file(payload, json_out)
    if output == "json":
        print(json.dumps(payload, ensure_ascii=True))
        return
    if output == "plain":
        _render_plain(response)
        return
    _render_rich(response)
