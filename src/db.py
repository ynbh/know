import hashlib
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()
client = chromadb.PersistentClient(path="./srch_index")
collection = client.get_or_create_collection(name="documents")

SUPPORTED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx", ".pptx", ".html"]


def ingest(
    directory: str,
    extensions: list[str] | None = None,
    recursive: bool = True,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    log: bool = False,
) -> tuple[int, int]:
    ext_filter = extensions or SUPPORTED_EXTENSIONS

    if log:
        console.log(f"Scanning [cyan]{directory}[/] for {ext_filter}")

    reader = SimpleDirectoryReader(
        input_dir=directory,
        recursive=recursive,
        required_exts=ext_filter,
        filename_as_id=True,
    )

    try:
        documents = reader.load_data()
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        return 0, 0

    if not documents:
        console.print("[yellow]No documents found[/]")
        return 0, 0

    if log:
        console.log(f"Loaded [green]{len(documents)}[/] documents")

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    if log:
        console.log(f"Created [green]{len(nodes)}[/] chunks")

    added, skipped = 0, 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing...", total=len(nodes))

        for node in nodes:
            source_path = node.metadata.get("file_path", "unknown")
            chunk_id = hashlib.md5(f"{source_path}:{node.node_id}".encode()).hexdigest()

            if collection.get(ids=[chunk_id])["ids"]:
                skipped += 1
                progress.advance(task)
                continue

            file_path = Path(source_path)
            stat = file_path.stat() if file_path.exists() else None

            collection.upsert(
                ids=[chunk_id],
                documents=[node.text],
                metadatas=[{
                    "path": source_path,
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "size_bytes": stat.st_size if stat else 0,
                    "chunk_index": node.metadata.get("chunk_index", 0),
                    "node_id": node.node_id,
                }],
            )
            added += 1
            progress.advance(task)

    console.print(f"[green]OK[/] Indexed [bold]{added}[/] new, [dim]{skipped} unchanged[/]")
    return added, skipped


def ask(query: str, limit: int = 5) -> None:
    results = collection.query(query_texts=[query], n_results=limit)

    if not results["documents"][0]:
        console.print("[yellow]No results found[/]")
        return

    def preview_text(text: str, limit_chars: int) -> str:
        cleaned = text.replace("\n", " ")
        return cleaned[:limit_chars] + ("..." if len(cleaned) > limit_chars else "")

    def shorten_middle(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        keep = max_len - 3
        head = keep // 2
        tail = keep - head
        return f"{text[:head]}...{text[-tail:]}"

    def display_path(path: str, max_len: int = 60) -> str:
        try:
            p = Path(path)
            home = Path.home()
            try:
                display = f"~/{p.relative_to(home)}"
            except ValueError:
                display = str(p)
        except Exception:
            display = path
        return shorten_middle(display, max_len)

    table = Table(
        title=f"Results for: [italic]{query}[/]",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        pad_edge=False,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Distance", style="cyan", width=9, justify="right")
    table.add_column("File", style="green", overflow="fold")
    table.add_column("Path", style="dim", overflow="fold", max_width=40)
    table.add_column("Snippet", style="white", overflow="fold", max_width=60)
    
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), 1):
        table.add_row(
            str(i),
            f"{dist:.4f}",
            meta["filename"],
            display_path(meta["path"], 60),
            preview_text(doc, 200),
        )

    console.print(table)

    top_doc = results["documents"][0][0]
    top_meta = results["metadatas"][0][0]
    meta_line = Text(
        f"{display_path(top_meta['path'], 100)}  ·  "
        f"chunk {top_meta.get('chunk_index', 0)}  ·  "
        f"{top_meta.get('size_bytes', 0)} bytes",
        style="dim",
    )
    console.print(Panel(
        Group(meta_line, Text(preview_text(top_doc, 800))),
        title=f"[bold]Top match:[/] {top_meta['filename']}",
        border_style="green",
    ))


def clear():
    global collection
    client.delete_collection(name="documents")
    collection = client.get_or_create_collection(name="documents")
    console.print("[green]Collection cleared")


if __name__ == "__main__":
    ingest(
        "/Users/yashasbhat/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault",
        extensions=[".md"],
        log=True,
    )
