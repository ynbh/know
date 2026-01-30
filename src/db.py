import fnmatch
import hashlib
import shutil
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from src.bm25 import BM25Index, load_cached_index, save_cached_index
from src.models import BenchmarkResult, ResultSet
from src.retrieval import SearchItem, rrf_fuse

console = Console()
client = chromadb.PersistentClient(path="./know_index")

dense_collection = client.get_or_create_collection(name="documents")

SUPPORTED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx", ".pptx", ".html"]


def has_index() -> bool:
    try:
        return dense_collection.count() > 0
    except Exception:
        return False


def ingest(
    directory: str,
    extensions: list[str] | None = None,
    include_globs: list[str] | None = None,
    since_timestamp: float | None = None,
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

    base_dir = Path(directory)
    if include_globs:
        filtered: list = []
        base_resolved = base_dir.resolve()
        for doc in documents:
            source_path = doc.metadata.get("file_path", "")
            source_resolved = Path(source_path).resolve()
            if source_resolved.is_relative_to(base_resolved):
                rel_str = source_resolved.relative_to(base_resolved).as_posix()
            else:
                rel_str = source_resolved.name
            if any(
                fnmatch.fnmatch(rel_str, pattern) for pattern in include_globs
            ) or any(
                fnmatch.fnmatch(source_path, pattern) for pattern in include_globs
            ):
                filtered.append(doc)
        documents = filtered

    if since_timestamp is not None:
        filtered = []
        for doc in documents:
            source_path = doc.metadata.get("file_path", "")
            mtime = Path(source_path).stat().st_mtime
            if mtime >= since_timestamp:
                filtered.append(doc)
        documents = filtered

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
    batch_size = 100

    # prepare all chunks first, checking which already exist
    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_metas: list[dict] = []
    node_data: list[tuple[str, str, dict]] = []

    for node in nodes:
        source_path = node.metadata.get("file_path", "unknown")
        chunk_index = node.metadata.get("chunk_index", 0)
        chunk_id = hashlib.md5(
            f"{source_path}:{chunk_index}:{node.text}".encode()
        ).hexdigest()
        file_path = Path(source_path)
        stat = file_path.stat() if file_path.exists() else None
        doc_text = f"{file_path.name}\n\n{node.text}"
        meta = {
            "path": source_path,
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size_bytes": stat.st_size if stat else 0,
            "chunk_index": chunk_index,
            "node_id": node.node_id,
        }
        node_data.append((chunk_id, doc_text, meta))

    # batch check for existing ids
    all_ids = [nd[0] for nd in node_data]
    existing_ids: set[str] = set()
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        result = dense_collection.get(ids=batch_ids)
        existing_ids.update(result["ids"])

    # filter to only new chunks
    for chunk_id, doc_text, meta in node_data:
        if chunk_id in existing_ids:
            skipped += 1
        else:
            pending_ids.append(chunk_id)
            pending_docs.append(doc_text)
            pending_metas.append(meta)

    if log:
        console.log(f"Skipping {skipped} existing, adding {len(pending_ids)} new")

    with Progress(
        TextColumn("[bold cyan]Indexing[/]"),
        BarColumn(
            bar_width=32,
            complete_style="green",
            finished_style="green",
        ),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TextColumn(
            "[dim]added {task.fields[added]} · skipped {task.fields[skipped]}[/]"
        ),
        console=console,
        transient=False,
    ) as progress:
        total_batches = (len(pending_ids) + batch_size - 1) // batch_size
        task = progress.add_task("", total=total_batches, added=0, skipped=skipped)

        # batch upsert for embeddings
        for i in range(0, len(pending_ids), batch_size):
            batch_ids = pending_ids[i : i + batch_size]
            batch_docs = pending_docs[i : i + batch_size]
            batch_metas = pending_metas[i : i + batch_size]

            dense_collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            added += len(batch_ids)
            progress.update(task, advance=1, added=added, skipped=skipped)

    console.print(
        f"[green]OK[/] Indexed [bold]{added}[/] new, [dim]{skipped} unchanged[/]"
    )
    return added, skipped


def _filter_items(
    items: list[SearchItem],
    include_globs: list[str] | None,
    since_timestamp: float | None,
) -> list[SearchItem]:
    if not include_globs and since_timestamp is None:
        return items

    filtered: list[SearchItem] = []
    for item in items:
        source_path = item.meta.get("path", "")
        if include_globs:
            path_str = Path(source_path).as_posix()
            base_name = Path(source_path).name
            if not any(
                fnmatch.fnmatch(path_str, pattern)
                or fnmatch.fnmatch(base_name, pattern)
                for pattern in include_globs
            ):
                continue
        if since_timestamp is not None:
            path = Path(source_path)
            if not path.exists():
                continue
            mtime = path.stat().st_mtime
            if mtime < since_timestamp:
                continue
        filtered.append(item)
    return filtered


def _query_items(
    collection: chromadb.Collection,
    query: str,
    limit: int,
    include_globs: list[str] | None,
    since_timestamp: float | None,
) -> list[SearchItem]:
    results = collection.query(query_texts=[query], n_results=limit)
    if not results["documents"][0]:
        return []

    items: list[SearchItem] = []
    for doc, meta, dist, item_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        items.append(
            SearchItem(
                key=item_id,
                doc=doc,
                meta=meta,
                distance=dist,
            )
        )

    return _filter_items(items, include_globs, since_timestamp)


def _bm25_query_items(
    query: str,
    limit: int,
    include_globs: list[str] | None,
    since_timestamp: float | None,
) -> list[SearchItem]:
    doc_count = dense_collection.count()
    if doc_count == 0:
        return []

    cache = load_cached_index(doc_count)
    ids: list[str]
    if cache is None:
        docs = dense_collection.get(include=["documents"])
        if not docs["documents"]:
            return []
        index = BM25Index.from_documents(
            ids=docs["ids"],
            documents=docs["documents"],
            metadatas=[],
        )
        retriever = index.bm25
        ids = docs["ids"]
        save_cached_index(retriever, ids)
    else:
        retriever, ids = cache

    extra = max(limit * 3, 20)
    query_index = BM25Index(ids=ids, documents=[], metadatas=[], bm25=retriever)
    ranked_ids, scores = query_index.query_ids(query, limit=min(extra, len(ids)))
    if not ranked_ids:
        return []

    fetched = dense_collection.get(ids=ranked_ids, include=["documents", "metadatas"])
    by_id = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"]
        )
    }
    items: list[SearchItem] = []
    for doc_id, score in zip(ranked_ids, scores):
        payload = by_id.get(doc_id)
        if payload is None:
            continue
        doc, meta = payload
        items.append(
            SearchItem(
                key=doc_id,
                doc=doc,
                meta=meta,
                distance=score,
            )
        )

    items = _filter_items(items, include_globs, since_timestamp)
    return items[:limit]


def search(
    query: str,
    limit: int = 5,
    include_globs: list[str] | None = None,
    since_timestamp: float | None = None,
    mode: str = "dense",
    benchmark: bool = False,
) -> ResultSet | BenchmarkResult:
    needs_filter = bool(include_globs) or since_timestamp is not None
    base_limit = max(limit, 10) if benchmark else limit
    candidate_limit = max(base_limit * 3, 20) if needs_filter else base_limit
    dense_items = _query_items(
        dense_collection,
        query,
        candidate_limit,
        include_globs,
        since_timestamp,
    )
    if len(dense_items) > limit:
        dense_items = dense_items[:limit]

    if benchmark:
        bm25_items = _bm25_query_items(
            query, max(limit, 10), include_globs, since_timestamp
        )
        dense_set = ResultSet(
            query=query,
            mode="dense",
            title=f"Dense results: {query}",
            score_label="Distance",
            items=dense_items,
        )
        bm25_set = ResultSet(
            query=query,
            mode="bm25",
            title=f"BM25 results: {query}",
            score_label="Score",
            items=bm25_items,
        )
        return BenchmarkResult(query=query, dense=dense_set, bm25=bm25_set)

    if mode == "bm25":
        bm25_items = _bm25_query_items(query, limit, include_globs, since_timestamp)
        return ResultSet(
            query=query,
            mode="bm25",
            title=f"BM25 results: {query}",
            score_label="Score",
            items=bm25_items,
        )

    if mode == "hybrid":
        dense_items = _query_items(
            dense_collection, query, max(limit * 3, 20), include_globs, since_timestamp
        )
        bm25_items = _bm25_query_items(
            query, max(limit * 3, 20), include_globs, since_timestamp
        )
        fused = rrf_fuse([dense_items, bm25_items], limit=limit)
        fused_items = [
            SearchItem(
                key=f.item.key,
                doc=f.item.doc,
                meta=f.item.meta,
                distance=f.score,
            )
            for f in fused
        ]
        return ResultSet(
            query=query,
            mode="hybrid",
            title=f"Hybrid results: {query}",
            score_label="RRF",
            items=fused_items,
        )

    return ResultSet(
        query=query,
        mode="dense",
        title=f"Results for: {query}",
        score_label="Distance",
        items=dense_items,
    )


def clear():
    global dense_collection
    client.delete_collection(name="documents")
    dense_collection = client.get_or_create_collection(name="documents")
    shutil.rmtree("./know_index/bm25", ignore_errors=True)
    console.print("[green]Collection cleared")


if __name__ == "__main__":
    ingest(
        "/Users/yashasbhat/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault",
        extensions=[".md"],
        log=True,
    )
