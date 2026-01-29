# srch

`srch` is a small semantic search CLI for local files. It watches directories,
chunks documents, and stores embeddings in a local Chroma index so you can
query your notes or docs quickly from the terminal.

## What it does

- Add directories to a local watch list
- Index supported file types into a local vector store
- Search with a short CLI command and view ranked results

Supported extensions: `.md`, `.txt`, `.pdf`, `.docx`, `.pptx`, `.html`.

## Install

Requires Python 3.13+ and `uv`.

```bash
uv sync
```

## Quick start

```bash
srch add ~/Documents/notes
srch index
srch search "retrieval augmented generation"
```

## Commands

```bash
srch add <dir>
srch index [--log] [--ext .md --ext .txt] [--recursive/--no-recursive] \
  [--chunk-size 512] [--overlap 50] [--force] [--glob "**/*.md"] \
  [--since 7d]
srch search <query> [--limit 5] [--glob "**/*.md"] [--since 7d] \
  [--bm25 | --hybrid] [--benchmark] [--plain | --json] [--json-out results.json]
srch dirs
srch reset
```

### Tips

- Use `--log` when debugging what is being scanned and chunked.
- Use `--ext` to focus on a small set of file types for faster indexing.
- Use `--glob` to narrow indexing to matching paths (e.g. `notes/**`).
- Use `--since` with `7d`, `12h`, or `2024-01-15` to skip older files.
- Use `--force` to clear and rebuild the index from scratch.
- Use `--bm25` for lexical search, or `--hybrid` for BM25 + vector fusion.
- BM25 search builds a lightweight index from stored chunks and caches it under `srch_index/bm25`.
- Use `--benchmark` to compare dense vs BM25 results side-by-side.
- Use `--plain` for plain-text output, `--json` for JSON output, and `--json-out` to save JSON to a file.

## How it works

`srch` reads files with `llama-index`, splits them into chunks with a sentence
splitter, and stores them in a local Chroma collection under `./srch_index`.
Search results show a ranked table plus a detailed preview of the top match.

## Development

```bash
uv sync --dev
```
