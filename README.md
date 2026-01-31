# know

`know` is a small semantic search CLI for local files. It watches directories,
chunks documents, and stores embeddings in a local Chroma index so you can
query your notes, docs, or code quickly from the terminal.

## What it does

- Add directories to a local watch list
- Index supported file types into a local vector store
- Search with a short CLI command and view ranked results

Supported extensions: `.md`, `.txt`, `.pdf`, `.docx`, `.pptx`, `.html`,
`.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.go`, `.rs`, `.java`, `.c`, `.cpp`,
`.h`, `.hpp`, `.rb`, `.sh`, `.lua`, `.swift`.

## Install

Requires Python 3.13+ and `uv`.

```bash
uv sync
```

## Quick start

```bash
uv run know add ~/Documents/notes
uv run know index
uv run know "retrieval augmented generation"
```

## Commands

```bash
know add <dir>
know remove <dir>
know index [--log] [--ext .md --ext .txt] [--recursive/--no-recursive] \
  [--chunk-size 512] [--overlap 50] [--force] [--dry] [--glob "**/*.md"] \
  [--since 7d] [--report report.json]
know search <query> [--limit 5] [--glob "**/*.md"] [--since 7d] \
  [--bm25 | --hybrid] [--benchmark] [--plain | --json] [--json-out results.json]
know <query> [--limit 5] [--glob "**/*.md"] [--since 7d] \
  [--bm25 | --hybrid] [--benchmark] [--plain | --json] [--json-out results.json]
know dirs
know reset
```

### Tips

- Use `--log` when debugging what is being scanned and chunked.
- Use `--ext` to focus on a small set of file types for faster indexing.
- Use `--glob` to narrow indexing to matching paths (e.g. `notes/**`).
- Use `--since` with `7d`, `12h`, or `2024-01-15` to skip older files.
- Use `--force` to clear and rebuild the index from scratch.
- Use `--dry` to preview how many chunks would be added without writing.
- Use `--bm25` for lexical search, or `--hybrid` for BM25 + vector fusion.
- BM25 search builds a lightweight index from stored chunks and caches it under `know_index/bm25`.
- Use `--benchmark` to compare dense vs BM25 results side-by-side.
- Use `--plain` for plain-text output, `--json` for JSON output, and `--json-out` to save JSON to a file.
- Use `--report` to capture skipped chunks (already indexed or duplicate content).

## How it works

`know` reads files with `llama-index`, splits them into chunks with a sentence
splitter, and stores them in a local Chroma collection under `./know_index`.
Search results show a ranked table plus a detailed preview of the top match.

Directories are tracked in `~/.know_dirs`. BM25 search builds a cached index
under `./know_index/bm25`.

## Development

```bash
uv sync --dev
```
