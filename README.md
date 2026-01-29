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
  [--chunk-size 512] [--overlap 50] [--force]
srch search <query> [--limit 5]
srch dirs
srch reset
```

### Tips

- Use `--log` when debugging what is being scanned and chunked.
- Use `--ext` to focus on a small set of file types for faster indexing.
- Use `--force` to clear and rebuild the index from scratch.

## How it works

`srch` reads files with `llama-index`, splits them into chunks with a sentence
splitter, and stores them in a local Chroma collection under `./srch_index`.
Search results show a ranked table plus a detailed preview of the top match.

## Development

```bash
uv sync --dev
```
