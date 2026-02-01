import json
from pathlib import Path

CACHE_PATH = Path("./know_index/file_cache.json")


def load_file_cache(chunk_size: int, chunk_overlap: int) -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    data = json.loads(CACHE_PATH.read_text())
    config = data.get("config", {})
    if (
        config.get("chunk_size") != chunk_size
        or config.get("chunk_overlap") != chunk_overlap
    ):
        return {}
    files = data.get("files", {})
    if not isinstance(files, dict):
        return {}
    return files


def save_file_cache(
    files: dict[str, dict], chunk_size: int, chunk_overlap: int
) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        "files": files,
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=2))
