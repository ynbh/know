import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class FileContent:
    path: str
    content: str | None
    error: str | None = None


def collect_files(
    directory: str, name_filter: str = "", recursive: bool = True
) -> list[str]:
    files: list[str] = []
    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if not name_filter or name_filter in filename:
                    files.append(os.path.join(root, filename))
    else:
        for entry in os.scandir(directory):
            if entry.is_file() and (not name_filter or name_filter in entry.name):
                files.append(entry.path)
    return files


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_files_parallel(
    file_paths: list[str], max_workers: int | None = None
) -> list[FileContent]:
    def read_single(path: str) -> FileContent:
        try:
            return FileContent(path=path, content=read_file(path))
        except Exception as e:
            return FileContent(path=path, content=None, error=str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(read_single, file_paths))


def collect_and_read_files(
    directory: str,
    name_filter: str = "",
    recursive: bool = True,
    max_workers: int | None = None,
) -> list[FileContent]:
    return read_files_parallel(
        collect_files(directory, name_filter, recursive), max_workers
    )
