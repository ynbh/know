from dataclasses import dataclass

from src.retrieval import SearchItem


@dataclass(frozen=True)
class ResultSet:
    query: str
    mode: str
    title: str
    score_label: str
    items: list[SearchItem]


@dataclass(frozen=True)
class BenchmarkResult:
    query: str
    dense: ResultSet
    bm25: ResultSet
