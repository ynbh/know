from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SearchItem:
    key: str
    doc: str
    meta: dict
    distance: float | None


@dataclass(frozen=True)
class FusedItem:
    item: SearchItem
    score: float


def rrf_fuse(
    result_lists: Iterable[list[SearchItem]], k: int = 60, limit: int = 5
) -> list[FusedItem]:
    scores: dict[str, dict] = {}
    for items in result_lists:
        for rank, item in enumerate(items, 1):
            entry = scores.get(item.key)
            if entry is None:
                scores[item.key] = {"item": item, "score": 0.0}
            scores[item.key]["score"] += 1.0 / (k + rank)

    fused = [FusedItem(v["item"], v["score"]) for v in scores.values()]
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:limit]
