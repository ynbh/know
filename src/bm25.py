import json
from dataclasses import dataclass
from pathlib import Path

import bm25s

from src.retrieval import SearchItem

BM25_CACHE_DIR = Path("./know_index/bm25")
BM25_META_PATH = BM25_CACHE_DIR / "meta.json"
BM25_IDS_PATH = BM25_CACHE_DIR / "ids.json"

import Stemmer

_STEMMER = Stemmer.Stemmer("english")


@dataclass
class BM25Index:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    bm25: bm25s.BM25

    @classmethod
    def from_documents(
        cls, ids: list[str], documents: list[str], metadatas: list[dict]
    ) -> "BM25Index":
        corpus_tokens = bm25s.tokenize(
            documents,
            stopwords="en",
            stemmer=_STEMMER,
            show_progress=False,
        )
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        return cls(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            bm25=retriever,
        )

    def query_ids(self, query: str, limit: int) -> tuple[list[str], list[float]]:
        if not self.ids:
            return [], []
        query_tokens = bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=_STEMMER,
            show_progress=False,
        )
        results = self.bm25.retrieve(
            query_tokens,
            corpus=self.ids,
            k=min(limit, len(self.ids)),
            show_progress=False,
        )
        ids = list(results.documents[0])
        scores = [float(score) for score in results.scores[0]]
        return ids, scores


def load_cached_index(expected_count: int) -> tuple[bm25s.BM25, list[str]] | None:
    if (
        not BM25_CACHE_DIR.exists()
        or not BM25_META_PATH.exists()
        or not BM25_IDS_PATH.exists()
    ):
        return None
    meta = json.loads(BM25_META_PATH.read_text())
    if meta.get("count") != expected_count:
        return None
    retriever = bm25s.BM25.load(str(BM25_CACHE_DIR), load_corpus=False)
    ids = json.loads(BM25_IDS_PATH.read_text())
    if not isinstance(ids, list) or len(ids) != expected_count:
        return None
    return retriever, ids


def save_cached_index(retriever: bm25s.BM25, ids: list[str]) -> None:
    BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(str(BM25_CACHE_DIR))
    BM25_IDS_PATH.write_text(json.dumps(ids))
    BM25_META_PATH.write_text(json.dumps({"count": len(ids)}))
