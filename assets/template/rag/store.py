"""ChromaDB (dense) + BM25 (keyword) + weighted Reciprocal Rank Fusion."""
import re
import threading
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Plus

from config import settings
from rag import bedrock

_tok = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _tok.findall(text.lower())


def rrf_fuse(rankings: list[tuple[float, list[str]]], rrf_k: int) -> list[tuple[str, float]]:
    """rankings: [(weight, [ids best-first]), ...] -> [(id, fused_score)] best-first."""
    scores: dict[str, float] = {}
    for weight, ids in rankings:
        for rank, _id in enumerate(ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridStore:
    def __init__(self):
        self._chroma = chromadb.PersistentClient(
            path=settings.CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.col = self._chroma.get_or_create_collection(
            settings.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        self._lock = threading.RLock()
        self._dirty = True
        self._bm25 = None
        self._ids, self._docs, self._metas, self._toks = [], [], [], []

    # ---------- write ----------
    def add_document(self, filename: str, chunks: list[str]) -> dict:
        doc_id = uuid.uuid4().hex[:12]
        vectors = bedrock.embed_many(chunks)
        with self._lock:
            self.col.add(
                ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=vectors,
                metadatas=[{"doc_id": doc_id, "filename": filename, "chunk_index": i}
                           for i in range(len(chunks))],
            )
            self._dirty = True
        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self.col.delete(where={"doc_id": doc_id})
            self._dirty = True

    def list_documents(self) -> list[dict]:
        with self._lock:
            metas = self.col.get(include=["metadatas"])["metadatas"] or []
        docs: dict[str, dict] = {}
        for m in metas:
            d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "filename": m["filename"], "chunks": 0})
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["filename"].lower())

    def count(self) -> int:
        return self.col.count()

    # ---------- read ----------
    def _ensure_bm25(self):
        with self._lock:
            if not self._dirty:
                return
            data = self.col.get(include=["documents", "metadatas"])
            self._ids = data["ids"] or []
            self._docs = data["documents"] or []
            self._metas = data["metadatas"] or []
            self._toks = [tokenize(d) for d in self._docs]
            self._tokset = [set(t) for t in self._toks]
            self._bm25 = BM25Plus(self._toks) if self._toks else None
            self._dirty = False

    def _dense_ids(self, query: str, n: int) -> list[str]:
        n = min(n, self.col.count())
        if n == 0:
            return []
        vec = bedrock.embed(query)
        res = self.col.query(query_embeddings=[vec], n_results=n, include=[])
        return res["ids"][0]

    def _keyword_ids(self, query: str, n: int) -> list[str]:
        self._ensure_bm25()
        if not self._bm25:
            return []
        q = tokenize(query)
        qset = set(q)
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # BM25Plus gives every doc a small positive score; keep only docs sharing a query term
        return [self._ids[i] for i in order if qset & self._tokset[i]][:n]

    def search(self, query: str, mode: str = "hybrid", k: int | None = None) -> list[dict]:
        k = k or settings.TOP_K
        pool = k * 3
        if mode == "dense":
            fused = [(i, 1.0) for i in self._dense_ids(query, k)]
        elif mode == "keyword":
            fused = [(i, 1.0) for i in self._keyword_ids(query, k)]
        else:
            fused = rrf_fuse([
                (settings.HYBRID_DENSE_WEIGHT, self._dense_ids(query, pool)),
                (settings.HYBRID_KEYWORD_WEIGHT, self._keyword_ids(query, pool)),
            ], settings.RRF_K)[:k]
        if not fused:
            return []
        got = self.col.get(ids=[i for i, _ in fused], include=["documents", "metadatas"])
        by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
        out = []
        for _id, score in fused:
            if _id in by_id:
                text, meta = by_id[_id]
                out.append({"id": _id, "text": text, "filename": meta["filename"],
                            "chunk_index": meta["chunk_index"], "score": round(score, 5)})
        return out


_store = None


def get_store() -> HybridStore:
    global _store
    if _store is None:
        _store = HybridStore()
    return _store
