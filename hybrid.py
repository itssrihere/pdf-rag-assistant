"""
Hybrid retrieval: BM25 keyword ranking fused with vector similarity by reciprocal rank fusion (RRF).

Vector search matches meaning but can miss exact rare strings (gene names, SNP IDs, catalog numbers);
BM25 matches exact words but not paraphrases. Each list is ranked independently, then RRF merges them:
a chunk's score is the sum of 1 / (RRF_K + rank) over the lists it appears in.

The BM25 index is held in memory (built from every chunk in the collection), which is fine for thousands
of chunks; a much larger corpus would call for Postgres full-text search or a dedicated search engine.
"""

import re
from collections import defaultdict

import psycopg
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config

RRF_K = 60  # standard RRF damping constant
TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it of on or that the this to was were what when "
    "where which who why with did does do into than then these those their there".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def rrf_fuse(rankings: list[list[str]], top: int | None = None, k: int = RRF_K,
             weights: list[float] | None = None) -> list[str]:
    """Merges ranked lists of ids, optionally weighting each list.
    Ties go to whichever id appeared earliest in the first list it was in."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):  # one weight per ranking
        for rank, key in enumerate(ranking, start=1):
            scores[key] += weight / (k + rank)
            first_seen.setdefault(key, len(first_seen))
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return ordered[:top]


class KeywordIndex:
    """BM25 over chunk text. `chunks` is a list of (id, Document)."""

    def __init__(self, chunks: list[tuple[str, Document]]):
        self.chunks = dict(chunks)
        self.ids = [chunk_id for chunk_id, _ in chunks]
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for _, doc in chunks])

    def search(self, query: str, n: int) -> list[str]:
        """Ids of the n best-matching chunks; chunks sharing no query word are never returned."""
        terms = tokenize(query)
        if not terms:
            return []
        scores = self.bm25.get_scores(terms)
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self.ids[i] for i in best if scores[i] > 0]


def load_collection(collection: str | None = None) -> list[tuple[str, Document]]:
    """Reads every chunk of a collection straight from Postgres."""
    url = config.DB_URL.replace("postgresql+psycopg://", "postgresql://")
    query = (
        "select e.id, e.document, e.cmetadata from langchain_pg_embedding e "
        "join langchain_pg_collection c on c.uuid = e.collection_id where c.name = %s order by e.id"
    )
    with psycopg.connect(url) as conn:
        rows = conn.execute(query, (collection or config.COLLECTION_NAME,)).fetchall()
    return [(id_, Document(id=id_, page_content=text, metadata=meta or {})) for id_, text, meta in rows]


class HybridRetriever:
    """`store` is the pgvector store; `index` the matching KeywordIndex."""

    def __init__(self, store, index: KeywordIndex, fetch_k: int = 50, bm25_weight: float = 1.0):
        self.store, self.index, self.fetch_k, self.bm25_weight = store, index, fetch_k, bm25_weight

    def search(self, question: str, k: int, mode: str = "hybrid") -> list[Document]:
        """mode: 'hybrid' (fused), 'bm25' (keyword only) or 'dense' (vector only)."""
        dense_docs = self.store.similarity_search(question, k=self.fetch_k) if mode != "bm25" else []
        by_id = {doc.id: doc for doc in dense_docs}
        dense = [doc.id for doc in dense_docs]
        keyword = self.index.search(question, self.fetch_k) if mode != "dense" else []
        for chunk_id in keyword:
            by_id.setdefault(chunk_id, self.index.chunks[chunk_id])
        return [by_id[chunk_id] for chunk_id in rrf_fuse([dense, keyword], top=k, weights=[1.0, self.bm25_weight])]

    @classmethod
    def from_store(cls, store, fetch_k: int = 50, bm25_weight: float = 1.0) -> "HybridRetriever":
        return cls(store, KeywordIndex(load_collection()), fetch_k, bm25_weight)
