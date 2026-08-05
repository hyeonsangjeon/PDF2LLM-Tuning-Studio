"""BM25 query-time scoring over a :class:`~pdf_qa.retrieval.index.BM25Index`."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .index import BM25Index, tokenize


@dataclass
class Hit:
    element_id: str
    score: float
    rank: int
    meta: Dict[str, Any]
    text: Optional[str] = None


class Retriever:
    """Deterministic BM25 retriever. Ties break by ascending element_id, so identical
    queries always return an identical ordering."""

    def __init__(self, index: BM25Index):
        self.index = index

    def score(self, query: str) -> List[float]:
        idx = self.index
        q_terms = list(Counter(tokenize(query, idx.tokenizer)))
        scores = [0.0] * len(idx.element_ids)
        for i, toks in enumerate(idx.doc_tokens):
            if not toks:
                continue
            tf = Counter(toks)
            dl = idx.doc_len[i] or 1
            denom_norm = idx.k1 * (1 - idx.b + idx.b * dl / (idx.avg_len or 1.0))
            s = 0.0
            for t in q_terms:
                f = tf.get(t)
                if not f:
                    continue
                s += idx.idf.get(t, 0.0) * (f * (idx.k1 + 1)) / (f + denom_norm)
            scores[i] = s
        return scores

    def search(self, query: str, k: int = 5, *, include_text: bool = False) -> List[Hit]:
        scores = self.score(query)
        order = sorted(range(len(scores)),
                       key=lambda i: (-scores[i], self.index.element_ids[i]))
        hits: List[Hit] = []
        for rank, i in enumerate(order[:k]):
            if scores[i] <= 0.0:
                break  # never return non-matching docs as spurious hits
            meta = dict(self.index.docs_meta[i])
            text = None
            if include_text:
                text = " ".join(self.index.doc_tokens[i])
            hits.append(Hit(self.index.element_ids[i], round(scores[i], 6), rank, meta, text))
        return hits

    def retrieve_ids(self, query: str, k: int = 5) -> List[str]:
        return [h.element_id for h in self.search(query, k)]
