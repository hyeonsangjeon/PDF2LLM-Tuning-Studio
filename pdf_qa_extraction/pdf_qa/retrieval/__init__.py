"""Deterministic, dependency-light retrieval for the PDF-native benchmark.

A pure-Python Okapi BM25 index over source-document elements. No heavy embedding
model and no external service, so the retrieval arms of the benchmark are fully
reproducible and CPU-runnable. Korean text is tokenized as character bigrams (plus
whitespace unigrams), which recovers far more agglutinative Korean overlap than
whitespace tokens alone.

This is a **core** utility (``pdf_qa``) — it must not import ``workflows``.

    from pdf_qa.retrieval import BM25Index, Retriever
    idx = BM25Index.build(corpus)          # corpus: [{"element_id","text","page"?}]
    hits = Retriever(idx).search("질문", k=5)   # -> [Hit(element_id, score, ...)]
"""

from .index import BM25Index, index_config_hash
from .retrieve import Hit, Retriever

__all__ = ["BM25Index", "Retriever", "Hit", "index_config_hash"]
