"""Okapi BM25 index over source-document elements (deterministic, pure Python)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

# BM25 free parameters (Okapi). Pinned so the ranking is reproducible + hashable.
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_TOKENIZER = "char2+ws"   # char bigrams + whitespace unigrams

_PUNC = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~·…“”‘’「」『』（）]")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _PUNC.sub(" ", text)
    return " ".join(text.split()).lower()


def tokenize(text: str, mode: str = DEFAULT_TOKENIZER) -> List[str]:
    """Deterministic tokenizer. ``char2+ws`` = whitespace unigrams + char bigrams over
    the whitespace-removed normalized string (good Korean lexical recall)."""
    norm = _normalize(text)
    toks: List[str] = []
    if "ws" in mode:
        toks.extend(t for t in norm.split() if t)
    if "char2" in mode:
        compact = norm.replace(" ", "")
        toks.extend(compact[i:i + 2] for i in range(len(compact) - 1))
    return toks


def index_config_hash(tokenizer: str, k1: float, b: float,
                      doc_ids: Sequence[str]) -> str:
    """Stable hash of the retrieval config + corpus identity (fairness-contract record)."""
    payload = json.dumps({"tokenizer": tokenizer, "k1": k1, "b": b,
                          "doc_ids": sorted(doc_ids)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BM25Index:
    """An immutable BM25 index. Build once, then query with :class:`Retriever`."""
    element_ids: List[str]
    docs_meta: List[Dict[str, Any]]
    doc_tokens: List[List[str]]
    df: Dict[str, int]
    idf: Dict[str, float]
    doc_len: List[int]
    avg_len: float
    tokenizer: str = DEFAULT_TOKENIZER
    k1: float = BM25_K1
    b: float = BM25_B
    config_hash: str = field(default="")

    @classmethod
    def build(cls, corpus: Sequence[Dict[str, Any]], *, tokenizer: str = DEFAULT_TOKENIZER,
              k1: float = BM25_K1, b: float = BM25_B) -> "BM25Index":
        """corpus: list of {"element_id", "text", ...optional meta}. Duplicate element_ids
        are collapsed to their first occurrence so the corpus identity is stable."""
        seen = set()
        element_ids: List[str] = []
        docs_meta: List[Dict[str, Any]] = []
        doc_tokens: List[List[str]] = []
        for el in corpus:
            eid = el.get("element_id")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            element_ids.append(eid)
            docs_meta.append({k: v for k, v in el.items() if k != "text"})
            doc_tokens.append(tokenize(el.get("text", ""), tokenizer))

        n = len(doc_tokens)
        df: Dict[str, int] = {}
        for toks in doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        # BM25+ style idf, floored at a small positive value so common terms still rank.
        idf = {t: max(math.log(1.0 + (n - d + 0.5) / (d + 0.5)), 1e-6) for t, d in df.items()}
        doc_len = [len(toks) for toks in doc_tokens]
        avg_len = (sum(doc_len) / n) if n else 0.0
        return cls(element_ids, docs_meta, doc_tokens, df, idf, doc_len, avg_len,
                   tokenizer, k1, b, index_config_hash(tokenizer, k1, b, element_ids))

    def __len__(self) -> int:
        return len(self.element_ids)

    def to_json(self) -> Dict[str, Any]:
        return {"tokenizer": self.tokenizer, "k1": self.k1, "b": self.b,
                "config_hash": self.config_hash, "n_docs": len(self),
                "avg_len": self.avg_len, "element_ids": self.element_ids,
                "docs_meta": self.docs_meta, "doc_tokens": self.doc_tokens,
                "df": self.df, "idf": self.idf, "doc_len": self.doc_len}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "BM25Index":
        return cls(data["element_ids"], data["docs_meta"], data["doc_tokens"],
                   {k: int(v) for k, v in data["df"].items()},
                   {k: float(v) for k, v in data["idf"].items()},
                   [int(x) for x in data["doc_len"]], float(data["avg_len"]),
                   data.get("tokenizer", DEFAULT_TOKENIZER),
                   float(data.get("k1", BM25_K1)), float(data.get("b", BM25_B)),
                   data.get("config_hash", ""))
