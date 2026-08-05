"""Unit tests for the deterministic BM25 retrieval core (``pdf_qa.retrieval``).

Pure CPU, no models. Covers: build/config-hash determinism, top-k ordering, tie-breaking,
the no-spurious-hit guard, save/load round-trip, and Korean bigram overlap recall.
"""

import json
import os

from pdf_qa.retrieval import BM25Index, Retriever, index_config_hash

_CORPUS = [
    {"element_id": "d1", "text": "2024년 연간 매출액은 1,250억 원이다", "page": 1},
    {"element_id": "d2", "text": "영업이익률은 12.3 퍼센트로 개선되었다", "page": 2},
    {"element_id": "d3", "text": "부채비율은 45 퍼센트 수준을 유지한다", "page": 3},
    {"element_id": "d4", "text": "직원 수는 3,200명으로 집계되었다", "page": 4},
]


def _index():
    return BM25Index.build(_CORPUS)


def test_build_len_and_config_hash_deterministic():
    idx = _index()
    assert len(idx) == 4
    # config hash is a stable function of tokenizer/k1/b/element-ids
    again = index_config_hash(idx.tokenizer, idx.k1, idx.b, idx.element_ids)
    assert idx.config_hash == again
    assert BM25Index.build(_CORPUS).config_hash == idx.config_hash


def test_search_returns_relevant_top_hit():
    r = Retriever(_index())
    hits = r.search("연간 매출액은 얼마입니까?", k=2)
    assert hits, "expected at least one hit"
    assert hits[0].element_id == "d1"
    # scores are monotonically non-increasing
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_no_spurious_hits_for_unrelated_query():
    r = Retriever(_index())
    # a query with no shared bigrams must not return fabricated matches
    hits = r.search("XYZ 완전히 무관한 영어단어 zzzz", k=4)
    assert all(h.score > 0 for h in hits)


def test_determinism_and_tie_break_by_element_id():
    # two identical documents -> stable ascending element_id ordering on ties
    corpus = [{"element_id": "b", "text": "동일한 문장 내용", "page": 1},
              {"element_id": "a", "text": "동일한 문장 내용", "page": 1}]
    r = Retriever(BM25Index.build(corpus))
    ids1 = r.retrieve_ids("동일한 문장", k=2)
    ids2 = r.retrieve_ids("동일한 문장", k=2)
    assert ids1 == ids2
    assert ids1 == ["a", "b"]  # tie -> ascending element_id


def test_json_roundtrip(tmp_path):
    idx = _index()
    p = os.path.join(tmp_path, "idx.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(idx.to_json(), fh, ensure_ascii=False)
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    assert "element_ids" in data
    reloaded = BM25Index.from_json(data)
    assert reloaded.config_hash == idx.config_hash
    a = Retriever(idx).retrieve_ids("영업이익률", k=3)
    b = Retriever(reloaded).retrieve_ids("영업이익률", k=3)
    assert a == b


def test_k_caps_results():
    r = Retriever(_index())
    assert len(r.search("퍼센트", k=1)) <= 1
