"""P1-5: same-contract evaluation for the PDF-native benchmark.

A pure, dependency-light metric library (no ``workflows`` import — it is a core
package) that scores raw per-example predictions against gold labels and builds an
aggregate table **auto-generated from that raw**, so a reader can re-derive every
headline number. It implements the benchmark's minimum metric contract:

    EM / F1, numeric·date·unit exactness, citation page+span accuracy,
    retrieval recall@k + no-answer retrieval rate, answerable/unanswerable
    abstention precision/recall, schema validity, evidence groundedness,
    PII leakage rate, per-category accuracy + failure taxonomy.

It also provides the leakage-audited split guard (``assert_no_split_leakage``)
that keeps document families — and a family's v1/v2 versions — from crossing
train/dev/final, and a deterministic reference retriever so recall@k is real on
the public regression fixture. Model-comparison arms (Base/SFT/PTQ/QAT ± retrieval)
are produced by the GPU workflow; this module scores whatever predictions it is
handed and never fabricates them.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.error_taxonomy import ERROR_CATEGORIES  # shared category vocabulary

# --------------------------------------------------------------------------- #
# text normalization + char F1 (KorQuAD-style, self-contained)
# --------------------------------------------------------------------------- #
_PUNC = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~·…“”‘’「」『』()')


def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = "".join(ch for ch in s if ch not in _PUNC)
    s = " ".join(s.split())
    return s.lower()


def _char_tokens(text: str) -> List[str]:
    return [c for c in normalize_answer(text).replace(" ", "")]


def f1_score(prediction: str, ground_truth: str) -> float:
    pred, gold = _char_tokens(prediction), _char_tokens(ground_truth)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(pred), n / len(gold)
    return 2 * p * r / (p + r)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# --------------------------------------------------------------------------- #
# typed exactness (numeric / currency / date / unit)
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_DATE = re.compile(r"\d{4}\s*[-./년]\s*\d{1,2}\s*[-./월]?\s*\d{0,2}\s*일?|\d{4}년|\d{1,2}월|\d{1,2}일")
_UNIT = re.compile(r"(%|퍼센트|원|억\s*원|만\s*원|천\s*원|년|월|일|건|명|개)")


def _nums(text: str) -> List[str]:
    return [m.replace(",", "") for m in _NUM.findall(text or "")]


def _units(text: str) -> List[str]:
    return [re.sub(r"\s+", "", u) for u in _UNIT.findall(text or "")]


def _dates(text: str) -> List[str]:
    return [re.sub(r"\s+", "", d) for d in _DATE.findall(text or "")]


def numeric_exact(prediction: str, gold: str) -> Optional[bool]:
    g = _nums(gold)
    if not g:
        return None  # not a numeric-bearing gold answer
    return set(g).issubset(set(_nums(prediction)))


def unit_exact(prediction: str, gold: str) -> Optional[bool]:
    g = _units(gold)
    if not g:
        return None
    return set(g).issubset(set(_units(prediction)))


def date_exact(prediction: str, gold: str) -> Optional[bool]:
    g = _dates(gold)
    if not g:
        return None
    return set(g).issubset(set(_dates(prediction)))


# --------------------------------------------------------------------------- #
# citation page + span accuracy
# --------------------------------------------------------------------------- #
def _gold_pages(evidence: Sequence[Dict[str, Any]]) -> set:
    return {e.get("page") for e in evidence if e.get("page") is not None}


def citation_page_correct(pred_citations: Sequence[Dict[str, Any]],
                          gold_evidence: Sequence[Dict[str, Any]]) -> Optional[bool]:
    gold_pages = _gold_pages(gold_evidence)
    if not gold_pages:
        return None
    pred_pages = {c.get("page") for c in (pred_citations or []) if c.get("page") is not None}
    if not pred_pages:
        return False
    return pred_pages.issubset(gold_pages)


def citation_span_correct(pred_citations: Sequence[Dict[str, Any]],
                          gold_evidence: Sequence[Dict[str, Any]]) -> Optional[bool]:
    """A predicted citation span is correct if it matches a gold quote by hash, by
    element_id, or by substantial substring overlap."""
    if not gold_evidence:
        return None
    gold_ids = {e.get("element_id") for e in gold_evidence if e.get("element_id")}
    gold_hashes = {e.get("quote_sha256") for e in gold_evidence if e.get("quote_sha256")}
    gold_quotes = [normalize_answer(e.get("quote", "")) for e in gold_evidence if e.get("quote")]
    if not (pred_citations or []):
        return False
    for c in pred_citations:
        if c.get("element_id") in gold_ids or c.get("quote_sha256") in gold_hashes:
            return True
        q = normalize_answer(c.get("quote", ""))
        if q and any(q in gq or gq in q for gq in gold_quotes):
            return True
    return False


# --------------------------------------------------------------------------- #
# deterministic reference retriever + recall@k
# --------------------------------------------------------------------------- #
def _tok(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", normalize_answer(text)) if t]


def lexical_retrieve(corpus: Sequence[Dict[str, Any]], question: str, k: int = 5
                     ) -> List[str]:
    """Rank corpus elements by lexical overlap with the question (deterministic).

    corpus: [{"element_id", "text"}]. Returns the top-k element_ids.
    """
    q = Counter(_tok(question))
    scored: List[Tuple[float, str]] = []
    for el in corpus:
        toks = Counter(_tok(el.get("text", "")))
        overlap = sum((q & toks).values())
        if overlap:
            denom = (sum(toks.values()) ** 0.5) or 1.0
            scored.append((overlap / denom, el.get("element_id")))
    scored.sort(key=lambda x: (-x[0], str(x[1])))
    return [eid for _, eid in scored[:k]]


def recall_at_k(gold_evidence_ids: Iterable[str], retrieved_ids: Sequence[str],
                k: int) -> Optional[float]:
    gold = {g for g in gold_evidence_ids if g}
    if not gold:
        return None
    got = set(retrieved_ids[:k])
    return len(gold & got) / len(gold)


# --------------------------------------------------------------------------- #
# abstention / schema / groundedness / PII
# --------------------------------------------------------------------------- #
_ABSTAIN = re.compile(r"확인할\s*수\s*없|모르|답할\s*수\s*없|알\s*수\s*없|해당\s*없|근거\s*없|cannot|not\s+found|unknown", re.I)


def predicted_abstained(prediction: Dict[str, Any]) -> bool:
    if prediction.get("abstained") is True:
        return True
    return bool(_ABSTAIN.search(prediction.get("answer", "") or ""))


def schema_valid(prediction: Dict[str, Any],
                 required: Sequence[str] = ("qa_id", "answer")) -> bool:
    if not isinstance(prediction, dict):
        return False
    if any(prediction.get(f) in (None, "") for f in required):
        return False
    cits = prediction.get("citations")
    if cits is not None and not isinstance(cits, list):
        return False
    return True


def is_grounded(prediction: Dict[str, Any], gold_evidence: Sequence[Dict[str, Any]]
                ) -> bool:
    """Answer is grounded if abstained, or its numeric/content tokens are covered by
    the cited evidence text."""
    if predicted_abstained(prediction):
        return True
    ans = prediction.get("answer", "") or ""
    ev_text = " ".join(e.get("quote", "") for e in (gold_evidence or []))
    nums = _nums(ans)
    if nums:
        return set(nums).issubset(set(_nums(ev_text)))
    a, e = set(_char_tokens(ans)), set(_char_tokens(ev_text))
    if not a:
        return False
    return len(a & e) / len(a) >= 0.6


def pii_leaked(prediction: Dict[str, Any], pii_terms: Sequence[str]) -> bool:
    ans = prediction.get("answer", "") or ""
    return any(term and term in ans for term in pii_terms)


# --------------------------------------------------------------------------- #
# per-example scoring
# --------------------------------------------------------------------------- #
def score_example(gold: Dict[str, Any], pred: Dict[str, Any], *,
                  retrieved_ids: Optional[Sequence[str]] = None, k: int = 5,
                  pii_terms: Sequence[str] = ()) -> Dict[str, Any]:
    answerable = bool(gold.get("answerable", True))
    gold_answer = gold.get("answer", "")
    pred_answer = pred.get("answer", "")
    gold_ev = gold.get("evidence", []) or []
    abstained = predicted_abstained(pred)

    rec: Dict[str, Any] = {
        "qa_id": gold.get("qa_id"),
        "category": gold.get("category", "unknown"),
        "answerable": answerable,
        "abstained": abstained,
        "em": exact_match(pred_answer, gold_answer) if answerable else None,
        "f1": f1_score(pred_answer, gold_answer) if answerable else None,
        "numeric_exact": numeric_exact(pred_answer, gold_answer) if answerable else None,
        "unit_exact": unit_exact(pred_answer, gold_answer) if answerable else None,
        "date_exact": date_exact(pred_answer, gold_answer) if answerable else None,
        "citation_page_correct": citation_page_correct(pred.get("citations"), gold_ev) if answerable else None,
        "citation_span_correct": citation_span_correct(pred.get("citations"), gold_ev) if answerable else None,
        "schema_valid": schema_valid(pred),
        "grounded": is_grounded(pred, gold_ev),
        "pii_leaked": pii_leaked(pred, pii_terms),
    }
    if retrieved_ids is not None:
        gold_ids = [e.get("element_id") for e in gold_ev]
        rec["recall_at_k"] = recall_at_k(gold_ids, list(retrieved_ids), k)
        rec["retrieved_any"] = len(list(retrieved_ids)[:k]) > 0

    rec["error_categories"] = failure_categories(rec)
    return rec


def failure_categories(rec: Dict[str, Any]) -> List[str]:
    """Derive a failure taxonomy from the contract metrics themselves (self-consistent,
    unlike a foreign classifier). A perfect prediction yields an empty list."""
    cats: List[str] = []
    if rec["answerable"]:
        if (rec.get("f1") or 0.0) < 0.5:
            cats.append("wrong_answer")
        if False in (rec.get("numeric_exact"), rec.get("unit_exact"), rec.get("date_exact")):
            cats.append("numeric_unit")
        if rec.get("citation_page_correct") is False or rec.get("citation_span_correct") is False:
            cats.append("citation")
        if rec.get("recall_at_k") == 0:
            cats.append("retrieval")
        if rec["abstained"]:
            cats.append("over_abstention")
    elif not rec["abstained"]:
        cats.append("missed_abstention")
    if not rec["grounded"]:
        cats.append("grounding")
    if not rec["schema_valid"]:
        cats.append("schema")
    if rec["pii_leaked"]:
        cats.append("pii")
    return cats


def _summarize_failures(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counter: Counter = Counter()
    for r in records:
        counter.update(r["error_categories"])
    return {
        "n_reports": len(records),
        "n_with_error": sum(1 for r in records if r["error_categories"]),
        "by_category": dict(sorted(counter.items(), key=lambda x: (-x[1], x[0]))),
    }


# --------------------------------------------------------------------------- #
# aggregate (auto-generated from raw per-example records)
# --------------------------------------------------------------------------- #
def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in vals if isinstance(x, (int, float, bool))]
    return round(sum(v) / len(v), 4) if v else None


def _rate(flags: Sequence[Optional[bool]]) -> Optional[float]:
    v = [bool(x) for x in flags if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def abstention_precision_recall(records: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    tp = sum(1 for r in records if r["abstained"] and not r["answerable"])
    fp = sum(1 for r in records if r["abstained"] and r["answerable"])
    fn = sum(1 for r in records if not r["abstained"] and not r["answerable"])
    prec = round(tp / (tp + fp), 4) if (tp + fp) else None
    rec = round(tp / (tp + fn), 4) if (tp + fn) else None
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def aggregate(records: Sequence[Dict[str, Any]], *, k: int = 5) -> Dict[str, Any]:
    """Build the aggregate table from raw per-example records (nothing hand-typed)."""
    ans = [r for r in records if r["answerable"]]
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    per_category = {}
    for cat, rows in sorted(by_cat.items()):
        a = [r for r in rows if r["answerable"]]
        per_category[cat] = {
            "n": len(rows),
            "em": _mean([r["em"] for r in a]),
            "f1": _mean([r["f1"] for r in a]),
            "grounded_rate": _rate([r["grounded"] for r in rows]),
        }

    taxonomy = _summarize_failures(records)

    agg = {
        "n_examples": len(records),
        "n_answerable": len(ans),
        "n_unanswerable": len(records) - len(ans),
        "em": _mean([r["em"] for r in ans]),
        "f1": _mean([r["f1"] for r in ans]),
        "numeric_exact_rate": _rate([r["numeric_exact"] for r in ans]),
        "unit_exact_rate": _rate([r["unit_exact"] for r in ans]),
        "date_exact_rate": _rate([r["date_exact"] for r in ans]),
        "citation_page_accuracy": _rate([r["citation_page_correct"] for r in ans]),
        "citation_span_accuracy": _rate([r["citation_span_correct"] for r in ans]),
        "schema_validity_rate": _rate([r["schema_valid"] for r in records]),
        "groundedness_rate": _rate([r["grounded"] for r in records]),
        "pii_leakage_rate": _rate([r["pii_leaked"] for r in records]),
        "abstention": abstention_precision_recall(records),
        "per_category": per_category,
        "failure_taxonomy": taxonomy,
    }
    if any("recall_at_k" in r for r in records):
        agg["retrieval_recall_at_k"] = _mean([r.get("recall_at_k") for r in ans])
        agg["retrieval_k"] = k
        unans = [r for r in records if not r["answerable"] and "retrieved_any" in r]
        if unans:
            agg["no_answer_retrieval_rate"] = _rate([not r["retrieved_any"] for r in unans])
    return agg


def score_dataset(gold: Sequence[Dict[str, Any]], predictions: Dict[str, Dict[str, Any]], *,
                  corpus: Optional[Sequence[Dict[str, Any]]] = None, k: int = 5,
                  pii_terms: Sequence[str] = ()) -> Dict[str, Any]:
    """Score a whole set: returns {"per_example": [...], "aggregate": {...}}.

    predictions keyed by qa_id. When ``corpus`` is given, recall@k is computed with
    the deterministic reference retriever.
    """
    per_example: List[Dict[str, Any]] = []
    for g in gold:
        qid = g.get("qa_id")
        pred = predictions.get(qid, {"qa_id": qid, "answer": ""})
        retrieved = lexical_retrieve(corpus, g.get("question", ""), k) if corpus else None
        per_example.append(score_example(g, pred, retrieved_ids=retrieved, k=k, pii_terms=pii_terms))
    return {"per_example": per_example, "aggregate": aggregate(per_example, k=k)}


# --------------------------------------------------------------------------- #
# leakage-audited splits (document_family_id based)
# --------------------------------------------------------------------------- #
class SplitLeakageError(AssertionError):
    """Raised when a document family or source span crosses train/dev/final."""


def _family(rec: Dict[str, Any]) -> Optional[str]:
    return rec.get("document_family_id") or rec.get("family_id")


def _span_keys(rec: Dict[str, Any]) -> set:
    keys = set()
    for e in rec.get("evidence", []) or []:
        if e.get("quote_sha256"):
            keys.add("h:" + e["quote_sha256"])
        elif e.get("element_id"):
            keys.add("e:" + str(e["element_id"]))
    return keys


def assert_no_split_leakage(splits: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Audit that no document family — and no exact source span — crosses splits.

    A family (incl. its v1/v2 document versions) must live in exactly one split.
    Returns an audit dict; raises SplitLeakageError on any overlap.
    """
    fam_to_splits: Dict[str, set] = defaultdict(set)
    span_to_splits: Dict[str, set] = defaultdict(set)
    fam_versions: Dict[str, set] = defaultdict(set)
    for split_name, rows in splits.items():
        for r in rows:
            fam = _family(r)
            if fam is not None:
                fam_to_splits[fam].add(split_name)
                fam_versions[fam].add(r.get("document_version"))
            for sk in _span_keys(r):
                span_to_splits[sk].add(split_name)

    family_overlaps = {f: sorted(s) for f, s in fam_to_splits.items() if len(s) > 1}
    span_overlaps = {k: sorted(s) for k, s in span_to_splits.items() if len(s) > 1}
    if family_overlaps or span_overlaps:
        raise SplitLeakageError(
            f"family overlaps={family_overlaps} span overlaps={list(span_overlaps)[:5]}")

    return {
        "splits": {name: len(rows) for name, rows in splits.items()},
        "families": {name: sorted({_family(r) for r in rows if _family(r)})
                     for name, rows in splits.items()},
        "n_families": len(fam_to_splits),
        "family_versions_together": {f: sorted(v for v in vs if v)
                                     for f, vs in fam_versions.items() if len(vs) > 1},
        "intersection_size": 0,
        "disjoint": True,
    }
