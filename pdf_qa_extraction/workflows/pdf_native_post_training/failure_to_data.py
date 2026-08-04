"""P1-8: leakage-safe failure-to-data loop.

Standard flow::

    dev prediction -> error taxonomy -> source/evidence review
                   -> approved correction/curriculum row
                   -> new dataset version -> train
                   -> dev gate -> one-time protected current-final evaluation

Non-negotiable guardrails (why this module exists):

- **Failure mining uses the dev set only.** Final IDs are excluded from the
  training/export/reward input allowlist.
- **A final ID entering a correction/export/train set raises** ``FinalLeakageError``
  (so CI fails). The guard is wired into every producing function here.
- **Each new training row is traceable** to the dev failure + source evidence it
  came from (``derived_from``), and requires human approval (a review event).
- **The final score is produced once, after all design decisions, with an access
  record** (``FinalAccessLedger``). Repeated dev optimisation is counted
  (``DevReuseLedger``) so overfitting to dev is visible.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from evaluation.error_taxonomy import ErrorReport, classify_error

Record = Dict[str, Any]


class FinalLeakageError(Exception):
    """Raised when a protected final/holdout ID reaches corrections/export/train."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_ids(row: Record) -> Set[str]:
    """All IDs a row is associated with: its own id + any dev id it derives from."""
    ids = {str(row.get("qa_id"))} if row.get("qa_id") is not None else set()
    dv = (row.get("derived_from") or {}).get("dev_qa_id")
    if dv is not None:
        ids.add(str(dv))
    return ids


def assert_no_final_leakage(rows: Iterable[Record], final_ids: Iterable[str]) -> None:
    """Raise ``FinalLeakageError`` if any row is, or derives from, a final ID."""
    final = {str(x) for x in final_ids}
    offenders = sorted({i for row in rows for i in _row_ids(row) if i in final})
    if offenders:
        raise FinalLeakageError(
            "protected final IDs present in training/correction input: " + ", ".join(offenders))


# --------------------------------------------------------------------------- #
# mine failures (dev only)
# --------------------------------------------------------------------------- #
@dataclass
class Failure:
    dev_qa_id: str
    categories: List[str]
    prediction: Record
    gold: Optional[Record]
    evidence: List[Record] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)


def mine_failures(predictions: Sequence[Record],
                  gold_by_id: Dict[str, Record],
                  *,
                  dev_ids: Iterable[str],
                  final_ids: Iterable[str] = (),
                  latest_versions: Optional[Dict[str, str]] = None,
                  documents: Optional[Dict[str, Any]] = None) -> List[Failure]:
    """Classify dev predictions and keep only the failing ones.

    Enforces that every mined prediction is a *dev* example and **never** a final
    example (``FinalLeakageError`` otherwise).
    """
    dev = {str(x) for x in dev_ids}
    final = {str(x) for x in final_ids}
    latest_versions = latest_versions or {}

    # Guard the inputs first — mining must not even read final examples.
    pred_ids = {str(p.get("qa_id")) for p in predictions}
    leaked = sorted(pred_ids & final)
    if leaked:
        raise FinalLeakageError("final IDs present in mining input: " + ", ".join(leaked))
    non_dev = sorted(pred_ids - dev)
    if non_dev:
        raise ValueError("failure mining restricted to the dev set; non-dev IDs: "
                         + ", ".join(non_dev))

    failures: List[Failure] = []
    for pred in predictions:
        qid = str(pred.get("qa_id"))
        gold = gold_by_id.get(qid)
        report: ErrorReport = classify_error(
            pred, gold, latest_version=latest_versions.get(qid), documents=documents)
        if report.has_error:
            failures.append(Failure(
                dev_qa_id=qid, categories=report.categories, prediction=pred, gold=gold,
                evidence=list(pred.get("evidence") or []), detail=report.detail))
    return failures


# --------------------------------------------------------------------------- #
# build approved correction rows (with lineage)
# --------------------------------------------------------------------------- #
def build_correction(failure: Failure, corrected_answer: str, reviewer: str, *,
                     review_log=None,
                     new_qa_id: Optional[str] = None,
                     final_ids: Iterable[str] = (),
                     timestamp: Optional[str] = None) -> Record:
    """Create a human-approved correction row carrying full lineage.

    The row records which dev failure and which source evidence it came from.
    Requires a review event (approval); appends one to ``review_log`` when given.
    """
    final = {str(x) for x in final_ids}
    if failure.dev_qa_id in final:
        raise FinalLeakageError(f"correction derives from a final ID: {failure.dev_qa_id}")

    qa_id = new_qa_id or f"corr-{failure.dev_qa_id}"
    row: Record = {
        "qa_id": qa_id,
        "question": failure.prediction.get("question"),
        "answer": corrected_answer,
        "answerable": failure.prediction.get("answerable", True),
        "category": failure.prediction.get("category"),
        "evidence": failure.evidence,
        "generation": {"provider": "correction", "model": "human"},
        "review_status": "owner_review_pending",
        "split": "train",
        "derived_from": {
            "dev_qa_id": failure.dev_qa_id,
            "error_categories": list(failure.categories),
            "evidence": [{"document_sha256": e.get("document_sha256"),
                          "quote_sha256": e.get("quote_sha256"),
                          "page": e.get("page"), "element_id": e.get("element_id")}
                         for e in failure.evidence],
        },
    }

    if review_log is not None:
        from .review import edit as review_edit
        ev = review_edit(review_log, row, reviewer, {"answer": corrected_answer},
                         timestamp=timestamp, note=f"correction for {failure.dev_qa_id}")
        row["review_status"] = review_log.status_of(qa_id)
        row["review_event_id"] = ev["event_id"]
        row["reviewer"] = reviewer
    return row


def _dataset_version(base_version: str, rows: Sequence[Record]) -> str:
    h = hashlib.sha256()
    h.update(base_version.encode("utf-8"))
    for i in sorted(f"{r.get('qa_id')}<-{(r.get('derived_from') or {}).get('dev_qa_id')}"
                    for r in rows):
        h.update(b"\0")
        h.update(i.encode("utf-8"))
    return "ds-" + h.hexdigest()[:12]


def assemble_dataset_version(correction_rows: Sequence[Record], *,
                             base_version: str,
                             final_ids: Iterable[str] = ()) -> Dict[str, Any]:
    """Bundle approved corrections into a new, lineage-tracked dataset version.

    Fails closed (``FinalLeakageError``) if any row is or derives from a final ID.
    """
    assert_no_final_leakage(correction_rows, final_ids)
    version = _dataset_version(base_version, correction_rows)
    lineage = [{"qa_id": r.get("qa_id"),
                "dev_qa_id": (r.get("derived_from") or {}).get("dev_qa_id"),
                "error_categories": (r.get("derived_from") or {}).get("error_categories", []),
                "review_event_id": r.get("review_event_id")}
               for r in correction_rows]
    return {
        "dataset_version": version,
        "base_version": base_version,
        "n_rows": len(correction_rows),
        "created_at": _utc_now(),
        "lineage": lineage,
        "rows": list(correction_rows),
    }


# --------------------------------------------------------------------------- #
# dev-reuse counting (overfitting visibility)
# --------------------------------------------------------------------------- #
class DevReuseLedger:
    """Counts how many optimisation rounds each dev example has been used in."""

    def __init__(self, counts: Optional[Dict[str, int]] = None):
        self._counts: Dict[str, int] = dict(counts or {})
        self._rounds = 0

    @classmethod
    def load(cls, path: str) -> "DevReuseLedger":
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            led = cls(data.get("counts", {}))
            led._rounds = data.get("rounds", 0)
            return led
        return cls()

    def record_round(self, dev_ids: Iterable[str]) -> None:
        self._rounds += 1
        for i in {str(x) for x in dev_ids}:
            self._counts[i] = self._counts.get(i, 0) + 1

    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def max_reuse(self) -> int:
        return max(self._counts.values(), default=0)

    def report(self) -> Dict[str, Any]:
        return {"rounds": self._rounds, "n_examples": len(self._counts),
                "max_reuse": self.max_reuse(), "counts": self.counts()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"rounds": self._rounds, "counts": self._counts}, fh,
                      ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# one-time protected final evaluation
# --------------------------------------------------------------------------- #
class FinalAccessLedger:
    """Append-only record of every access to the protected final set.

    The final score must be produced once, after all design decisions; this
    ledger makes that auditable and blocks a silent second scoring pass.
    """

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        self._records: List[Dict[str, Any]] = list(records or [])

    @classmethod
    def load(cls, path: str) -> "FinalAccessLedger":
        recs: List[Dict[str, Any]] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
        return cls(recs)

    def access(self, final_ids: Iterable[str], reason: str, *,
               scoring: bool = False, path: Optional[str] = None) -> Dict[str, Any]:
        rec = {"timestamp": _utc_now(), "reason": reason, "scoring": bool(scoring),
               "n_ids": len({str(x) for x in final_ids})}
        self._records.append(rec)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def scorings(self) -> List[Dict[str, Any]]:
        return [r for r in self._records if r.get("scoring")]

    def assert_single_scoring(self) -> None:
        n = len(self.scorings())
        if n > 1:
            raise RuntimeError(f"final set scored {n} times; it must be scored once")

    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)
