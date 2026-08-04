"""P1-6: separate *stable* facts from *mutable* facts, and select the latest
valid source for a fact (or abstain) instead of memorising a volatile value.

The problem: interest rates, limits, fees, terms and effective dates change.
Baking those into model weights is wrong — the *current* value of a mutable fact
belongs to retrieval / source selection, while SFT should learn *stable behaviour*
(citation format, refusing ungrounded answers, calculation steps, answer shape).

This module is pure-Python, deterministic, and dependency-light. It powers:

- `select_source(...)` / `resolve_fact(...)` — pick the latest valid source for a
  fact, or abstain when the winner is ambiguous or only stale/revoked sources
  exist for a *mutable* fact (never confidently answer an outdated value).
- `partition_for_export(...)` — keep stale/revoked/superseded rows out of the
  active training export and route them to a versioned archive.
- `affected_by_version_change(...)` — track which Q&A and which dataset version
  are impacted when a source document changes version.
- `mutable_fact_report(...)` — recency / citation / abstention counts, reported
  as a *separate category* (feeds the P1-5 base-vs-SFT+retrieval comparison).

All fields are the schema fields already defined in
``pdf_qa/schemas/qa_with_evidence.schema.json`` (``fact_volatility``,
``source_status``, ``document_version``, ``effective_from``, ``effective_until``)
plus the optional ``supersedes`` and grouping hint ``fact_key``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

Record = Dict[str, Any]

_NEG_INF = float("-inf")

ACTIVE = "active"
STALE = "stale"
REVOKED = "revoked"
UNKNOWN = "unknown"

STABLE = "stable"
MUTABLE = "mutable"

# review_status values whose rows must never enter an active training export.
_NON_EXPORT_REVIEW = {"rejected", "owner_review_pending"}


# --------------------------------------------------------------------------- #
# small parsers
# --------------------------------------------------------------------------- #
def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` (best effort)."""
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    m = re.match(r"^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$", s)
    if not m:
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None
    y = int(m.group(1))
    mo = int(m.group(2) or 1)
    d = int(m.group(3) or 1)
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _date_ordinal(value: Optional[str]) -> float:
    d = _parse_date(value)
    return float(d.toordinal()) if d else _NEG_INF


def _version_num(value: Optional[str]) -> float:
    """Extract a sortable number from ``v2``, ``2``, ``rev3`` etc."""
    if value is None:
        return _NEG_INF
    m = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(m.group(1)) if m else _NEG_INF


def normalize_fact_key(rec: Record) -> str:
    """Group records that answer the same underlying fact.

    Uses an explicit ``fact_key`` when present, else a normalised question.
    """
    key = rec.get("fact_key")
    if key:
        return str(key).strip().lower()
    q = str(rec.get("question", "")).strip().lower()
    return re.sub(r"\s+", " ", q)


def _order_key(rec: Record) -> tuple:
    """Ordering for 'latest' — effective_from date first, then document_version."""
    return (_date_ordinal(rec.get("effective_from")), _version_num(rec.get("document_version")))


def _in_effect(rec: Record, as_of: Optional[str]) -> bool:
    """Whether ``rec`` is in force at ``as_of`` (open bounds when a date is null)."""
    if as_of is None:
        return True
    ao = _parse_date(as_of)
    if ao is None:
        return True
    lo = _parse_date(rec.get("effective_from"))
    hi = _parse_date(rec.get("effective_until"))
    if lo is not None and ao < lo:
        return False
    if hi is not None and ao > hi:
        return False
    return True


def is_stale_or_revoked(rec: Record) -> bool:
    return rec.get("source_status") in (STALE, REVOKED)


def is_mutable(rec: Record) -> bool:
    return rec.get("fact_volatility") == MUTABLE


# --------------------------------------------------------------------------- #
# core selection
# --------------------------------------------------------------------------- #
@dataclass
class SourceDecision:
    selected: Optional[Record]
    abstain: bool
    reason: str
    considered: int = 0
    dropped: List[str] = field(default_factory=list)

    @property
    def answer(self) -> Optional[str]:
        return None if self.selected is None else self.selected.get("answer")


def select_source(candidates: Sequence[Record], as_of: Optional[str] = None) -> SourceDecision:
    """Select the latest *valid* source among candidates answering one fact.

    Rules (in order):
      1. Revoked sources are never selected.
      2. If ``as_of`` is given, only sources in force at that date are considered.
      3. Prefer ``active`` sources; pick the latest by (effective_from, version).
      4. If two active sources are equally-ranked but disagree, **abstain**
         (unresolved conflict — never guess).
      5. If only stale/unknown sources remain:
         - a *mutable* fact **abstains** (don't answer an outdated value);
         - a *stable* fact may still be selected (its value does not expire),
           flagged via the decision reason.
    """
    cands = list(candidates)
    n = len(cands)
    if n == 0:
        return SourceDecision(None, True, "no_candidates", 0)

    dropped: List[str] = []
    live = []
    for c in cands:
        if c.get("source_status") == REVOKED:
            dropped.append("revoked")
            continue
        if not _in_effect(c, as_of):
            dropped.append("not_in_effect")
            continue
        live.append(c)

    if not live:
        return SourceDecision(None, True, "none_in_effect", n, dropped)

    mutable = any(is_mutable(c) for c in cands)
    active = [c for c in live if c.get("source_status") == ACTIVE]
    pool = active if active else live

    pool_sorted = sorted(pool, key=_order_key, reverse=True)
    top = pool_sorted[0]
    top_key = _order_key(top)

    # Unresolved conflict: several equally-ranked winners that disagree.
    tied = [c for c in pool_sorted if _order_key(c) == top_key]
    tied_answers = {str(c.get("answer")) for c in tied}
    if len(tied_answers) > 1:
        return SourceDecision(None, True, "conflict_no_order", n, dropped)

    if not active:
        if mutable:
            return SourceDecision(None, True, "only_stale_mutable", n, dropped)
        return SourceDecision(top, False, "selected_stable_from_stale", n, dropped)

    return SourceDecision(top, False, "selected_latest_active", n, dropped)


def group_by_fact(records: Sequence[Record]) -> Dict[str, List[Record]]:
    groups: Dict[str, List[Record]] = {}
    for r in records:
        groups.setdefault(normalize_fact_key(r), []).append(r)
    return groups


def resolve_fact(records: Sequence[Record], fact_key: str,
                 as_of: Optional[str] = None) -> SourceDecision:
    groups = group_by_fact(records)
    return select_source(groups.get(fact_key.strip().lower(), []), as_of=as_of)


# --------------------------------------------------------------------------- #
# training-export partitioning
# --------------------------------------------------------------------------- #
@dataclass
class ExportPartition:
    active_export: List[Record] = field(default_factory=list)
    versioned_archive: List[Record] = field(default_factory=list)
    held_for_review: List[Record] = field(default_factory=list)
    decisions: Dict[str, str] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        return {
            "active_export": len(self.active_export),
            "versioned_archive": len(self.versioned_archive),
            "held_for_review": len(self.held_for_review),
        }


def _qid(rec: Record) -> str:
    return str(rec.get("qa_id", id(rec)))


def partition_for_export(records: Sequence[Record],
                         as_of: Optional[str] = None) -> ExportPartition:
    """Route rows so stale/revoked/superseded facts stay out of active training.

    - ``stale``/``revoked`` sources → versioned archive.
    - rows explicitly ``rejected``/``owner_review_pending`` → held for review.
    - for a *mutable* fact, only the selected latest source stays in the active
      export; older (superseded) siblings go to the archive; an unresolved
      conflict / only-stale group is held (never silently trained on).
    - *stable* facts and abstention (unanswerable) rows are behaviour and stay
      in the active export as long as their source is not revoked.
    """
    part = ExportPartition()
    groups = group_by_fact(records)

    for key, members in groups.items():
        decision = select_source(members, as_of=as_of)
        part.decisions[key] = decision.reason
        selected_id = _qid(decision.selected) if decision.selected is not None else None
        group_mutable = any(is_mutable(m) for m in members)

        for rec in members:
            if rec.get("review_status") in _NON_EXPORT_REVIEW:
                part.held_for_review.append(rec)
                continue
            if is_stale_or_revoked(rec):
                part.versioned_archive.append(rec)
                continue
            # Unanswerable / abstention rows are stable behaviour.
            if not rec.get("answerable", True):
                part.active_export.append(rec)
                continue
            if group_mutable:
                if decision.abstain:
                    part.held_for_review.append(rec)
                elif _qid(rec) == selected_id:
                    part.active_export.append(rec)
                else:
                    part.versioned_archive.append(rec)  # superseded sibling
            else:
                part.active_export.append(rec)  # stable fact / behaviour

    return part


# --------------------------------------------------------------------------- #
# version-change lineage
# --------------------------------------------------------------------------- #
def _dataset_version(affected: Sequence[str], new_version: str) -> str:
    h = hashlib.sha256()
    h.update(new_version.encode("utf-8"))
    for q in sorted(affected):
        h.update(b"\0")
        h.update(q.encode("utf-8"))
    return "ds-" + h.hexdigest()[:12]


def affected_by_version_change(records: Sequence[Record], *,
                               new_version: str,
                               document_sha256: Optional[str] = None,
                               document_id: Optional[str] = None) -> Dict[str, Any]:
    """Q&A affected when a source document moves to ``new_version``.

    A record is *affected* if any of its evidence cites the identified document
    at a version other than ``new_version`` (i.e. it now points at an old
    revision and must be re-reviewed). Returns the affected ``qa_id`` set plus a
    deterministic ``dataset_version`` tag for lineage.
    """
    if not document_sha256 and not document_id:
        raise ValueError("document_sha256 or document_id is required")

    affected: List[str] = []
    for rec in records:
        hit = False
        for ev in rec.get("evidence", []) or []:
            same_doc = (
                (document_sha256 and ev.get("document_sha256") == document_sha256)
                or (document_id and ev.get("document_id") == document_id)
            )
            if not same_doc:
                continue
            ev_ver = ev.get("document_version", rec.get("document_version"))
            if ev_ver != new_version:
                hit = True
                break
        if hit:
            affected.append(_qid(rec))

    affected = sorted(set(affected))
    return {
        "new_version": new_version,
        "affected_qa_ids": affected,
        "n_affected": len(affected),
        "dataset_version": _dataset_version(affected, new_version),
    }


# --------------------------------------------------------------------------- #
# reporting (separate mutable-fact category for P1-5)
# --------------------------------------------------------------------------- #
def _has_citation(rec: Record) -> bool:
    return bool(rec.get("evidence"))


def mutable_fact_report(records: Sequence[Record],
                        as_of: Optional[str] = None) -> Dict[str, Any]:
    """Recency / citation / abstention breakdown for *mutable* facts only.

    Reported as its own category so a base-vs-SFT+retrieval comparison (P1-5)
    can show mutable-fact freshness separately from static QA accuracy.
    """
    groups = group_by_fact(records)
    n_facts = 0
    recency_ok = 0
    abstained = 0
    resolved = 0
    citation_ok = 0

    for members in groups.values():
        if not any(is_mutable(m) for m in members):
            continue
        n_facts += 1
        decision = select_source(members, as_of=as_of)
        if decision.abstain:
            abstained += 1
            continue
        resolved += 1
        sel = decision.selected or {}
        if sel.get("source_status") == ACTIVE:
            recency_ok += 1
        if _has_citation(sel):
            citation_ok += 1

    return {
        "category": "mutable_fact_recency",
        "n_mutable_facts": n_facts,
        "resolved": resolved,
        "abstained": abstained,
        "recency_ok": recency_ok,
        "citation_ok": citation_ok,
        "recency_rate": (recency_ok / resolved) if resolved else None,
        "citation_rate": (citation_ok / resolved) if resolved else None,
        "abstention_rate": (abstained / n_facts) if n_facts else None,
    }


# --------------------------------------------------------------------------- #
# convenience loader
# --------------------------------------------------------------------------- #
def load_records(path: str) -> List[Record]:
    out: List[Record] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
