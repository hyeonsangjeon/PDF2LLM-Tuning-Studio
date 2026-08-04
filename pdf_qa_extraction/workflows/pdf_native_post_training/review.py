"""P1-7: evidence-centered **local review workflow** and approval queue.

Generated Q&A never enters training unreviewed. This is a small local /
single-user reviewer, so it is deliberately called a *local review workflow* —
not an enterprise review system. It has no authn/authz/audit-retention, and this
module never claims to.

Design guarantees (the parts that matter for trust):

- **Approval is a projection over an append-only event log**, not an in-place
  edit of a Q&A row's ``review_status`` field. A row that merely *says*
  ``"review_status": "approved"`` in JSONL but has no accepting event is treated
  as **unreviewed** and stays out of training.
- **Default training export contains only ``approved`` (incl. ``edited``) rows.**
  Rejected rows are quarantined; unreviewed rows are held. (Composes with the
  P1-6 source-selection partition, so stale/revoked/superseded rows are excluded
  too.)
- **Edited rows preserve the original generated value and the change diff.**
- **Every exported row is traceable** to a reviewer event and its source evidence.
- **Source snippets are redactable** for readers without source access.

CLI: ``python -m workflows.pdf_native_post_training.review <cmd> ...``
(``list | accept | edit | reject | reopen | export | trace | verify``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from pdf_qa.run_bundle import sha256_canonical  # noqa: E402

Record = Dict[str, Any]
Event = Dict[str, Any]

# reject-reason taxonomy (mirrors the JSON schema enum, minus null).
REJECT_REASONS = (
    "ungrounded", "wrong_value", "wrong_version", "numeric_unit_error",
    "pii_exposure", "schema_violation", "policy_violation", "ocr_error",
    "hallucinated_citation", "duplicate", "other",
)

ACTIONS = ("accept", "edit", "reject", "reopen")

# projected review_status per latest action.
_ACTION_STATUS = {
    "accept": "approved",
    "edit": "edited",
    "reject": "rejected",
    "reopen": "owner_review_pending",
}
_APPROVED_STATES = {"approved", "edited"}
_DEFAULT_STATUS = "owner_review_pending"

WORKFLOW_LABEL = "local review workflow"


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _event_content(qa_id: str, action: str, reviewer: str, timestamp: str,
                   reject_reason: Optional[str], edits: List[Dict[str, Any]],
                   evidence_anchor: Optional[Dict[str, Any]],
                   prev_event_id: Optional[str], note: Optional[str]) -> Dict[str, Any]:
    return {
        "qa_id": qa_id, "action": action, "reviewer": reviewer, "timestamp": timestamp,
        "reject_reason": reject_reason, "edits": edits,
        "evidence_anchor": evidence_anchor, "prev_event_id": prev_event_id, "note": note,
    }


def make_event(qa_id: str, action: str, reviewer: str, *,
               timestamp: Optional[str] = None,
               reject_reason: Optional[str] = None,
               edits: Optional[List[Dict[str, Any]]] = None,
               evidence_anchor: Optional[Dict[str, Any]] = None,
               prev_event_id: Optional[str] = None,
               note: Optional[str] = None) -> Event:
    """Build a hashed, chainable review event."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action}")
    if action == "reject" and not reject_reason:
        raise ValueError("reject requires a reject_reason")
    if reject_reason is not None and reject_reason not in REJECT_REASONS:
        raise ValueError(f"unknown reject_reason: {reject_reason}")
    ts = timestamp or _utc_now()
    edits = edits or []
    content = _event_content(qa_id, action, reviewer, ts, reject_reason, edits,
                             evidence_anchor, prev_event_id, note)
    content_hash = sha256_canonical(content)
    event = dict(content)
    event["event_id"] = content_hash[:24]
    event["event_sha256"] = content_hash
    return event


def _recompute_hash(event: Event) -> str:
    content = _event_content(
        event["qa_id"], event["action"], event["reviewer"], event["timestamp"],
        event.get("reject_reason"), event.get("edits", []) or [],
        event.get("evidence_anchor"), event.get("prev_event_id"), event.get("note"),
    )
    return sha256_canonical(content)


def anchor_from_record(record: Record) -> Optional[Dict[str, Any]]:
    ev = (record.get("evidence") or [])
    if not ev:
        return None
    first = ev[0]
    return {"document_sha256": first.get("document_sha256"),
            "quote_sha256": first.get("quote_sha256")}


# --------------------------------------------------------------------------- #
# append-only log
# --------------------------------------------------------------------------- #
class ReviewLog:
    """An append-only log of review events with per-qa chaining + integrity."""

    def __init__(self, events: Optional[Iterable[Event]] = None):
        self._events: List[Event] = list(events or [])

    # -- construction ------------------------------------------------------- #
    @classmethod
    def load(cls, path: str) -> "ReviewLog":
        events: List[Event] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        log = cls(events)
        log.verify()
        return log

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for e in self._events:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    # -- mutation (append only) -------------------------------------------- #
    def last_event_id(self, qa_id: str) -> Optional[str]:
        for e in reversed(self._events):
            if e["qa_id"] == qa_id:
                return e["event_id"]
        return None

    def append(self, event: Event) -> Event:
        """Append an event, auto-chaining ``prev_event_id`` when omitted."""
        if event.get("prev_event_id") is None:
            prev = self.last_event_id(event["qa_id"])
            if prev is not None:
                # rebuild with the correct chain so the hash stays valid
                event = make_event(
                    event["qa_id"], event["action"], event["reviewer"],
                    timestamp=event["timestamp"], reject_reason=event.get("reject_reason"),
                    edits=event.get("edits", []) or [],
                    evidence_anchor=event.get("evidence_anchor"),
                    prev_event_id=prev, note=event.get("note"),
                )
        if event["event_sha256"] != _recompute_hash(event):
            raise ValueError("event hash mismatch (tampered event)")
        self._events.append(event)
        return event

    def append_to_file(self, event: Event, path: str) -> Event:
        event = self.append(event)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    # -- reads -------------------------------------------------------------- #
    def events(self) -> List[Event]:
        return list(self._events)

    def events_for(self, qa_id: str) -> List[Event]:
        return [e for e in self._events if e["qa_id"] == qa_id]

    def verify(self) -> None:
        """Raise if any event hash is broken or a per-qa chain is inconsistent."""
        seen_last: Dict[str, Optional[str]] = {}
        ids = set()
        for e in self._events:
            if e["event_sha256"] != _recompute_hash(e):
                raise ValueError(f"integrity failure: bad hash for event {e.get('event_id')}")
            if e["event_id"] in ids:
                raise ValueError(f"integrity failure: duplicate event_id {e['event_id']}")
            ids.add(e["event_id"])
            expected_prev = seen_last.get(e["qa_id"])
            if e.get("prev_event_id") != expected_prev:
                raise ValueError(
                    f"integrity failure: broken chain for {e['qa_id']} "
                    f"(prev={e.get('prev_event_id')}, expected={expected_prev})")
            seen_last[e["qa_id"]] = e["event_id"]

    # -- projection --------------------------------------------------------- #
    def latest(self, qa_id: str) -> Optional[Event]:
        last = None
        for e in self._events:
            if e["qa_id"] == qa_id:
                last = e
        return last

    def status_of(self, qa_id: str) -> str:
        last = self.latest(qa_id)
        return _ACTION_STATUS.get(last["action"], _DEFAULT_STATUS) if last else _DEFAULT_STATUS

    def project(self, records: Iterable[Record]) -> Dict[str, str]:
        """Map qa_id -> projected review_status from the event log only."""
        return {r["qa_id"]: self.status_of(r["qa_id"]) for r in records}


# --------------------------------------------------------------------------- #
# high-level review actions (fill anchors + edit diffs)
# --------------------------------------------------------------------------- #
def accept(log: ReviewLog, record: Record, reviewer: str, *,
           timestamp: Optional[str] = None, note: Optional[str] = None) -> Event:
    return log.append(make_event(record["qa_id"], "accept", reviewer, timestamp=timestamp,
                                 evidence_anchor=anchor_from_record(record), note=note))


def reject(log: ReviewLog, record: Record, reviewer: str, reason: str, *,
           timestamp: Optional[str] = None, note: Optional[str] = None) -> Event:
    return log.append(make_event(record["qa_id"], "reject", reviewer, reject_reason=reason,
                                 timestamp=timestamp, evidence_anchor=anchor_from_record(record),
                                 note=note))


def reopen(log: ReviewLog, record: Record, reviewer: str, *,
           timestamp: Optional[str] = None, note: Optional[str] = None) -> Event:
    return log.append(make_event(record["qa_id"], "reopen", reviewer, timestamp=timestamp,
                                 note=note))


def edit(log: ReviewLog, record: Record, reviewer: str, changes: Dict[str, Any], *,
         timestamp: Optional[str] = None, note: Optional[str] = None) -> Event:
    """Record an edit as an accept-with-diff; ``changes`` maps field -> new value."""
    edits = [{"field": f, "old": copy.deepcopy(record.get(f)), "new": v}
             for f, v in changes.items()]
    return log.append(make_event(record["qa_id"], "edit", reviewer, edits=edits,
                                 timestamp=timestamp, evidence_anchor=anchor_from_record(record),
                                 note=note))


def apply_edits(record: Record, events: List[Event]) -> Record:
    """Apply recorded edits, preserving the original generated values + diff."""
    out = copy.deepcopy(record)
    applied: List[Dict[str, Any]] = []
    original: Dict[str, Any] = {}
    for e in events:
        if e["action"] != "edit":
            continue
        for ch in e.get("edits", []) or []:
            f = ch["field"]
            if f not in original:
                original[f] = copy.deepcopy(out.get(f))
            out[f] = ch["new"]
            applied.append({"field": f, "old": ch["old"], "new": ch["new"],
                            "event_id": e["event_id"]})
    if applied:
        out["_review_original"] = original
        out["_review_edits"] = applied
    return out


# --------------------------------------------------------------------------- #
# export + trace + redaction
# --------------------------------------------------------------------------- #
@dataclass
class ReviewExport:
    train: List[Record]
    quarantine: List[Record]
    pending: List[Record]

    def counts(self) -> Dict[str, int]:
        return {"train": len(self.train), "quarantine": len(self.quarantine),
                "pending": len(self.pending)}


def export_training(records: Iterable[Record], log: ReviewLog, *,
                    apply_source_selection: bool = True,
                    as_of: Optional[str] = None) -> ReviewExport:
    """Project approvals from the log and split rows for training.

    Only ``approved``/``edited`` rows reach ``train``; rejected rows are
    quarantined; everything else is pending. When ``apply_source_selection`` is
    set (default), the P1-6 partition runs first so stale/revoked/superseded and
    unresolved-conflict rows never reach training either.
    """
    records = list(records)

    allowed_ids = {r["qa_id"] for r in records}
    if apply_source_selection:
        from .source_selection import partition_for_export
        part = partition_for_export(records)
        allowed_ids = {r["qa_id"] for r in part.active_export}

    train, quarantine, pending = [], [], []
    for rec in records:
        status = log.status_of(rec["qa_id"])
        if status == "rejected":
            quarantine.append(_with_review_meta(rec, log, status))
            continue
        if status in _APPROVED_STATES and rec["qa_id"] in allowed_ids:
            row = apply_edits(rec, log.events_for(rec["qa_id"]))
            train.append(_with_review_meta(row, log, status))
        else:
            pending.append(_with_review_meta(rec, log, status))
    return ReviewExport(train, quarantine, pending)


def _with_review_meta(record: Record, log: ReviewLog, status: str) -> Record:
    out = dict(record)
    latest = log.latest(record["qa_id"])
    out["review_status"] = status
    out["review_event_id"] = latest["event_id"] if latest else None
    out["reviewer"] = latest["reviewer"] if latest else None
    return out


def trace(qa_id: str, log: ReviewLog, records: Iterable[Record]) -> Dict[str, Any]:
    """Full lineage for a row: reviewer events + the source evidence it cites."""
    rec = next((r for r in records if r["qa_id"] == qa_id), None)
    events = log.events_for(qa_id)
    evidence = (rec.get("evidence") if rec else None) or []
    status = log.status_of(qa_id)
    return {
        "qa_id": qa_id,
        "review_status": status,
        "approved": status in _APPROVED_STATES,
        "events": events,
        "evidence": [{"document_sha256": e.get("document_sha256"),
                      "quote_sha256": e.get("quote_sha256"),
                      "page": e.get("page"), "element_id": e.get("element_id")}
                     for e in evidence],
        "traceable": bool(events) and bool(evidence),
    }


def redact_for_report(record: Record, has_source_access: bool) -> Record:
    """Drop verbatim source snippets for readers without source access.

    Evidence *addresses* (hashes, page, element id) are kept for auditing; the
    literal ``quote`` text is removed so a report cannot leak document content.
    """
    if has_source_access:
        return copy.deepcopy(record)
    out = copy.deepcopy(record)
    red = []
    for e in out.get("evidence", []) or []:
        e = dict(e)
        if "quote" in e:
            e["quote"] = None
            e["quote_redacted"] = True
        red.append(e)
    out["evidence"] = red
    out["_source_redacted"] = True
    return out


# --------------------------------------------------------------------------- #
# loaders + CLI
# --------------------------------------------------------------------------- #
def load_records(path: str) -> List[Record]:
    out: List[Record] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: str, rows: List[Record]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _find(records: List[Record], qa_id: str) -> Record:
    rec = next((r for r in records if r["qa_id"] == qa_id), None)
    if rec is None:
        raise SystemExit(f"qa_id not found: {qa_id}")
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="review",
        description=f"{WORKFLOW_LABEL} — evidence-centered approval queue (local/single-user; "
                    "no authn/authz/audit-retention).")
    p.add_argument("--log", required=True, help="append-only review event JSONL")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _rec_arg(sp):
        sp.add_argument("--records", required=True, help="generated Q&A JSONL")

    sp = sub.add_parser("list"); _rec_arg(sp)
    for name in ("accept", "reject", "reopen"):
        sp = sub.add_parser(name); _rec_arg(sp)
        sp.add_argument("--qa-id", required=True)
        sp.add_argument("--reviewer", required=True)
        if name == "reject":
            sp.add_argument("--reason", required=True, choices=REJECT_REASONS)
        sp.add_argument("--note", default=None)
    sp = sub.add_parser("edit"); _rec_arg(sp)
    sp.add_argument("--qa-id", required=True)
    sp.add_argument("--reviewer", required=True)
    sp.add_argument("--field", required=True)
    sp.add_argument("--value", required=True)
    sp.add_argument("--note", default=None)
    sp = sub.add_parser("export"); _rec_arg(sp)
    sp.add_argument("--out", required=True, help="output directory")
    sp.add_argument("--no-source-selection", action="store_true")
    sp = sub.add_parser("trace"); _rec_arg(sp)
    sp.add_argument("--qa-id", required=True)
    sub.add_parser("verify")

    args = p.parse_args(argv)

    if args.cmd == "verify":
        ReviewLog.load(args.log).verify()
        print(f"[review] OK — {args.log} chain + hashes verified.")
        return 0

    log = ReviewLog.load(args.log)

    if args.cmd == "list":
        records = load_records(args.records)
        from collections import Counter
        proj = log.project(records)
        counts = Counter(proj.values())
        print(f"[{WORKFLOW_LABEL}] {args.records}")
        for st in ("owner_review_pending", "approved", "edited", "rejected"):
            print(f"  {st}: {counts.get(st, 0)}")
        pend = [q for q, s in proj.items() if s == "owner_review_pending"]
        if pend:
            print("  pending qa_ids:", ", ".join(sorted(pend)[:20]))
        return 0

    if args.cmd in ("accept", "reject", "reopen", "edit"):
        records = load_records(args.records)
        rec = _find(records, args.qa_id)
        if args.cmd == "accept":
            ev = accept(log, rec, args.reviewer, note=args.note)
        elif args.cmd == "reject":
            ev = reject(log, rec, args.reviewer, args.reason, note=args.note)
        elif args.cmd == "reopen":
            ev = reopen(log, rec, args.reviewer, note=args.note)
        else:
            ev = edit(log, rec, args.reviewer, {args.field: args.value}, note=args.note)
        log.save(args.log)
        print(f"[review] {args.cmd} {args.qa_id} -> {log.status_of(args.qa_id)} "
              f"(event {ev['event_id']})")
        return 0

    if args.cmd == "trace":
        records = load_records(args.records)
        print(json.dumps(trace(args.qa_id, log, records), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "export":
        records = load_records(args.records)
        exp = export_training(records, log,
                              apply_source_selection=not args.no_source_selection)
        _write_jsonl(os.path.join(args.out, "train_approved.jsonl"), exp.train)
        _write_jsonl(os.path.join(args.out, "quarantine.jsonl"), exp.quarantine)
        _write_jsonl(os.path.join(args.out, "pending.jsonl"), exp.pending)
        print(f"[{WORKFLOW_LABEL}] export -> {args.out}: {exp.counts()}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
