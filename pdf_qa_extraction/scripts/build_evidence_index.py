#!/usr/bin/env python3
"""Evidence index (spec P0-11) — bind every strong README claim to the command +
raw artifact that proves it, and fail CI when they drift.

``--check`` (CI gate) verifies, for each ledger claim:
  * ``status`` is one of the allowed values;
  * a numeric ``expected`` matches its source-JSON value (resolved by JSON Pointer,
    rounded to the README's decimal places, within tolerance);
  * a string ``expected`` is present in the source JSON at the pointer;
  * the ``expected`` literal actually appears in the referenced README (README↔JSON);
  * ``planned`` features carry **no** measured number (they cannot masquerade as done);
  * ``docs/EVIDENCE.md`` is up to date (regenerate with ``--emit``).

``--emit`` regenerates ``docs/EVIDENCE.md`` (capability status table + per-claim
code→input→raw-result path) from the ledger.

Runs offline, no third-party deps beyond PyYAML (already a workflow dep).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # pdf_qa_extraction/
DEFAULT_LEDGER = os.path.join(_ROOT, "docs", "evidence_ledger.yaml")
DEFAULT_EVIDENCE_MD = os.path.join(_ROOT, "docs", "EVIDENCE.md")

ALLOWED_STATUS = ("ci_verified", "recorded_hardware_run",
                  "historical_not_reproduced", "planned")
STATUS_LABEL = {
    "ci_verified": "✅ ci_verified",
    "recorded_hardware_run": "🟢 recorded_hardware_run",
    "historical_not_reproduced": "🟡 historical_not_reproduced",
    "planned": "⚪ planned",
}
_DEFAULT_TOL = 0.01


# --------------------------------------------------------------------------- #
# JSON Pointer (RFC 6901) + helpers
# --------------------------------------------------------------------------- #
def resolve_pointer(doc: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return doc
    cur = doc
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(token)]
        elif isinstance(cur, dict):
            cur = cur[token]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {token!r}")
    return cur


def _decimals(s: str) -> int:
    return len(s.split(".", 1)[1]) if "." in s else 0


def load_ledger(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "claims" not in data:
        raise ValueError("ledger must be a mapping with a 'claims' list")
    return data


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def verify_claim(claim: Dict[str, Any], root: str, tol: float = _DEFAULT_TOL) -> List[str]:
    """Return a list of human-readable error strings (empty == claim OK)."""
    errs: List[str] = []
    cid = claim.get("id", "<no-id>")
    for req in ("id", "capability", "status"):
        if not claim.get(req):
            errs.append(f"[{cid}] missing required field {req!r}")
    status = claim.get("status")
    if status not in ALLOWED_STATUS:
        errs.append(f"[{cid}] status {status!r} not in {ALLOWED_STATUS}")

    has_measure = any(k in claim for k in ("expected", "pointer"))
    if status == "planned" and has_measure:
        errs.append(f"[{cid}] planned feature must NOT carry a measured "
                    f"number/pointer (found one)")

    expected = claim.get("expected")
    pointer = claim.get("pointer")
    source = claim.get("source")

    if expected is not None and source and pointer:
        src_path = os.path.join(root, source)
        if not os.path.isfile(src_path):
            errs.append(f"[{cid}] source JSON not found: {source}")
        else:
            try:
                doc = json.loads(_read_text(src_path))
                val = resolve_pointer(doc, pointer)
            except Exception as exc:  # noqa: BLE001
                errs.append(f"[{cid}] pointer {pointer!r} failed on {source}: {exc}")
                val = None
            if val is not None:
                exp_str = str(expected)
                num: Optional[float] = None
                try:
                    num = float(exp_str)
                except ValueError:
                    num = None
                if num is not None and isinstance(val, (int, float)):
                    if round(float(val), _decimals(exp_str)) != round(num, _decimals(exp_str)) \
                            and abs(float(val) - num) > tol:
                        errs.append(f"[{cid}] JSON {val} != expected {exp_str} "
                                    f"({source}{pointer})")
                else:  # string expectation: substring match against the JSON value
                    if exp_str not in str(val):
                        errs.append(f"[{cid}] expected {exp_str!r} not in JSON value "
                                    f"{val!r} ({source}{pointer})")

    # README ↔ ledger: the literal expected string must appear in the README.
    readme = claim.get("readme")
    if expected is not None and readme:
        rd_path = os.path.join(root, readme)
        if not os.path.isfile(rd_path):
            errs.append(f"[{cid}] readme not found: {readme}")
        elif str(expected) not in _read_text(rd_path):
            errs.append(f"[{cid}] expected {str(expected)!r} not found in {readme}")

    return errs


def verify_all(ledger: Dict[str, Any], root: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    seen = set()
    for claim in ledger["claims"]:
        cid = claim.get("id")
        if cid in seen:
            errors.append(f"[{cid}] duplicate claim id")
        seen.add(cid)
        errors.extend(verify_claim(claim, root))
    return (not errors), errors


# --------------------------------------------------------------------------- #
# docs/EVIDENCE.md generation
# --------------------------------------------------------------------------- #
def render_evidence_md(ledger: Dict[str, Any]) -> str:
    claims = ledger["claims"]
    meta = ledger.get("meta", {})
    lines: List[str] = []
    lines.append("<!-- AUTO-GENERATED by scripts/build_evidence_index.py --emit — do not edit by hand. -->")
    lines.append("# EVIDENCE — claim → command → raw artifact")
    lines.append("")
    lines.append("Every strong quantitative or capability claim in the READMEs is registered "
                 "in [`docs/evidence_ledger.yaml`](evidence_ledger.yaml) and auto-verified by "
                 "`scripts/build_evidence_index.py --check` (a CI gate): the number here must "
                 "match its raw source JSON **and** appear in the referenced README, or CI fails.")
    lines.append("")
    if meta.get("note"):
        lines.append(f"> {meta['note'].strip()}")
        lines.append("")
    lines.append("**Status vocabulary** — `ci_verified` (green in CI now) · "
                 "`recorded_hardware_run` (one recorded A100 run, reproducible from "
                 "`config.yaml`) · `historical_not_reproduced` (frozen prior artifact, not "
                 "re-run) · `planned` (implemented or scoped, **no** measured result yet).")
    lines.append("")

    # 60-second path
    lines.append("## Follow one claim in 60 seconds")
    lines.append("")
    lines.append("```bash")
    lines.append("cd pdf_qa_extraction")
    lines.append("pip install -e \".[test]\"")
    lines.append("pdf2llm verify-demo                 # ci_verified golden path (EM/F1 == 1.0)")
    lines.append("python scripts/build_evidence_index.py --check   # every number below vs raw JSON")
    lines.append("```")
    lines.append("")

    # Capability status table
    lines.append("## Capability status")
    lines.append("")
    lines.append("| capability | status | command | raw artifact | value |")
    lines.append("|---|---|---|---|---|")
    for c in claims:
        cap = c.get("capability", c.get("id", ""))
        status = STATUS_LABEL.get(c.get("status"), c.get("status", ""))
        cmd = c.get("command", "")
        cmd_md = f"`{cmd}`" if cmd else "—"
        src = c.get("source")
        if src:
            ptr = c.get("pointer", "")
            src_md = f"[`{src}`]({_rellink(src)}){('`' + ptr + '`') if ptr else ''}"
        else:
            src_md = "—"
        exp = c.get("expected")
        val_md = f"**{exp}**" if exp is not None else "—"
        lines.append(f"| {cap} | {status} | {cmd_md} | {src_md} | {val_md} |")
    lines.append("")

    # Planned callout
    planned = [c for c in claims if c.get("status") == "planned"]
    if planned:
        lines.append("## Planned (no result claimed yet)")
        lines.append("")
        for c in planned:
            note = (c.get("note") or "").strip()
            lines.append(f"- **{c.get('capability')}** — {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _rellink(source: str) -> str:
    # EVIDENCE.md lives in docs/; sources are given relative to pdf_qa_extraction/.
    return os.path.relpath(source, "docs").replace(os.sep, "/")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="build_evidence_index",
        description="Verify README claims against raw JSON; regenerate docs/EVIDENCE.md.")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--root", default=_ROOT,
                    help="base dir that ledger paths are relative to (default: pdf_qa_extraction/)")
    ap.add_argument("--evidence-md", default=DEFAULT_EVIDENCE_MD)
    ap.add_argument("--check", action="store_true",
                    help="verify claims + that EVIDENCE.md is current; non-zero exit on failure")
    ap.add_argument("--emit", action="store_true", help="regenerate docs/EVIDENCE.md")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args([] if argv is None else argv)
    ledger = load_ledger(args.ledger)

    if args.emit:
        md = render_evidence_md(ledger)
        os.makedirs(os.path.dirname(args.evidence_md), exist_ok=True)
        with open(args.evidence_md, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"[evidence] wrote {args.evidence_md} ({len(ledger['claims'])} claims)")
        if not args.check:
            return 0

    ok, errors = verify_all(ledger, args.root)
    if args.check or not args.emit:
        # EVIDENCE.md freshness
        want = render_evidence_md(ledger)
        if not os.path.isfile(args.evidence_md):
            ok = False
            errors.append(f"[docs] {args.evidence_md} missing — run --emit")
        elif _read_text(args.evidence_md) != want:
            ok = False
            errors.append(f"[docs] {os.path.relpath(args.evidence_md, args.root)} "
                          f"is stale — run: python scripts/build_evidence_index.py --emit")

    if errors:
        print(f"[evidence] FAIL — {len(errors)} problem(s):")
        for e in errors:
            print("  -", e)
        return 1
    print(f"[evidence] OK — {len(ledger['claims'])} claims verified against raw JSON + READMEs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
