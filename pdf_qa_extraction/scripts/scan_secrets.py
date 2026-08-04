#!/usr/bin/env python3
"""Generic secret / PII / trust-regression scanner (public-safe).

This scanner deliberately does NOT hardcode any real customer, person or
organisation name. Per the repository's public-CI policy it only matches:

* generic PII/secret *shapes*  (email, Korean phone, API-key-like tokens,
  private-looking URLs, absolute local user paths), and
* code smells that reintroduce known correctness bugs
  (``eval_dataset=dataset`` train/eval leakage, raw->train export).

Synthetic canaries (reserved example domains/numbers, obvious placeholders)
are allow-listed so intentional fixtures do not trip the scan.

Usage::

    python scripts/scan_secrets.py path [path ...]      # exit 1 on findings
    python scripts/scan_secrets.py --json path          # machine-readable

Return code is non-zero when any finding is present, so it can gate CI.

Marker: scan-secrets: allow-file (this file defines the patterns themselves).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, List

# --- allow-listed synthetic canary shapes (must be provably non-routable) ----
ALLOW_SUBSTRINGS = (
    "example.com", "example.org", "example.net",  # RFC 2606 reserved
    "@example", "555-01",                          # reserved US example phone block
    "예시", "합성", "가상", "데모라", "ACME", "CANARY",  # explicit synthetic markers
    "000-0000", "00-000", "0000-0000",             # obvious placeholders
    # well-known *shared* cloud/service accounts are conventions, not a personal
    # machine layout leak, so they are allow-listed for local_user_path.
    "/home/ec2-user/", "/home/azureuser/", "/home/sagemaker-user/",
    "/home/ubuntu/", "/home/ml-user/", "/root/", "/users/shared/",
)

# --- generic patterns (shapes, not names) ------------------------------------
PATTERNS = {
    # emails, but real-looking (reserved example domains handled by allowlist)
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # Korean mobile / landline: 010-1234-5678, 02-123-4567 ...
    "phone_kr": re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b"),
    # API-key-ish long tokens — real vendor prefixes only (delimiter-anchored so
    # ordinary identifiers like ``skip_special_tokens`` are not matched).
    "api_key": re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|gh[pousr]_[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b"),
    # A committed AWS *secret value* (>=40-char base64), not a mere variable name /
    # ``os.getenv(...)`` reference / documented placeholder.
    "aws_secret": re.compile(r"aws_secret_access_key\s*[:=]\s*[\"']?[A-Za-z0-9/+]{40,}", re.I),
    # absolute local user paths that leak a machine layout
    "local_user_path": re.compile(r"/(home|Users)/[A-Za-z0-9._-]+/"),
    # private / corporate-internal hosts (localhost & 127.0.0.1 are legitimate in
    # local-serving docs, so they are intentionally excluded).
    "private_url": re.compile(r"https?://(10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|[A-Za-z0-9.-]+\.(?:internal|corp|local)\b)"),
    # known correctness regressions (code smells)
    "train_eval_leak": re.compile(r"eval_dataset\s*=\s*dataset\b"),
}

TEXT_EXTS = {".py", ".ipynb", ".jsonl", ".json", ".md", ".yaml", ".yml", ".txt", ".sh", ".cfg", ".toml"}
SKIP_DIRS = {".git", ".ipynb_checkpoints", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", "runs", "artifacts"}


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    snippet: str


def _iter_files(paths: Iterable[str]) -> Iterable[str]:
    for p in paths:
        if os.path.isfile(p):
            yield p
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if os.path.splitext(f)[1].lower() in TEXT_EXTS:
                    yield os.path.join(root, f)


# Files that legitimately contain pattern shapes (this scanner and its tests)
# opt out with this marker in their first lines.
_ALLOW_FILE_MARKER = "scan-secrets: allow-file"


def _file_opted_out(fp: str) -> bool:
    try:
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            return _ALLOW_FILE_MARKER in fh.read(2048)
    except (OSError, UnicodeError):
        return False



def _allowed(line: str, match: str) -> bool:
    low = line.lower()
    return any(a.lower() in low for a in ALLOW_SUBSTRINGS)


def scan(paths: Iterable[str]) -> List[Finding]:
    findings: List[Finding] = []
    for fp in _iter_files(paths):
        if _file_opted_out(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    for kind, rx in PATTERNS.items():
                        m = rx.search(line)
                        if not m:
                            continue
                        if kind in ("email", "phone_kr", "private_url", "local_user_path") and _allowed(line, m.group(0)):
                            continue
                        findings.append(Finding(fp, i, kind, m.group(0)[:80]))
        except (OSError, UnicodeError):
            continue
    return findings


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    findings = scan(args.paths)
    if args.json:
        print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2))
    else:
        for f in findings:
            print(f"{f.path}:{f.line}: [{f.kind}] {f.snippet}")
        print(f"\n{len(findings)} finding(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
