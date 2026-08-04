"""Korean PII baseline detector + mechanical-fake (canary) validator.

This is a *baseline* fail-closed gate, deliberately NOT presented as complete
financial-PII detection (P0-9 item 4). It matches high-risk Korean identifier
*shapes* so restricted documents can be blocked before any external egress.

Two responsibilities are kept separate:

* :func:`detect` finds identifier-shaped spans (so the gate actually triggers on
  the synthetic canaries used in fixtures), and
* :func:`is_mechanically_fake` proves a matched span is a non-routable synthetic
  canary (Luhn-invalid card, impossible RRN date, reserved phone block, reserved
  example email domain). Fixtures rely on this so the PII gate can be exercised
  without embedding a single real, valid identifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import List

_PATTERNS = {
    "rrn": re.compile(r"\b(\d{6})-(\d{7})\b"),                 # 주민등록번호
    "phone": re.compile(r"\b01[016789]-?\d{3,4}-?\d{4}\b"),     # 휴대폰
    "card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),          # 카드번호
    "account": re.compile(r"\b\d{2,6}-\d{2,6}-\d{2,7}\b"),      # 계좌(대략)
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}

_RESERVED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")


@dataclass
class PIIHit:
    kind: str
    value: str
    start: int
    end: int
    mechanically_fake: bool


def _luhn_ok(digits: str) -> bool:
    ds = [int(c) for c in re.sub(r"\D", "", digits)]
    if len(ds) < 12:
        return False
    checksum = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _rrn_date_valid(six: str) -> bool:
    yy, mm, dd = int(six[:2]), int(six[2:4]), int(six[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return False
    for century in (1900, 2000):
        try:
            date(century + yy, mm, dd)
            return True
        except ValueError:
            continue
    return False


def is_mechanically_fake(kind: str, value: str) -> bool:
    """True when the identifier is provably non-real (safe synthetic canary)."""
    if kind == "email":
        return any(value.lower().endswith("@" + d) or value.lower().endswith("." + d) for d in _RESERVED_EMAIL_DOMAINS)
    if kind == "card":
        return not _luhn_ok(value)
    if kind == "rrn":
        m = _PATTERNS["rrn"].search(value)
        if not m:
            return True
        return not _rrn_date_valid(m.group(1))
    if kind == "phone":
        # reserved example block: xxx-555-01xx style or all-zero subscriber
        digits = re.sub(r"\D", "", value)
        return "55501" in digits or digits.endswith("0000000") or "00000000" in digits
    if kind == "account":
        return value.replace("-", "").count("0") >= max(4, len(value.replace("-", "")) - 2)
    return False


# Higher-priority kinds win when spans overlap (a 16-digit card must not be
# re-interpreted as a lower-priority "account" sub-span).
_PRIORITY = {"rrn": 0, "card": 1, "phone": 2, "email": 3, "account": 4}


def detect(text: str) -> List[PIIHit]:
    raw: List[PIIHit] = []
    for kind, rx in _PATTERNS.items():
        for m in rx.finditer(text or ""):
            raw.append(PIIHit(kind, m.group(0), m.start(), m.end(), is_mechanically_fake(kind, m.group(0))))

    # Resolve overlaps by (priority, longer span) — drop lower-priority overlaps.
    raw.sort(key=lambda h: (_PRIORITY.get(h.kind, 9), -(h.end - h.start), h.start))
    accepted: List[PIIHit] = []
    for h in raw:
        if any(h.start < a.end and a.start < h.end for a in accepted):
            continue
        accepted.append(h)
    accepted.sort(key=lambda h: h.start)
    return accepted


def has_real_pii(text: str) -> bool:
    """True if any detected identifier is NOT a mechanically-fake canary."""
    return any(not h.mechanically_fake for h in detect(text))


def redact(text: str) -> str:
    out = text or ""
    for kind, rx in _PATTERNS.items():
        out = rx.sub(f"[REDACTED_{kind.upper()}]", out)
    return out
