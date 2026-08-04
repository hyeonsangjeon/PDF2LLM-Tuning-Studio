"""Normalized Korean EM / token-F1 scorer for demo eval (KorQuAD-style).

Kept tiny and dependency-free. This scores answer text against a gold answer; it
is deliberately separate from evidence-address integrity (which is mechanical).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

_PUNC = re.compile(r"[\s\.,!?\"'`~\-_/\\()\[\]{}:;·…“”‘’]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _PUNC.sub(" ", text)
    return " ".join(text.split()).strip().lower()


def _char_bag(text: str) -> List[str]:
    return list(normalize(text).replace(" ", ""))


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = _char_bag(pred)
    g = _char_bag(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common: Dict[str, int] = {}
    gg = list(g)
    for ch in p:
        if ch in gg:
            common[ch] = common.get(ch, 0) + 1
            gg.remove(ch)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p)
    rec = n / len(g)
    return 2 * prec * rec / (prec + rec)


def score_pairs(pairs: List[Dict[str, str]]) -> Dict[str, float]:
    """pairs: [{"pred":..,"gold":..}] -> {"em":.., "f1":.., "n":..}"""
    if not pairs:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    em = sum(exact_match(p["pred"], p["gold"]) for p in pairs) / len(pairs)
    f1 = sum(token_f1(p["pred"], p["gold"]) for p in pairs) / len(pairs)
    return {"em": round(em, 4), "f1": round(f1, 4), "n": len(pairs)}
