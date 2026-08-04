"""P1-1: frozen selection_dev vs final_holdout separation for the KorQuAD track.

Model/prompt/hyperparameter selection and the final comparison previously drew
from the **same** seed-shuffled slice of the official KorQuAD validation split
(base-select used ``[0:800]``, the final eval used ``[0:1000]`` — they overlap).
That lets selection decisions leak into the headline numbers.

This module defines two **disjoint** slices over one fixed ordering
(``ds["validation"].shuffle(seed=data.seed)``) and the guard rails around them:

* ``selection_dev``  = ``[0:800]``    — base-model / prompt / hyperparameter tuning.
* ``final_holdout``  = ``[800:1800]`` — *frozen policy holdout*: evaluation command
  and release gate **only**.

Because KorQuAD labels are already public, this is a **frozen policy holdout**, not
a ``sealed`` / ``unseen`` test set: the barrier is a code-path allowlist, **not** a
security boundary against a human reading the labels outside the repository.

Guarantees exercised by ``tests/test_splits.py``:

- the two splits' example-ID intersection is **empty** (disjoint index ranges);
- a manifest records each split's slice bounds, count and ID-list SHA-256;
- placing a ``final_holdout`` ID into a training / selection / export set raises
  ``HoldoutLeakageError`` (so CI fails) — final IDs are excluded from those inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SELECTION_DEV = "selection_dev"
FINAL_HOLDOUT = "final_holdout"
SPLIT_NAMES = (SELECTION_DEV, FINAL_HOLDOUT)

# Default slice bounds when ``config.yaml`` does not specify ``data.splits``.
_DEFAULT_SPLITS: Dict[str, Dict[str, int]] = {
    SELECTION_DEV: {"start": 0, "size": 800},
    FINAL_HOLDOUT: {"start": 800, "size": 1000},
}

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "results", "split_manifest.json")


class HoldoutLeakageError(Exception):
    """Raised when a frozen ``final_holdout`` ID reaches a non-eval input set."""


# --------------------------------------------------------------------------- #
# slice resolution
# --------------------------------------------------------------------------- #
def resolve_split(cfg: Dict[str, Any], name: str) -> Tuple[int, int]:
    """Return ``(start, size)`` for a named split from cfg (or the defaults)."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    splits = (cfg.get("data", {}) or {}).get("splits") or _DEFAULT_SPLITS
    spec = splits.get(name) or _DEFAULT_SPLITS[name]
    start = int(spec.get("start", _DEFAULT_SPLITS[name]["start"]))
    size = int(spec.get("size", _DEFAULT_SPLITS[name]["size"]))
    return start, size


def split_bounds(cfg: Dict[str, Any], name: str) -> Tuple[int, int]:
    """Return the half-open index range ``[start, end)`` for a named split."""
    start, size = resolve_split(cfg, name)
    return start, start + size


def assert_config_splits_disjoint(cfg: Dict[str, Any]) -> None:
    """Fail fast if the configured selection/holdout index ranges overlap."""
    s0, e0 = split_bounds(cfg, SELECTION_DEV)
    s1, e1 = split_bounds(cfg, FINAL_HOLDOUT)
    if max(s0, s1) < min(e0, e1):
        raise HoldoutLeakageError(
            f"configured splits overlap: {SELECTION_DEV}=[{s0}:{e0}) "
            f"{FINAL_HOLDOUT}=[{s1}:{e1}) — final holdout must be disjoint")


# --------------------------------------------------------------------------- #
# real KorQuAD ordering (network-dependent) + structural helpers
# --------------------------------------------------------------------------- #
def shuffled_validation_ids(cfg: Dict[str, Any]) -> List[str]:
    """The fixed, seed-shuffled ordering of official-validation example IDs.

    Mirrors exactly how ``load_slices`` / ``load_korquad`` order the validation
    split (``shuffle(seed=data.seed)``), so manifest IDs match the rows a run
    actually evaluates. Requires the dataset (network) at call time.
    """
    from datasets import load_dataset

    dcfg = cfg["data"]
    seed = int(dcfg.get("seed", 42))
    val = load_dataset(dcfg["dataset"], split="validation").shuffle(seed=seed)
    return [str(r) for r in val["id"]]


def slice_ids(all_ids: Sequence[str], cfg: Dict[str, Any], name: str) -> List[str]:
    """Slice a full ordering down to a named split's IDs."""
    start, end = split_bounds(cfg, name)
    return list(all_ids[start:min(end, len(all_ids))])


def id_list_sha256(ids: Iterable[str]) -> str:
    """Order-independent hash of an example-ID set (sorted, newline-joined)."""
    h = hashlib.sha256()
    h.update("\n".join(sorted(str(i) for i in ids)).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# leakage guard (the CI-failing check)
# --------------------------------------------------------------------------- #
def assert_no_holdout_leakage(candidate_ids: Iterable[str], final_ids: Iterable[str],
                              *, where: str = "training/selection/export") -> None:
    """Raise ``HoldoutLeakageError`` if any candidate ID is a frozen final ID."""
    final = {str(x) for x in final_ids}
    offenders = sorted({str(i) for i in candidate_ids if str(i) in final})
    if offenders:
        raise HoldoutLeakageError(
            f"frozen final_holdout IDs present in {where}: " + ", ".join(offenders[:10])
            + (" ..." if len(offenders) > 10 else ""))


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def build_manifest(cfg: Dict[str, Any], all_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Build the split manifest. Loads KorQuAD if ``all_ids`` is not supplied."""
    assert_config_splits_disjoint(cfg)
    if all_ids is None:
        all_ids = shuffled_validation_ids(cfg)

    dcfg = cfg["data"]
    per_split: Dict[str, Any] = {}
    id_sets: Dict[str, set] = {}
    for name in SPLIT_NAMES:
        start, end = split_bounds(cfg, name)
        ids = slice_ids(all_ids, cfg, name)
        id_sets[name] = set(ids)
        per_split[name] = {"slice": [start, end], "n": len(ids),
                           "id_sha256": id_list_sha256(ids)}

    inter = sorted(id_sets[SELECTION_DEV] & id_sets[FINAL_HOLDOUT])
    try:
        import datasets as _d
        dsv = getattr(_d, "__version__", None)
    except Exception:
        dsv = None

    return {
        "policy": "frozen_policy_holdout",
        "note": ("KorQuAD labels are public; this is a code-path allowlist, not a "
                 "security boundary against human reading of labels."),
        "dataset": dcfg["dataset"],
        "seed": int(dcfg.get("seed", 42)),
        "n_validation": len(all_ids),
        "datasets_version": dsv,
        "generated_by": "quantization.splits",
        "splits": per_split,
        "intersection_size": len(inter),
        "disjoint": len(inter) == 0,
    }


def write_manifest(cfg: Dict[str, Any], path: str = _MANIFEST_PATH,
                   all_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    man = build_manifest(cfg, all_ids=all_ids)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return man


def load_manifest(path: str = _MANIFEST_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_manifest(cfg: Dict[str, Any], path: str = _MANIFEST_PATH,
                    all_ids: Optional[Sequence[str]] = None) -> List[str]:
    """Recompute the manifest and diff against the committed file.

    Returns a list of human-readable mismatches (empty == OK). Recomputation of
    the real ID hashes needs the dataset; pass ``all_ids`` to avoid network.
    """
    committed = load_manifest(path)
    fresh = build_manifest(cfg, all_ids=all_ids)
    problems: List[str] = []
    if not fresh["disjoint"]:
        problems.append(f"splits overlap (intersection={fresh['intersection_size']})")
    for name in SPLIT_NAMES:
        c, f = committed["splits"].get(name, {}), fresh["splits"][name]
        if c.get("slice") != f["slice"]:
            problems.append(f"{name} slice {c.get('slice')} != {f['slice']}")
        if c.get("id_sha256") != f["id_sha256"]:
            problems.append(f"{name} id_sha256 mismatch")
    return problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_cfg(smoke: bool):
    from . import data_korquad as D
    return D.load_config(force_mode="cpu" if smoke else None)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="quantization.splits",
        description="Frozen selection_dev / final_holdout split manifest + guard (P1-1).")
    ap.add_argument("--write", action="store_true", help="(re)write results/split_manifest.json")
    ap.add_argument("--check", action="store_true",
                    help="recompute from KorQuAD and diff the committed manifest (needs network)")
    ap.add_argument("--path", default=_MANIFEST_PATH)
    args = ap.parse_args(argv)

    cfg = _load_cfg(smoke=False)
    if args.write:
        man = write_manifest(cfg, args.path)
        print(f"[splits] wrote {args.path}: "
              f"{SELECTION_DEV}={man['splits'][SELECTION_DEV]['slice']} "
              f"{FINAL_HOLDOUT}={man['splits'][FINAL_HOLDOUT]['slice']} "
              f"disjoint={man['disjoint']}")
        return 0
    if args.check:
        problems = verify_manifest(cfg, args.path)
        if problems:
            print("[splits] MANIFEST MISMATCH:")
            for p in problems:
                print("  -", p)
            return 1
        print("[splits] OK — committed manifest reproduces; splits disjoint.")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
