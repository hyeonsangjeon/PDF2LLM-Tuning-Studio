"""P1-1: frozen selection_dev vs final_holdout separation — tests.

The mandated guarantees:
- the two splits' example-ID intersection is empty (structural, network-free);
- the committed manifest matches the configured disjoint slices;
- a final_holdout ID entering a training/selection/export set raises
  ``HoldoutLeakageError`` (so CI fails if the guard is ever removed).
"""

import json
import os

import pytest

from quantization import data_korquad as D
from quantization import splits as S


def _cfg():
    return D.load_config()


# Synthetic ordering that stands in for the seed-shuffled validation IDs; the
# intersection/guard properties hold structurally, without any network access.
_SYNTH = [f"v{i}" for i in range(5774)]


def test_config_default_bounds():
    cfg = _cfg()
    assert S.split_bounds(cfg, S.SELECTION_DEV) == (0, 800)
    assert S.split_bounds(cfg, S.FINAL_HOLDOUT) == (800, 1800)


def test_structural_intersection_is_empty():
    cfg = _cfg()
    dev = set(S.slice_ids(_SYNTH, cfg, S.SELECTION_DEV))
    final = set(S.slice_ids(_SYNTH, cfg, S.FINAL_HOLDOUT))
    assert len(dev) == 800 and len(final) == 1000
    assert dev.isdisjoint(final)
    assert dev & final == set()


def test_build_manifest_disjoint_shapes():
    man = S.build_manifest(_cfg(), all_ids=_SYNTH)
    assert man["disjoint"] is True
    assert man["intersection_size"] == 0
    assert man["splits"][S.SELECTION_DEV]["slice"] == [0, 800]
    assert man["splits"][S.SELECTION_DEV]["n"] == 800
    assert man["splits"][S.FINAL_HOLDOUT]["slice"] == [800, 1800]
    assert man["splits"][S.FINAL_HOLDOUT]["n"] == 1000
    assert man["splits"][S.SELECTION_DEV]["id_sha256"] != man["splits"][S.FINAL_HOLDOUT]["id_sha256"]


def test_committed_manifest_matches_config():
    path = os.path.join(os.path.dirname(S.__file__), "results", "split_manifest.json")
    assert os.path.exists(path), "results/split_manifest.json must be committed"
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    cfg = _cfg()
    assert man["disjoint"] is True and man["intersection_size"] == 0
    assert man["splits"][S.SELECTION_DEV]["slice"] == list(S.split_bounds(cfg, S.SELECTION_DEV))
    assert man["splits"][S.FINAL_HOLDOUT]["slice"] == list(S.split_bounds(cfg, S.FINAL_HOLDOUT))
    assert man["splits"][S.SELECTION_DEV]["n"] == 800
    assert man["splits"][S.FINAL_HOLDOUT]["n"] == 1000
    assert man["policy"] == "frozen_policy_holdout"


def test_holdout_leakage_guard_raises_on_planted_final_id():
    cfg = _cfg()
    final_ids = set(S.slice_ids(_SYNTH, cfg, S.FINAL_HOLDOUT))
    # A clean training/selection set (selection_dev) must pass.
    clean = S.slice_ids(_SYNTH, cfg, S.SELECTION_DEV)
    S.assert_no_holdout_leakage(clean, final_ids)  # no raise
    # Planting one frozen final ID into a training/selection set fails CI.
    poisoned = clean + [next(iter(final_ids))]
    with pytest.raises(S.HoldoutLeakageError):
        S.assert_no_holdout_leakage(poisoned, final_ids)


def test_assert_config_splits_disjoint_detects_overlap():
    cfg = _cfg()
    S.assert_config_splits_disjoint(cfg)  # default config is disjoint
    bad = json.loads(json.dumps(cfg))     # deep copy
    bad["data"]["splits"] = {"selection_dev": {"start": 0, "size": 1000},
                             "final_holdout": {"start": 800, "size": 1000}}
    with pytest.raises(S.HoldoutLeakageError):
        S.assert_config_splits_disjoint(bad)


def test_id_list_sha256_is_order_independent():
    a = S.id_list_sha256(["b", "a", "c"])
    b = S.id_list_sha256(["c", "b", "a"])
    assert a == b
    assert a != S.id_list_sha256(["a", "b"])


def test_manifest_reproduces_from_korquad():
    """Network-gated: if KorQuAD loads with the manifest's datasets version, the
    committed ID hashes must reproduce exactly."""
    path = os.path.join(os.path.dirname(S.__file__), "results", "split_manifest.json")
    with open(path, encoding="utf-8") as fh:
        committed = json.load(fh)
    try:
        import datasets
    except Exception:
        pytest.skip("datasets not installed")
    if getattr(datasets, "__version__", None) != committed.get("datasets_version"):
        pytest.skip("datasets version differs from the manifest's; shuffle order may differ")
    try:
        all_ids = S.shuffled_validation_ids(_cfg())
    except Exception as exc:  # offline / hub error
        pytest.skip(f"KorQuAD unavailable: {type(exc).__name__}")
    problems = S.verify_manifest(_cfg(), path, all_ids=all_ids)
    assert problems == [], problems
