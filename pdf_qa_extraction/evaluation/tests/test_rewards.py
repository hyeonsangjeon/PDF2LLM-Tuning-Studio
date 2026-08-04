"""P2-2: unit tests for the programmatic-verifier reward components.

The rewards must exist and be tested *before* any RL is added. These use only
public/synthetic inputs: a few committed gold records from the public finance
demo fixture, plus small hand-built records. No GPU, no network, no RL run.
"""

import copy
import json
import os
import sys
from pathlib import Path

import pytest

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from evaluation import rewards as R  # noqa: E402

_GOLD = (Path(_PKG) / "workflows" / "pdf_native_post_training"
         / "public_finance_demo" / "gold_qa.jsonl")


def _gold_records():
    with open(_GOLD, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _find(records, category, answerable=None):
    for r in records:
        if r.get("category") == category and (answerable is None or r.get("answerable") == answerable):
            return copy.deepcopy(r)
    return None


# --------------------------------------------------------------------------- #
# Status / registry
# --------------------------------------------------------------------------- #
def test_rl_status_is_planned_not_run():
    assert R.RL_STATUS == "planned"
    assert R.RL_METHOD_NAME == "programmatic-verifier RL"


def test_rl_plan_doc_is_honest_and_planned():
    plan = Path(_PKG).parent / "docs" / "RL_EXPERIMENT_PLAN.md"
    assert plan.is_file(), "docs/RL_EXPERIMENT_PLAN.md must exist"
    text = plan.read_text(encoding="utf-8")
    low = text.lower()
    assert "planned" in low
    assert "programmatic-verifier rl" in low
    # Must not claim RL was actually run or produced results.
    for bad in ("grpo improved", "rl results show", "we trained with grpo",
                "rlhf experience", "ppo improved"):
        assert bad not in low, f"over-claim in RL plan: {bad!r}"
    # Must keep the config gated (not shipped).
    assert "rl-feasibility.yaml" in low
    assert "gated" in low


def test_reward_cards_cover_every_component():
    names = set(R.available_rewards())
    # Every component used by score_record has a card, plus the length guard.
    for comp in R.DEFAULT_WEIGHTS:
        assert comp in names, f"missing RewardCard for {comp}"
    assert "length_penalty" in names
    # Cards are well-formed.
    for c in R.REWARD_CARDS:
        assert c.name and c.definition and c.range and c.failure_modes


# --------------------------------------------------------------------------- #
# evidence_validity
# --------------------------------------------------------------------------- #
def test_evidence_validity_self_consistent_gold_scores_one():
    rec = _find(_gold_records(), "numeric_exact", answerable=True)
    assert rec is not None
    assert R.reward_evidence_validity(rec) == 1.0


def test_evidence_validity_detects_hash_tamper():
    rec = _find(_gold_records(), "numeric_exact", answerable=True)
    rec["evidence"][0]["quote_sha256"] = "0" * 64
    assert R.reward_evidence_validity(rec) == 0.0


def test_evidence_validity_answerable_without_evidence_is_zero():
    rec = {"answer": "1,250억 원", "answerable": True, "category": "single_fact",
           "evidence": []}
    assert R.reward_evidence_validity(rec) == 0.0


# --------------------------------------------------------------------------- #
# numeric_consistency
# --------------------------------------------------------------------------- #
def test_numeric_consistency_grounded_vs_hallucinated():
    grounded = {
        "answer": "매출은 1,250억 원입니다.", "answerable": True, "category": "numeric_exact",
        "evidence": [{"quote": "당기 매출은 1,250억 원으로 집계되었다."}],
    }
    assert R.reward_numeric_consistency(grounded) == 1.0
    hallucinated = copy.deepcopy(grounded)
    hallucinated["answer"] = "매출은 9,999억 원입니다."
    assert R.reward_numeric_consistency(hallucinated) == 0.0


def test_numeric_consistency_not_applicable_for_unanswerable():
    rec = {"answer": "문서에서 확인할 수 없습니다.", "answerable": False,
           "category": "unanswerable", "evidence": []}
    assert R.reward_numeric_consistency(rec) == 1.0


# --------------------------------------------------------------------------- #
# schema_compliance
# --------------------------------------------------------------------------- #
def test_schema_compliance_gold_passes_and_broken_fails():
    pytest.importorskip("jsonschema")
    rec = _find(_gold_records(), "numeric_exact", answerable=True)
    assert R.reward_schema_compliance(rec) == 1.0
    broken = copy.deepcopy(rec)
    broken.pop("qa_id")  # required field
    assert R.reward_schema_compliance(broken) == 0.0


# --------------------------------------------------------------------------- #
# calculation
# --------------------------------------------------------------------------- #
def test_calculation_verifies_declared_computation():
    rec = {"category": "calculation", "answer": "영업이익률은 20% 입니다.",
           "computation": {"op": "percent", "operands": [250, 1250]}}
    assert R.reward_calculation(rec) == 1.0
    wrong = copy.deepcopy(rec)
    wrong["answer"] = "영업이익률은 25% 입니다."
    assert R.reward_calculation(wrong) == 0.0


def test_calculation_gold_value_and_na():
    ok = {"category": "calculation", "answer": "합계는 300", "answer_value": 300}
    assert R.reward_calculation(ok) == 1.0
    # No verifier spec -> not applicable (never fabricates a ground truth).
    na = {"category": "calculation", "answer": "합계는 300"}
    assert R.reward_calculation(na) == 1.0
    # Non-calculation category -> not applicable.
    assert R.reward_calculation({"category": "single_fact", "answer": "x"}) == 1.0


# --------------------------------------------------------------------------- #
# abstention
# --------------------------------------------------------------------------- #
def test_abstention_rewards_correct_refusal_and_penalises_fabrication():
    unans = {"answer": "문서에서 확인할 수 없습니다.", "answerable": False,
             "category": "unanswerable", "evidence": []}
    assert R.reward_abstention(unans) == 1.0
    # Fabricated citation on an unanswerable question.
    fab = copy.deepcopy(unans)
    fab["evidence"] = [{"quote": "made up"}]
    assert R.reward_abstention(fab) == 0.0
    # Confidently answering an unanswerable question.
    confident = copy.deepcopy(unans)
    confident["answer"] = "부채비율은 42% 입니다."
    assert R.reward_abstention(confident) == 0.0


def test_abstention_penalises_over_refusal_on_answerable():
    rec = {"answer": "제공된 정보로는 알 수 없습니다.", "answerable": True,
           "category": "single_fact", "evidence": [{"quote": "..."}]}
    assert R.reward_abstention(rec) == 0.0
    ok = {"answer": "1,250억 원입니다.", "answerable": True, "category": "single_fact",
          "evidence": [{"quote": "1,250억 원"}]}
    assert R.reward_abstention(ok) == 1.0


# --------------------------------------------------------------------------- #
# pii_nonexposure
# --------------------------------------------------------------------------- #
def test_pii_nonexposure_flags_real_pii():
    clean = {"answer": "매출은 1,250억 원입니다.", "evidence": []}
    assert R.reward_pii_nonexposure(clean) == 1.0
    leaky = {"answer": "담당자 주민번호는 900101-1234567 입니다.", "evidence": []}
    assert R.reward_pii_nonexposure(leaky) == 0.0


# --------------------------------------------------------------------------- #
# version_recency
# --------------------------------------------------------------------------- #
def test_version_recency_latest_vs_stale():
    latest = {"source_status": "active", "document_version": "v2",
              "evidence": [{"document_version": "v2"}]}
    assert R.reward_version_recency(latest, latest_version="v2") == 1.0
    old = {"source_status": "active", "evidence": [{"document_version": "v1"}]}
    assert R.reward_version_recency(old, latest_version="v2") == 0.0
    revoked = {"source_status": "revoked", "evidence": [{"document_version": "v2"}]}
    assert R.reward_version_recency(revoked, latest_version="v2") == 0.0


# --------------------------------------------------------------------------- #
# length_penalty / reward hacking
# --------------------------------------------------------------------------- #
def test_length_penalty_flags_padding_and_copy_through():
    ev = [{"quote": "매출 1,250억"}]
    terse = {"answer": "1,250억 원", "answerable": True, "category": "single_fact",
             "evidence": ev}
    assert R.length_penalty(terse) == 0.0
    padded = {"answer": "매출" * 200, "answerable": True, "category": "single_fact",
              "evidence": ev}
    assert R.length_penalty(padded) > 0.5
    long_quote = "이 문장은 마흔 자를 훌쩍 넘는 충분히 긴 근거 인용문입니다 정말로 길어요"
    copy_through = {"answer": long_quote, "answerable": True, "category": "single_fact",
                    "evidence": [{"quote": long_quote}]}
    assert R.length_penalty(copy_through) == 1.0


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_score_record_high_for_clean_gold_low_for_hallucination():
    pytest.importorskip("jsonschema")
    records = _gold_records()
    good = _find(records, "numeric_exact", answerable=True)
    good_score = R.score_record(good, latest_version=good.get("document_version"))
    assert good_score["total"] >= 0.9
    assert set(good_score["components"]) == set(R.DEFAULT_WEIGHTS)

    bad = copy.deepcopy(good)
    bad["answer"] = "매출은 9,999억 원입니다."          # ungrounded number
    bad["evidence"][0]["quote_sha256"] = "0" * 64      # broken citation
    bad_score = R.score_record(bad)
    assert bad_score["total"] < good_score["total"]


def test_score_record_total_is_bounded():
    rec = {"answer": "x", "answerable": True, "category": "single_fact", "evidence": []}
    out = R.score_record(rec)
    assert 0.0 <= out["total"] <= 1.0
