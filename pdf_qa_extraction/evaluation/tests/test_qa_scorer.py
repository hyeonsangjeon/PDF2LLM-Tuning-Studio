"""Meta-eval for the QA scorer (spec §6): *calibrate the judge* against the
human-labelled golden cases before trusting it.

Two tiers, both hermetic (no network / credentials at test time):

* **Layer 1 (deterministic)** is asserted directly on the committed sample
  datasets — REGISTER / FIRST_PERSON / LEADING_Q / FORMAT.
* **Layer 2 (judge)** is asserted via a :class:`ReplayJudge` over verdicts that
  were recorded once from a *real* gpt-4o judge (temperature 0). So the golden
  COHERENT / GROUNDED / strict-PASS assertions calibrate against genuine judge
  output, reproducibly, offline.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evaluation.qa_scorer import (  # noqa: E402
    DimResult,
    ReplayJudge,
    StubJudge,
    aggregate,
    check_first_person,
    check_format,
    check_leading_q,
    check_register,
    judge_key,
    load_pairs,
    load_rubric,
    normalize_pair,
    score_pairs,
)

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "fixtures")
SAMPLES = os.path.join(HERE, "..", "..", "data", "samples")
JUDGE_CACHE = os.path.join(FIXTURES, "judge_cache_gpt4o.json")
SOURCE_TXT = os.path.join(FIXTURES, "memoir_source_long.txt")

# The 검정 distortion appears at index 5 of every v4 run (숯검정이 되었다 flattened
# into "검정이 되었던 상황이었고" — a plain-register but incoherent compression).
KUMJUNG_INDEX = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rubric():
    return load_rubric()


@pytest.fixture(scope="module")
def source():
    with open(SOURCE_TXT, "r", encoding="utf-8") as handle:
        return handle.read().strip()


@pytest.fixture(scope="module")
def judge():
    return ReplayJudge.from_file(JUDGE_CACHE)


def sample(name):
    return load_pairs(os.path.join(SAMPLES, name))


# ===========================================================================
# Schema handling
# ===========================================================================
def test_normalize_accepts_both_schemas():
    a = normalize_pair({"QUESTION": "질문?", "ANSWER": "답."})
    b = normalize_pair({"instruction": "질문?", "output": "답."})
    c = normalize_pair({"question": "질문?", "answer": "답."})
    assert a["question"] == b["question"] == c["question"] == "질문?"
    assert a["answer"] == b["answer"] == c["answer"] == "답."


# ===========================================================================
# Layer 1 — REGISTER (deterministic honorific detection)
# ===========================================================================
def test_register_jamo_distinguishes_polite_from_plain(rubric):
    # 존댓말 (polite) -> FAIL
    for polite in [
        "그것은 사실입니다.",
        "정성껏 올렸습니다.",
        "지금도 생생합니다.",
        "무엇이었습니까?",
        "그리 하였어요.",
        "그렇지요.",
        "부질없는 눈물만 벗이 되었기 때문입니다.",
    ]:
        assert not check_register(polite, rubric).passed, polite
    # plain 문어체 -> PASS (특히 '아니다'는 '니다'로 끝나지만 ㅂ받침이 없어 통과)
    for plain in [
        "그것은 사실이 아니다.",
        "나는 그 말을 잊지 아니하였다.",
        "십 리 길을 매일 걸어 다녔다.",
        "꿈결같이 흘러갔음을 느꼈다.",
        "지켜 주기를 바라는 것이다.",
    ]:
        assert check_register(plain, rubric).passed, plain


def test_register_ignores_quoted_polite_line(rubric):
    # A quoted modern-polite line must not make the plain narration fail.
    answer = '손주가 "할아버지 왜 그래요?" 하고 물었을 때, 나는 허허 웃었다.'
    assert check_register(answer, rubric).passed


def test_golden_v4_run1_first5_register_fail(rubric):
    """§6: v4_run1의 앞 5쌍 → REGISTER FAIL(존댓말)."""
    pairs = sample("qa_memoirist_v4_run1.jsonl")
    for i in range(5):
        result = check_register(pairs[i]["answer"], rubric)
        assert not result.passed, f"run1[{i}] should be honorific: {result.reason}"
    # The remaining pairs are plain 문어체.
    for i in range(5, len(pairs)):
        assert check_register(pairs[i]["answer"], rubric).passed, i


def test_golden_v2_enlarged_register_fail_10of10(rubric):
    """§6: v2_enlarged 10쌍 → REGISTER FAIL 10/10."""
    pairs = sample("qa_memoirist_v2_enlarged.jsonl")
    fails = [i for i, p in enumerate(pairs) if not check_register(p["answer"], rubric).passed]
    assert len(fails) == 10, f"expected 10/10 honorific, got {len(fails)}: {fails}"


def test_v4_run2_run3_all_plain(rubric):
    for name in ["qa_memoirist_v4_run2.jsonl", "qa_memoirist_v4_run3.jsonl"]:
        pairs = sample(name)
        fails = [i for i, p in enumerate(pairs) if not check_register(p["answer"], rubric).passed]
        assert fails == [], f"{name} should be all-plain, honorific at {fails}"


def test_register_count_reproduces_manual_v1_v4(rubric):
    """Deterministic register tally must reproduce the manual 5/30 figure."""
    for prefix in ("v1", "v4"):
        total_fail = 0
        for r in (1, 2, 3):
            pairs = sample(f"qa_memoirist_{prefix}_run{r}.jsonl")
            total_fail += sum(1 for p in pairs if not check_register(p["answer"], rubric).passed)
        assert total_fail == 5, f"{prefix}: manual said 5/30 honorific, scorer says {total_fail}"


# ===========================================================================
# Layer 1 — FIRST_PERSON / LEADING_Q / FORMAT
# ===========================================================================
def test_first_person_detection(rubric):
    assert check_first_person("나는 그날을 잊지 못한다.", rubric).passed
    third = check_first_person("그는 그날을 잊지 못하였다.", rubric)
    assert not third.passed
    # No pronoun either way -> pass but WARN (deferred to judge).
    warn = check_first_person("부모 살아 계실 제 섬기기를 다하여라.", rubric)
    assert warn.passed and warn.warn


def test_leading_q_flags_synthesis_prompts(rubric):
    assert check_leading_q("가장 기억에 남는 순간은 무엇이었나요?", rubric).warn
    assert check_leading_q("그 일에서 어떤 교훈을 얻으셨습니까?", rubric).warn
    # A grounded, concrete question is not flagged.
    assert not check_leading_q("아홉 살에 처음 서당에 간 날은 어떠하였습니까?", rubric).warn


def test_format_rejects_empty_short_and_duplicate(rubric):
    seen = set()
    assert not check_format("", "", rubric, seen).passed
    assert not check_format("질문입니까?", "", rubric, seen).passed
    assert not check_format("짧다", "답변이다.", rubric, seen).passed  # question too short
    assert check_format("충분히 긴 질문입니까?", "충분히 긴 답이다.", rubric, seen).passed
    # Exact duplicate of the previous accepted pair -> rejected.
    assert not check_format("충분히 긴 질문입니까?", "충분히 긴 답이다.", rubric, seen).passed


# ===========================================================================
# Layer 2 — judge golden fixtures (via recorded gpt-4o verdicts)
# ===========================================================================
def test_judge_cache_covers_v4_runs(source, judge):
    """Guard: every v4 pair must be present in the replay cache (no misses)."""
    for name in [
        "qa_memoirist_v4_run1.jsonl",
        "qa_memoirist_v4_run2.jsonl",
        "qa_memoirist_v4_run3.jsonl",
    ]:
        for p in sample(name):
            key = judge_key(source, p["question"], p["answer"])
            assert key in judge._cache, f"{name}: cache miss for {p['question'][:20]}…"


@pytest.mark.parametrize(
    "name",
    ["qa_memoirist_v4_run1.jsonl", "qa_memoirist_v4_run2.jsonl", "qa_memoirist_v4_run3.jsonl"],
)
def test_golden_kumjung_coherent_fail(name, source, rubric, judge):
    """§6: the "검정이 되었던 상황" pair -> COHERENT FAIL (register-only can't catch it)."""
    pairs = sample(name)
    scores = score_pairs(pairs, source, rubric, judge=judge, pairs_per_chunk=5)
    kumjung = scores[KUMJUNG_INDEX]
    assert "검정이 되었던" in kumjung.answer
    # Register is PLAIN (passes) — proving a non-register dimension is required...
    assert kumjung.dims["register"].passed
    # ...and the judge flags it as incoherent (the spec's headline calibration).
    assert not kumjung.dims["coherent"].passed, kumjung.dims["coherent"].reason
    assert not kumjung.strict_pass


def test_golden_v4_run2_clean_pairs_strict_pass(source, rubric, judge):
    """§6: plain·grounded pairs in run2 -> strict PASS."""
    pairs = sample("qa_memoirist_v4_run2.jsonl")
    scores = score_pairs(pairs, source, rubric, judge=judge, pairs_per_chunk=5)
    # Indices 0/2/3/7 are plain, first-person, grounded, coherent.
    for i in (0, 2, 3, 7):
        assert scores[i].strict_pass, f"run2[{i}] should be strict PASS: {scores[i].failed_dims()}"


def test_golden_v4_run3_clean_pairs_strict_pass(source, rubric, judge):
    pairs = sample("qa_memoirist_v4_run3.jsonl")
    scores = score_pairs(pairs, source, rubric, judge=judge, pairs_per_chunk=5)
    for i in (0, 1, 2, 7):
        assert scores[i].strict_pass, f"run3[{i}] should be strict PASS: {scores[i].failed_dims()}"


# ===========================================================================
# §6 aggregate calibration target
# ===========================================================================
def test_v4_aggregate_strict_matches_spec_target(source, rubric, judge):
    """§6 target: v4 strict ≈ 20~22/30 (존댓말 5 + 검정왜곡 3 감안)."""
    strict = register_fail = total = 0
    for name in [
        "qa_memoirist_v4_run1.jsonl",
        "qa_memoirist_v4_run2.jsonl",
        "qa_memoirist_v4_run3.jsonl",
    ]:
        scores = score_pairs(sample(name), source, rubric, judge=judge, pairs_per_chunk=5)
        agg = aggregate(scores)
        strict += agg["strict_pass"]
        register_fail += agg["dim_fail"]["register"]
        total += agg["total"]
    assert total == 30
    assert 20 <= strict <= 22, f"v4 strict {strict}/30 outside spec target 20-22"
    assert register_fail == 5, f"register fails {register_fail} != manual 5"


def test_v2_strict_is_zero_due_to_register_gate(source, rubric):
    """§6: v2 ≈ 0/10. Even if a judge passed everything, REGISTER gates it to 0."""
    # StubJudge that would pass every judge dimension — isolates the register gate.
    all_pass = StubJudge(
        lambda s, q, a: {d: True for d in ("grounded", "coherent", "voice_preserved", "q_grounded")}
    )
    scores = score_pairs(sample("qa_memoirist_v2_enlarged.jsonl"), source, rubric, judge=all_pass)
    assert sum(1 for s in scores if s.strict_pass) == 0


# ===========================================================================
# PASS logic + aggregation (StubJudge — no LLM)
# ===========================================================================
def test_strict_and_lenient_logic_with_stub(rubric):
    good = [{"question": "아홉 살에 서당에 간 날은 어떠하였는가?", "answer": "나는 아홉 살에 서당에 나갔다.", "raw": {}}]
    all_pass = StubJudge(lambda s, q, a: {"grounded": True, "coherent": True, "voice_preserved": True, "q_grounded": True})
    s = score_pairs(good, "서당에 나갔다.", rubric, judge=all_pass)[0]
    assert s.strict_pass and s.lenient_pass

    # Flip grounded -> both strict and lenient fail (grounded is in both).
    fab = StubJudge(lambda s, q, a: {"grounded": False, "coherent": True, "voice_preserved": True, "q_grounded": True})
    s2 = score_pairs(good, "서당에 나갔다.", rubric, judge=fab)[0]
    assert not s2.strict_pass and not s2.lenient_pass
    assert "grounded" in s2.failed_dims()

    # Flip only voice_preserved -> strict fails, lenient still passes.
    voice = StubJudge(lambda s, q, a: {"grounded": True, "coherent": True, "voice_preserved": False, "q_grounded": True})
    s3 = score_pairs(good, "서당에 나갔다.", rubric, judge=voice)[0]
    assert not s3.strict_pass and s3.lenient_pass


def test_no_judge_leaves_layer2_unchecked(rubric):
    pairs = sample("qa_memoirist_v4_run2.jsonl")
    scores = score_pairs(pairs, "src", rubric, judge=None)
    # Layer-2 dims are unchecked -> not "judged", and strict cannot be claimed.
    assert not scores[0].judged
    assert not scores[0].strict_pass
    # Layer-1 register still runs.
    assert scores[0].dims["register"].checked


def test_aggregate_shapes(source, rubric, judge):
    scores = score_pairs(sample("qa_memoirist_v4_run1.jsonl"), source, rubric, judge=judge, pairs_per_chunk=5)
    agg = aggregate(scores)
    assert agg["total"] == 10 and agg["judged"] == 10
    assert 0 <= agg["strict_pass"] <= 10
    assert set(("format", "register", "grounded", "coherent")).issubset(agg["dim_fail"])
