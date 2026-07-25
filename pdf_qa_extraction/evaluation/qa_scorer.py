"""Two-layer QA evaluation scorer for the memoirist fine-tuning dataset.

Turns the *manual* "문체 보존 / 날조 / 존댓말 / 왜곡" judgement into a repeatable,
auditable scorer. Two uses: a dataset QC gate (drop bad pairs before training)
and a regression eval (objectively compare persona versions / extractor models).

Two layers (see ``rubric.yaml`` — every rule lives there, nothing is hardcoded):

* **Layer 1 — deterministic** (no LLM, instant, every pair): ``REGISTER``
  (존댓말 검출), ``FIRST_PERSON``, ``LEADING_Q``, ``FORMAT``.
* **Layer 2 — LLM judge** (one temperature-0 call per pair, *independent* of the
  generator): ``GROUNDED``, ``COHERENT``, ``VOICE_PRESERVED``, ``Q_GROUNDED``.

This module never imports the heavy PDF stack at module load and never touches
``pdf_qa`` state; it only *reads* two pure helpers (``custom_json_parser`` for
robust judge-JSON parsing and, lazily, ``extract_document_layout`` for a PDF
source), so the ``pdf_qa`` core / ``personas.yaml`` / web app stay untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import yaml

DEFAULT_RUBRIC_FILE = os.path.join(os.path.dirname(__file__), "rubric.yaml")

# ---------------------------------------------------------------------------
# Schema normalisation + JSONL IO
# ---------------------------------------------------------------------------
#: Accepted question/answer key spellings, in priority order.
_QUESTION_KEYS = ("QUESTION", "question", "instruction", "Instruction", "prompt")
_ANSWER_KEYS = ("ANSWER", "answer", "output", "Output", "response", "completion")


def normalize_pair(obj: dict) -> dict:
    """Normalise a raw record to ``{question, answer, raw}``.

    Accepts both the ``QUESTION``/``ANSWER`` and ``instruction``/``output``
    schemas (and a few common aliases) so the scorer is schema-agnostic.
    """
    question = next((obj[k] for k in _QUESTION_KEYS if obj.get(k) is not None), "")
    answer = next((obj[k] for k in _ANSWER_KEYS if obj.get(k) is not None), "")
    return {
        "question": str(question or "").strip(),
        "answer": str(answer or "").strip(),
        "raw": obj,
    }


def load_jsonl(path: str) -> List[dict]:
    """Read a UTF-8 JSONL file into a list of dicts (blank lines skipped)."""
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_pairs(path: str) -> List[dict]:
    """Load a JSONL dataset and normalise every record."""
    return [normalize_pair(o) for o in load_jsonl(path)]


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rubric:
    """The detection rules, loaded from ``rubric.yaml`` (nothing hardcoded)."""

    honorific_suffixes: Tuple[str, ...]
    formal_polite_suffixes: Tuple[str, ...]
    plain_examples: Tuple[str, ...]
    first_person_markers: Tuple[str, ...]
    third_person_markers: Tuple[str, ...]
    leading_q_patterns: Tuple[str, ...]
    min_question_chars: int
    min_answer_chars: int
    drop_exact_duplicates: bool
    judge_dimensions: Tuple[str, ...]
    judge_prompt: str


def load_rubric(path: Optional[str] = None) -> Rubric:
    """Read and validate the rubric ledger."""
    ledger_path = path or os.environ.get("QA_RUBRIC_FILE") or DEFAULT_RUBRIC_FILE
    with open(ledger_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    reg = data.get("register", {}) or {}
    fp = data.get("first_person", {}) or {}
    lq = data.get("leading_q", {}) or {}
    fmt = data.get("format", {}) or {}
    judge = data.get("judge", {}) or {}

    prompt = str(judge.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"Rubric '{ledger_path}' is missing judge.prompt.")

    def _tuple(seq) -> Tuple[str, ...]:
        return tuple(str(x) for x in (seq or []))

    return Rubric(
        honorific_suffixes=_tuple(reg.get("honorific_suffixes")),
        formal_polite_suffixes=_tuple(reg.get("formal_polite_suffixes")),
        plain_examples=_tuple(reg.get("plain_examples")),
        first_person_markers=_tuple(fp.get("markers")),
        third_person_markers=_tuple(fp.get("third_person_markers")),
        leading_q_patterns=_tuple(lq.get("patterns")),
        min_question_chars=int(fmt.get("min_question_chars", 6)),
        min_answer_chars=int(fmt.get("min_answer_chars", 4)),
        drop_exact_duplicates=bool(fmt.get("drop_exact_duplicates", True)),
        judge_dimensions=_tuple(judge.get("dimensions")),
        judge_prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Layer 1 — deterministic helpers
# ---------------------------------------------------------------------------
_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
# Jongseong (final-consonant) index of ㅂ and ㅄ in the Unicode Hangul algorithm.
_JONG_BIEUP = {17, 18}
# Paired quote spans stripped before register analysis so a *quoted* polite line
# ("할아버지 왜 그래요?") never makes the surrounding plain narration look honorific.
_QUOTE_SPAN = re.compile(r"[\"'“”‘’]([^\"'“”‘’]*)[\"'“”‘’]|「[^」]*」|『[^』]*』")
_SENTENCE_SPLIT = re.compile(r"[.!?…\n]+")
_TRAILING_HANGUL = re.compile(r"[가-힣]+$")


def _has_final_bieup(syllable: str) -> bool:
    """True if a single Hangul ``syllable`` carries a ㅂ-batchim (or is 습)."""
    if not syllable:
        return False
    code = ord(syllable[-1])
    if not (_HANGUL_BASE <= code <= _HANGUL_LAST):
        return False
    return (code - _HANGUL_BASE) % 28 in _JONG_BIEUP


def _final_words(answer: str) -> List[str]:
    """Return the trailing Hangul word of every sentence in ``answer``.

    Quoted spans are removed first, then the text is split into sentences and the
    last Hangul run of each is taken — that run carries the sentence-final ending
    used for register detection.
    """
    stripped = _QUOTE_SPAN.sub(" ", answer)
    words: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(stripped):
        sentence = sentence.strip()
        if not sentence:
            continue
        match = _TRAILING_HANGUL.search(sentence)
        if match:
            words.append(match.group(0))
    return words


def _is_honorific_word(word: str, rubric: Rubric) -> bool:
    """True if a sentence-final ``word`` is in the polite (존댓말) register."""
    for suffix in rubric.honorific_suffixes:
        if word.endswith(suffix):
            return True
    for suffix in rubric.formal_polite_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix):
            # -ㅂ니다/습니다 only: the syllable before the suffix must carry ㅂ,
            # so 입니다/합니다/생생합니다 fail but 아니다/지니다 pass.
            if _has_final_bieup(word[: -len(suffix)]):
                return True
    return False


# ---------------------------------------------------------------------------
# Dimension result + per-pair score
# ---------------------------------------------------------------------------
@dataclass
class DimResult:
    """One dimension's verdict: ``passed`` + a short human-readable reason."""

    passed: bool
    reason: str = ""
    warn: bool = False
    checked: bool = True  # False when the layer was skipped (e.g. no judge)


def check_register(answer: str, rubric: Rubric) -> DimResult:
    """REGISTER — FAIL if any answer sentence ends in the polite register."""
    offenders = [w for w in _final_words(answer) if _is_honorific_word(w, rubric)]
    if offenders:
        uniq = ", ".join(dict.fromkeys(offenders))
        return DimResult(False, f"존댓말 종결: {uniq}")
    return DimResult(True, "plain 문어체 종결 유지")


def check_first_person(answer: str, rubric: Rubric) -> DimResult:
    """FIRST_PERSON — 1인칭 화자 시점인가. 3인칭이 명확하면 FAIL, 미검출은 WARN."""
    has_first = any(m in answer for m in rubric.first_person_markers)
    has_third = any(m in answer for m in rubric.third_person_markers)
    if has_first:
        return DimResult(True, "1인칭 화자")
    if has_third:
        return DimResult(False, "3인칭 서술(1인칭 표지 없음)")
    # No explicit marker either way — imperatives / maxims are fine; defer.
    return DimResult(True, "1인칭 표지 없음(경고, judge 확인)", warn=True)


def check_leading_q(question: str, rubric: Rubric) -> DimResult:
    """LEADING_Q — flag ungrounded synthesis/moral prompts (warning only)."""
    for pattern in rubric.leading_q_patterns:
        if re.search(pattern, question):
            return DimResult(True, f"유도질문 패턴: /{pattern}/", warn=True)
    return DimResult(True, "유도질문 아님")


def check_format(
    question: str, answer: str, rubric: Rubric, seen: set
) -> DimResult:
    """FORMAT — empty / too-short / duplicate / schema violation => FAIL."""
    if not question and not answer:
        return DimResult(False, "빈 질문+빈 답변(스키마 위반)")
    if not question:
        return DimResult(False, "빈 질문")
    if not answer:
        return DimResult(False, "빈 답변")
    if len(question) < rubric.min_question_chars:
        return DimResult(False, f"질문 {len(question)}자 < {rubric.min_question_chars}")
    if len(answer) < rubric.min_answer_chars:
        return DimResult(False, f"답변 {len(answer)}자 < {rubric.min_answer_chars}")
    if rubric.drop_exact_duplicates:
        key = (question, answer)
        if key in seen:
            return DimResult(False, "중복 쌍")
        seen.add(key)
    return DimResult(True, "형식 정상")


# Deterministic (Layer 1) + judge (Layer 2) dimension names.
_LAYER1_DIMS = ("format", "register", "first_person", "leading_q")
_LAYER2_DIMS = ("grounded", "coherent", "voice_preserved", "q_grounded")
# Dimensions that gate strict / lenient PASS (LEADING_Q is a warning only).
_STRICT_DIMS = (
    "format",
    "register",
    "first_person",
    "grounded",
    "coherent",
    "voice_preserved",
    "q_grounded",
)
_LENIENT_DIMS = ("format", "grounded", "coherent", "register")


@dataclass
class PairScore:
    """The full verdict for one Q&A pair."""

    index: int
    question: str
    answer: str
    dims: Dict[str, DimResult]
    raw: dict = field(default_factory=dict)
    chunk: Optional[int] = None

    @property
    def judged(self) -> bool:
        """True when the Layer-2 judge actually ran for this pair."""
        return all(self.dims.get(d, DimResult(True)).checked for d in _LAYER2_DIMS)

    def _all_pass(self, names: Sequence[str]) -> bool:
        for name in names:
            dim = self.dims.get(name)
            if dim is None or not dim.checked or not dim.passed:
                return False
        return True

    @property
    def strict_pass(self) -> bool:
        return self._all_pass(_STRICT_DIMS)

    @property
    def lenient_pass(self) -> bool:
        return self._all_pass(_LENIENT_DIMS)

    def failed_dims(self, names: Sequence[str] = _STRICT_DIMS) -> List[str]:
        out = []
        for name in names:
            dim = self.dims.get(name)
            if dim is not None and dim.checked and not dim.passed:
                out.append(name)
        return out

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "chunk": self.chunk,
            "question": self.question,
            "answer": self.answer,
            "strict_pass": self.strict_pass,
            "lenient_pass": self.lenient_pass,
            "judged": self.judged,
            "dimensions": {
                name: {
                    "passed": d.passed,
                    "warn": d.warn,
                    "checked": d.checked,
                    "reason": d.reason,
                }
                for name, d in self.dims.items()
            },
        }


# ---------------------------------------------------------------------------
# Layer 2 — the judge abstraction
# ---------------------------------------------------------------------------
def judge_key(source: str, question: str, answer: str) -> str:
    """Stable content hash used to cache / replay judge verdicts."""
    digest = hashlib.sha1()
    for part in (source, question, answer):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class Judge:
    """Interface: turn ``(source, question, answer)`` into dimension verdicts."""

    def judge(self, source: str, question: str, answer: str) -> Dict[str, DimResult]:
        raise NotImplementedError


class StubJudge(Judge):
    """Wrap a plain ``fn(source, question, answer) -> dict`` for tests.

    The function returns a dict of ``{dim: bool}`` or ``{dim: (bool, reason)}``.
    """

    def __init__(self, fn: Callable[[str, str, str], Dict[str, object]]):
        self._fn = fn

    def judge(self, source: str, question: str, answer: str) -> Dict[str, DimResult]:
        raw = self._fn(source, question, answer)
        return _coerce_judge_dims(raw)


class ReplayJudge(Judge):
    """Replay recorded judge verdicts from a cache file (hermetic CI tests).

    The cache maps ``judge_key`` -> the raw judge JSON. Verdicts came from a real
    LLM judge run, so the meta-eval calibrates against genuine judge output with
    no credentials or network at test time.
    """

    def __init__(self, cache: Dict[str, dict]):
        self._cache = cache

    @classmethod
    def from_file(cls, path: str) -> "ReplayJudge":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def judge(self, source: str, question: str, answer: str) -> Dict[str, DimResult]:
        key = judge_key(source, question, answer)
        if key not in self._cache:
            raise KeyError(
                f"ReplayJudge cache miss for key {key[:12]}… "
                "(record it with run_eval --record-judge)."
            )
        return _coerce_judge_dims(self._cache[key])


class RecordingJudge(Judge):
    """Wrap any judge and record its verdicts so they can be replayed later.

    Used once (with a real :class:`LLMJudge`) to build the hermetic meta-eval
    cache: every verdict is stored keyed by :func:`judge_key`, then written with
    :meth:`dump` for :class:`ReplayJudge` to consume in CI.
    """

    def __init__(self, inner: Judge):
        self.inner = inner
        self.records: Dict[str, dict] = {}

    def judge(self, source: str, question: str, answer: str) -> Dict[str, DimResult]:
        dims = self.inner.judge(source, question, answer)
        record: Dict[str, object] = {}
        for name in _LAYER2_DIMS:
            if name in dims:
                record[name] = dims[name].passed
                record[f"{name}_reason"] = dims[name].reason
        self.records[judge_key(source, question, answer)] = record
        return dims

    def dump(self, path: str) -> None:
        _ensure_dir(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.records, handle, ensure_ascii=False, indent=2)


def _coerce_judge_dims(raw: Dict[str, object]) -> Dict[str, DimResult]:
    """Turn a raw judge dict into ``{dim: DimResult}`` for the four dimensions."""
    dims: Dict[str, DimResult] = {}
    for name in _LAYER2_DIMS:
        value = raw.get(name)
        reason = str(raw.get(f"{name}_reason", "") or "")
        if isinstance(value, tuple):
            passed, reason = bool(value[0]), str(value[1])
        elif isinstance(value, dict):
            passed = bool(value.get("passed", value.get("value")))
            reason = str(value.get("reason", reason))
        elif value is None:
            dims[name] = DimResult(True, "judge 미판정", checked=False)
            continue
        else:
            passed = _as_bool(value)
        dims[name] = DimResult(passed, reason)
    return dims


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t", "참", "예"}


class LLMJudge(Judge):
    """Real LLM judge — an *independent*, temperature-0 structured-JSON call.

    Built to be different from the generator (spec: judge ≠ generator model).
    Supports the Azure OpenAI (Entra ID keyless or api-key) and OpenAI routes.
    The heavy client SDK is imported lazily so importing this module never
    requires ``langchain_openai`` to be installed.
    """

    def __init__(
        self,
        prompt_template: str,
        provider: str = "azure",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 600,
        api_version: Optional[str] = None,
    ):
        self.prompt_template = prompt_template
        self.provider = (provider or "azure").strip().lower()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._llm = self._build_client(model, api_version)

    def _build_client(self, model: Optional[str], api_version: Optional[str]):
        if self.provider in {"azure", "foundry", "azure_openai", "azure-openai"}:
            from langchain_openai import AzureChatOpenAI

            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is required for the azure judge.")
            deployment = (
                model or os.getenv("JUDGE_MODEL") or os.getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT")
                or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4o"
            )
            self.model = deployment
            version = (
                api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
            )
            common = dict(
                azure_deployment=deployment,
                azure_endpoint=endpoint,
                api_version=version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if api_key:
                return AzureChatOpenAI(api_key=api_key, **common)
            # Keyless: reuse pdf_qa's shared Entra ID credential (read-only import).
            from azure.identity import get_bearer_token_provider

            from pdf_qa.providers.azure_foundry import azure_credential, _token_scope

            token_provider = get_bearer_token_provider(azure_credential(), _token_scope())
            return AzureChatOpenAI(azure_ad_token_provider=token_provider, **common)

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            self.model = model or os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
            return ChatOpenAI(
                model=self.model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

        raise ValueError(
            f"Unsupported judge provider '{self.provider}'. Use azure | openai."
        )

    def raw_judge(self, source: str, question: str, answer: str) -> dict:
        """Invoke the judge and return the parsed raw JSON dict."""
        from pdf_qa.parsing import custom_json_parser

        prompt = (
            self.prompt_template.replace("{source}", source)
            .replace("{question}", question)
            .replace("{answer}", answer)
        )
        response = self._llm.invoke(prompt)
        parsed = custom_json_parser(response)
        return parsed[0] if parsed else {}

    def judge(self, source: str, question: str, answer: str) -> Dict[str, DimResult]:
        return _coerce_judge_dims(self.raw_judge(source, question, answer))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_pairs(
    pairs: Sequence[dict],
    source: str,
    rubric: Rubric,
    judge: Optional[Judge] = None,
    pairs_per_chunk: Optional[int] = None,
) -> List[PairScore]:
    """Score every normalised pair through Layer 1 (+ Layer 2 when ``judge``).

    ``pairs_per_chunk`` (e.g. NUM_QUESTIONS) enables run×chunk aggregation by
    tagging each pair with a chunk index inferred from its position.
    """
    seen: set = set()
    scores: List[PairScore] = []
    for i, pair in enumerate(pairs):
        question = pair["question"]
        answer = pair["answer"]
        dims: Dict[str, DimResult] = {}
        fmt = check_format(question, answer, rubric, seen)
        dims["format"] = fmt
        dims["register"] = check_register(answer, rubric)
        dims["first_person"] = check_first_person(answer, rubric)
        dims["leading_q"] = check_leading_q(question, rubric)

        if fmt.passed and judge is not None:
            dims.update(judge.judge(source, question, answer))
        else:
            for name in _LAYER2_DIMS:
                reason = "형식 불량으로 미판정" if not fmt.passed else "judge 미실행"
                dims[name] = DimResult(True, reason, checked=False)

        chunk = (i // pairs_per_chunk) if pairs_per_chunk else None
        scores.append(
            PairScore(
                index=i,
                question=question,
                answer=answer,
                dims=dims,
                raw=pair.get("raw", {}),
                chunk=chunk,
            )
        )
    return scores


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
_REPORT_DIMS = _STRICT_DIMS + ("leading_q",)


def aggregate(scores: Sequence[PairScore]) -> dict:
    """Roll a list of :class:`PairScore` into counts / rates for the report."""
    total = len(scores)
    judged = [s for s in scores if s.judged]
    strict = sum(1 for s in scores if s.strict_pass)
    lenient = sum(1 for s in scores if s.lenient_pass)
    dim_fail: Dict[str, int] = {}
    dim_warn: Dict[str, int] = {}
    for name in _REPORT_DIMS:
        dim_fail[name] = sum(
            1 for s in scores
            if s.dims.get(name) and s.dims[name].checked and not s.dims[name].passed
        )
        dim_warn[name] = sum(
            1 for s in scores if s.dims.get(name) and s.dims[name].warn
        )
    return {
        "total": total,
        "judged": len(judged),
        "strict_pass": strict,
        "lenient_pass": lenient,
        "strict_rate": (strict / total) if total else 0.0,
        "lenient_rate": (lenient / total) if total else 0.0,
        "dim_fail": dim_fail,
        "dim_warn": dim_warn,
    }


def aggregate_by_chunk(scores: Sequence[PairScore]) -> Dict[int, dict]:
    """Per-chunk aggregates (register lock is chunk-clustered, so expose it)."""
    chunks: Dict[int, List[PairScore]] = {}
    for s in scores:
        if s.chunk is None:
            continue
        chunks.setdefault(s.chunk, []).append(s)
    return {c: aggregate(v) for c, v in sorted(chunks.items())}


def summarize_runs(run_aggs: Sequence[dict], key: str) -> dict:
    """min / max / mean across runs for a given aggregate ``key``."""
    values = [a[key] for a in run_aggs]
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def write_scored(scores: Sequence[PairScore], path: str) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        for s in scores:
            handle.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def write_clean_and_rejected(
    scores: Sequence[PairScore], clean_path: str, rejected_path: str
) -> Tuple[int, int]:
    """Split into strict-PASS (clean, training-ready) and rejected (with reasons)."""
    _ensure_dir(clean_path)
    _ensure_dir(rejected_path)
    kept = dropped = 0
    with open(clean_path, "w", encoding="utf-8") as clean, open(
        rejected_path, "w", encoding="utf-8"
    ) as rejected:
        for s in scores:
            if s.strict_pass:
                clean.write(json.dumps(s.raw, ensure_ascii=False) + "\n")
                kept += 1
            else:
                reasons = {
                    name: s.dims[name].reason
                    for name in s.failed_dims()
                    if name in s.dims
                }
                record = dict(s.raw)
                record["_reject_reasons"] = reasons
                record["_failed_dims"] = s.failed_dims()
                rejected.write(json.dumps(record, ensure_ascii=False) + "\n")
                dropped += 1
    return kept, dropped


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
