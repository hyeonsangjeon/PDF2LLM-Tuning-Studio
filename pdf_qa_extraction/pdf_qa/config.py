"""Cloud-agnostic runtime configuration for the PDF -> Q&A pipeline.

The same :class:`QAConfig` is consumed by every entrypoint (local Docker,
Azure ML Job, SageMaker Processing Job) so behaviour stays identical across
clouds. Values can come from environment variables or CLI arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _clean_optional(value: Optional[str]) -> Optional[str]:
    """Normalise ``"None"`` / empty strings coming from env or CLI to ``None``."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "none":
        return None
    return value


def _as_bool(value, default: bool = True) -> bool:
    """Parse a truthy/falsy env or CLI value; ``None`` -> ``default``."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value, default: int) -> int:
    """Parse an int from env/CLI; blank/invalid/``None`` -> ``default``."""
    if value is None:
        return default
    try:
        text = str(value).strip()
        return int(text) if text else default
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float) -> float:
    """Parse a float from env/CLI; blank/invalid/``None`` -> ``default``."""
    if value is None:
        return default
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


@dataclass
class QAConfig:
    """Parameters that drive Q&A generation, independent of the LLM provider."""

    domain: str = "International Finance"
    num_questions: str = "5"
    num_img_questions: str = "1"
    # Output-language lock for the generated Q&A. ``auto`` (default) makes the
    # model match the source document's language (fixes English-source ->
    # Korean-answer drift); a name/code (``korean``, ``en`` ...) forces it.
    language: str = "auto"
    # Layout-detection model for the ``hi_res`` path (``yolox`` default,
    # ``detectron2_onnx``, ``yolox_quantized`` ...). Passed to unstructured as
    # ``hi_res_model_name``; env ``TABLE_MODEL`` (legacy) or ``HI_RES_MODEL_NAME``.
    table_model: Optional[str] = None
    figures_dir: str = "figures"
    # Q&A persona/style (see ``pdf_qa.prompts.PERSONAS`` / ``personas.yaml``):
    # professor (default), socratic, consultant, interviewer, analyst, feynman,
    # memoirist. The ledger is editable and overridable via ``PERSONA_FILE``.
    persona: str = "professor"    # ``unstructured`` strategy: auto | fast | hi_res | ocr_only. ``auto`` is
    # escalated to ``hi_res`` when a GPU is detected and ``gpu_boost`` is on.
    strategy: str = "auto"
    # Route the heavy layout + table models to the GPU when one is reachable.
    gpu_boost: bool = True
    # Provider-specific model / deployment id. When ``None`` each provider
    # falls back to its own sensible default (see ``providers/``).
    model_id: Optional[str] = None
    # ---- generated-pair quality control (see ``pdf_qa.validate``) ----------
    # Drop empty/too-short/refusal pairs before the dataset is written.
    validate_qa: bool = True
    # Remove exact + near-duplicate questions.
    dedup_qa: bool = True
    # Minimum character length for a kept question / answer.
    min_question_chars: int = 6
    min_answer_chars: int = 4
    # Jaccard similarity (0-1) above which two questions count as near-dupes;
    # 1.0 keeps only exact-duplicate removal.
    dedup_similarity: float = 0.9
    # Drop answers that are model refusals / "not in the context".
    drop_refusals: bool = True

    @classmethod
    def from_env(cls) -> "QAConfig":
        """Build a config from environment variables (local / container runs)."""
        return cls(
            domain=os.environ.get("DOMAIN", "International Finance"),
            num_questions=os.environ.get("NUM_QUESTIONS", "5"),
            num_img_questions=os.environ.get("NUM_IMG_QUESTIONS", "1"),
            language=(
                os.environ.get("OUTPUT_LANGUAGE")
                or os.environ.get("QA_LANGUAGE")
                or os.environ.get("LANGUAGE")
                or "auto"
            ),
            table_model=_clean_optional(
                os.environ.get("TABLE_MODEL")
                or os.environ.get("HI_RES_MODEL_NAME")
                or os.environ.get("UNSTRUCTURED_HI_RES_MODEL_NAME")
            ),
            figures_dir=os.environ.get("FIGURES_DIR", "figures"),
            persona=os.environ.get("PERSONA", "professor"),
            strategy=os.environ.get("STRATEGY", "auto"),
            gpu_boost=_as_bool(os.environ.get("GPU_BOOST"), default=True),
            model_id=_clean_optional(os.environ.get("MODEL_ID")),
            validate_qa=_as_bool(os.environ.get("VALIDATE_QA"), default=True),
            dedup_qa=_as_bool(os.environ.get("DEDUP_QA"), default=True),
            min_question_chars=_as_int(
                os.environ.get("MIN_QUESTION_CHARS"), default=6
            ),
            min_answer_chars=_as_int(os.environ.get("MIN_ANSWER_CHARS"), default=4),
            dedup_similarity=_as_float(
                os.environ.get("DEDUP_SIMILARITY"), default=0.9
            ),
            drop_refusals=_as_bool(os.environ.get("DROP_REFUSALS"), default=True),
        )

    @classmethod
    def from_args(cls, args) -> "QAConfig":
        """Build a config from an ``argparse.Namespace`` (managed jobs)."""
        return cls(
            domain=getattr(args, "domain", "International Finance"),
            num_questions=getattr(args, "num_questions", "5"),
            num_img_questions=getattr(args, "num_img_questions", "1"),
            language=getattr(args, "language", None) or "auto",
            table_model=_clean_optional(getattr(args, "table_model", None)),
            figures_dir=getattr(args, "figures_dir", "figures"),
            persona=getattr(args, "persona", "professor") or "professor",
            strategy=getattr(args, "strategy", "auto") or "auto",
            gpu_boost=_as_bool(getattr(args, "gpu_boost", None), default=True),
            model_id=_clean_optional(getattr(args, "model_id", None)),
            validate_qa=_as_bool(getattr(args, "validate_qa", None), default=True),
            dedup_qa=_as_bool(getattr(args, "dedup_qa", None), default=True),
            min_question_chars=_as_int(
                getattr(args, "min_question_chars", None), default=6
            ),
            min_answer_chars=_as_int(
                getattr(args, "min_answer_chars", None), default=4
            ),
            dedup_similarity=_as_float(
                getattr(args, "dedup_similarity", None), default=0.9
            ),
            drop_refusals=_as_bool(
                getattr(args, "drop_refusals", None), default=True
            ),
        )
