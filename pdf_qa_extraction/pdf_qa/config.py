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


@dataclass
class QAConfig:
    """Parameters that drive Q&A generation, independent of the LLM provider."""

    domain: str = "International Finance"
    num_questions: str = "5"
    num_img_questions: str = "1"
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

    @classmethod
    def from_env(cls) -> "QAConfig":
        """Build a config from environment variables (local / container runs)."""
        return cls(
            domain=os.environ.get("DOMAIN", "International Finance"),
            num_questions=os.environ.get("NUM_QUESTIONS", "5"),
            num_img_questions=os.environ.get("NUM_IMG_QUESTIONS", "1"),
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
        )

    @classmethod
    def from_args(cls, args) -> "QAConfig":
        """Build a config from an ``argparse.Namespace`` (managed jobs)."""
        return cls(
            domain=getattr(args, "domain", "International Finance"),
            num_questions=getattr(args, "num_questions", "5"),
            num_img_questions=getattr(args, "num_img_questions", "1"),
            table_model=_clean_optional(getattr(args, "table_model", None)),
            figures_dir=getattr(args, "figures_dir", "figures"),
            persona=getattr(args, "persona", "professor") or "professor",
            strategy=getattr(args, "strategy", "auto") or "auto",
            gpu_boost=_as_bool(getattr(args, "gpu_boost", None), default=True),
            model_id=_clean_optional(getattr(args, "model_id", None)),
        )
