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


@dataclass
class QAConfig:
    """Parameters that drive Q&A generation, independent of the LLM provider."""

    domain: str = "International Finance"
    num_questions: str = "5"
    num_img_questions: str = "1"
    table_model: Optional[str] = None
    figures_dir: str = "figures"
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
            table_model=_clean_optional(os.environ.get("TABLE_MODEL")),
            figures_dir=os.environ.get("FIGURES_DIR", "figures"),
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
            model_id=_clean_optional(getattr(args, "model_id", None)),
        )
