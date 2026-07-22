"""The one-call convenience API.

Everything the pipeline needs is already wired through :class:`QAConfig`,
:func:`get_provider` and :func:`run_pipeline`; :func:`extract_qa` is the tiny
facade that ties them together so the common case is a single line::

    from pdf_qa import extract_qa

    pairs = extract_qa("report.pdf")                       # env-configured
    extract_qa("report.pdf", out="qa.jsonl", persona="feynman", provider="ollama")

Environment variables (see ``settings.yaml`` / ``.env``) are the baseline; any
keyword you pass overrides them. GPU/CPU is auto-detected, the persona ledger
and the chart<->context linkage all apply automatically.
"""

from __future__ import annotations

from typing import List, Optional

from .config import QAConfig
from .pipeline import generate_qa_pairs, run_pipeline
from .providers import get_provider
from .providers.base import LLMProvider
from .validate import clean_qa_pairs


def extract_qa(
    pdf: str,
    *,
    out: Optional[str] = None,
    provider: Optional[str] = None,
    persona: Optional[str] = None,
    domain: Optional[str] = None,
    language: Optional[str] = None,
    num_questions: Optional[str] = None,
    num_img_questions: Optional[str] = None,
    strategy: Optional[str] = None,
    gpu_boost: Optional[bool] = None,
    model_id: Optional[str] = None,
    table_model: Optional[str] = None,
    figures_dir: Optional[str] = None,
    provider_obj: Optional[LLMProvider] = None,
) -> List[dict]:
    """Run the full PDF -> Q&A pipeline in one call and return the pairs.

    Args:
        pdf: Path to the input PDF.
        out: If given, also write the pairs to this JSONL path.
        provider: Backend name (``azure`` | ``bedrock`` | ``openai`` | ``ollama``
            and aliases). ``None`` uses the ``LLM_PROVIDER`` env var, then
            ``azure``. Ignored when ``provider_obj`` is supplied.
        language: Output-language lock (``auto`` matches the source document;
            ``korean``/``en``/... forces it).
        persona, domain, num_questions, num_img_questions, strategy, gpu_boost,
        model_id, table_model, figures_dir: Optional overrides; anything left as
            ``None`` falls back to the environment (:meth:`QAConfig.from_env`).
        provider_obj: A pre-built :class:`LLMProvider` (skips ``get_provider``);
            handy for tests or reusing one client across many PDFs.

    Returns:
        The list of ``{"QUESTION": ..., "ANSWER": ..., ...}`` dicts (image-derived
        pairs also carry ``source``/``page``/``section``/``figure_index``).
    """
    config = QAConfig.from_env()
    overrides = {
        "persona": persona,
        "domain": domain,
        "num_questions": num_questions,
        "num_img_questions": num_img_questions,
        "strategy": strategy,
        "gpu_boost": gpu_boost,
        "model_id": model_id,
        "table_model": table_model,
        "figures_dir": figures_dir,
        "language": language,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    llm = provider_obj or get_provider(provider, config=config)

    pdf = str(pdf)
    if out:
        return run_pipeline(pdf, str(out), llm, config)
    # No output file: still return curated pairs (validate + de-dup) so the
    # one-call API is consistent with the saved-dataset path.
    raw = generate_qa_pairs(pdf, llm, config)
    cleaned, _ = clean_qa_pairs(raw, config)
    return cleaned
