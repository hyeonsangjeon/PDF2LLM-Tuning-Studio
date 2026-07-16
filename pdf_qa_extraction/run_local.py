#!/usr/bin/env python3
"""Unified local / Docker entrypoint for PDF -> Q&A extraction.

The LLM backend is selected with the ``LLM_PROVIDER`` environment variable
(``azure`` by default, also ``bedrock`` or ``openai``). All other parameters
come from the environment (see ``QAConfig.from_env``). Example::

    LLM_PROVIDER=azure PDF_PATH=data/fsi_data.pdf python run_local.py
"""

from __future__ import annotations

import os
import sys

# Load a local .env when present (optional dependency).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Import the core package whether it is pip-installed (container) or sitting
# next to this script (repo checkout).
try:
    import pdf_qa  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_qa import QAConfig, get_provider, run_pipeline


def main() -> None:
    provider_name = os.getenv("LLM_PROVIDER", "azure")
    pdf_path = os.getenv("PDF_PATH", "data/fsi_data.pdf")
    output_path = os.getenv("OUTPUT_PATH", "data/qa_pairs.jsonl")

    config = QAConfig.from_env()
    provider = get_provider(provider_name, config=config)
    run_pipeline(pdf_path, output_path, provider, config)


if __name__ == "__main__":
    main()
