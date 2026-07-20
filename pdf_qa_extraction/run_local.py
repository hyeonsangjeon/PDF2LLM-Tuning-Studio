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

from pdf_qa import extract_qa


def main() -> None:
    extract_qa(
        os.getenv("PDF_PATH", "data/fsi_data.pdf"),
        out=os.getenv("OUTPUT_PATH", "data/qa_pairs.jsonl"),
        provider=os.getenv("LLM_PROVIDER"),
    )


if __name__ == "__main__":
    main()
