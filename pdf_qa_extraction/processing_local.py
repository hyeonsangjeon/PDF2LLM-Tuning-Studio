#!/usr/bin/env python3
"""Backward-compatible shim: local run with **AWS Bedrock**.

Preferred command going forward::

    LLM_PROVIDER=bedrock python run_local.py

This file is kept so previously documented commands keep working.
"""

import os

os.environ.setdefault("LLM_PROVIDER", "bedrock")

from run_local import main  # noqa: E402

if __name__ == "__main__":
    main()
