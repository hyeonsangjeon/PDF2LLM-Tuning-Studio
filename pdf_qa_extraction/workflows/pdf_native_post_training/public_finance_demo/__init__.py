"""Packaged synthetic Korean-finance demo fixture (data + reproducible builder).

This subpackage ships the credential-free golden-path corpus used by
``pdf2llm verify-demo`` / ``make demo-replay``: the v1/v2 PDFs under ``docs/``,
the schema-valid ``gold_qa.jsonl`` (26 grounded Q&A), the deterministic
``recorded_generations.jsonl`` replay cache, the canary ledger and checksums.
``build_fixture.py`` regenerates every artifact from scratch.
"""
