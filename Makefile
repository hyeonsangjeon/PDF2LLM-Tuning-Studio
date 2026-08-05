# PDF2LLM-Tuning-Studio — one-command golden paths.
#
# All targets run from source (no install needed). The credential-free demo
# (`make demo-replay`) is deterministic, CPU-only and needs no network.
PKG := pdf_qa_extraction
PY  ?= python3
RUN := cd $(PKG) && PYTHONPATH=. $(PY) -m pdf_qa.cli

.PHONY: help demo-replay demo-live-ollama demo-train-smoke verify-demo ask build-fixture test scan-secrets install install-workflow

help:
	@echo "PDF2LLM-Tuning-Studio targets:"
	@echo "  make demo-replay        Credential-free replay of the synthetic finance demo (CPU, offline)"
	@echo "  make demo-train-smoke   Replay demo + tiny CPU SFT smoke (downloads a tiny model)"
	@echo "  make demo-live-ollama   Optional local live path (requires a running Ollama daemon)"
	@echo "  make verify-demo        Run the replay demo and assert evidence/eval integrity"
	@echo "  make ask                Ask the benchmark's real A100-trained models a question (offline replay)"
	@echo "  make build-fixture      Regenerate the synthetic demo fixture (PDFs + gold Q&A)"
	@echo "  make test               Run the full test suite"
	@echo "  make scan-secrets       Run the secret/PII scanner over the repo"
	@echo "  make install-workflow   pip install the package with the workflow + train extras"

demo-replay:
	$(RUN) demo-replay

demo-live-ollama:
	$(RUN) demo-live-ollama

demo-train-smoke:
	$(RUN) demo-train-smoke

verify-demo:
	$(RUN) verify-demo

ask:
	$(RUN) ask $(ARGS)

build-fixture:
	$(RUN) build-fixture

scan-secrets:
	$(RUN) scan-secrets

test:
	cd $(PKG) && PYTHONPATH=. $(PY) -m pytest -q

install:
	cd $(PKG) && $(PY) -m pip install -e .

install-workflow:
	cd $(PKG) && $(PY) -m pip install -e ".[workflow,train]"
