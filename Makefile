# PDF2LLM-Tuning-Studio — one-command golden paths.
#
# All targets run from source (no install needed). The credential-free demo
# (`make demo-replay`) is deterministic, CPU-only and needs no network.
PKG := pdf_qa_extraction
PY  ?= python3
RUN := cd $(PKG) && PYTHONPATH=. $(PY) -m pdf_qa.cli

.PHONY: help demo-replay demo-live-ollama demo-train-smoke verify-demo ask ask-hf bench publish-hf build-fixture test scan-secrets install install-workflow

help:
	@echo "PDF2LLM-Tuning-Studio targets:"
	@echo "  make demo-replay        Credential-free replay of the synthetic finance demo (CPU, offline)"
	@echo "  make demo-train-smoke   Replay demo + tiny CPU SFT smoke (downloads a tiny model)"
	@echo "  make demo-live-ollama   Optional local live path (requires a running Ollama daemon)"
	@echo "  make verify-demo        Run the replay demo and assert evidence/eval integrity"
	@echo "  make ask                Ask the benchmark's real A100-trained models a question (offline replay)"
	@echo "  make ask-hf HF=<repo|dir> Q=\"...\"  Load REAL fine-tuned weights (HF/local) and infer live"
	@echo "  make bench              Reproduce the 6-arm benchmark from scratch on YOUR GPU (writes runs/bench)"
	@echo "  make publish-hf MODEL_DIR=<dir> REPO=<id>  Upload fine-tuned weights to HuggingFace Hub"
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

# Load REAL fine-tuned weights (HF repo id OR local dir) and infer live — not a replay.
#   make ask-hf HF=your-name/pdf2llm-sft-qwen3-8b Q="2024년 연간 매출액은 얼마입니까?"
HF ?=
Q  ?= 2024년 연간 매출액은 얼마입니까?
ask-hf:
	$(RUN) ask --hf "$(HF)" -q "$(Q)" $(ARGS)

# Reproduce the 6-arm benchmark FROM SCRATCH on your own GPU (trains real fine-tuned weights).
# Writes to runs/bench so it never overwrites the committed historical_final results.
# Full 6 arms need a CUDA GPU (~1x A100, ~30 min); on CPU only the untrained base arms run.
# Publish the weights too:  make bench ARGS="--keep-artifacts --push-to-hub your-name/pdf2llm-sft-qwen3-8b"
BENCH_OUT ?= runs/bench
bench:
	cd $(PKG) && PYTHONPATH=. $(PY) -m workflows.pdf_native_post_training.benchmarks.pdf_native.run_arms --out-dir $(BENCH_OUT) $(ARGS)

# Upload fine-tuned weights to HuggingFace Hub (auto model card). Add ARGS="--dry-run" to preview.
#   make publish-hf MODEL_DIR=runs/bench/../sft_bf16_seed42 REPO=your-name/pdf2llm-sft-qwen3-8b
MODEL_DIR ?=
REPO ?=
publish-hf:
	$(RUN) publish-hf --model-dir "$(MODEL_DIR)" --repo-id "$(REPO)" $(ARGS)

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
