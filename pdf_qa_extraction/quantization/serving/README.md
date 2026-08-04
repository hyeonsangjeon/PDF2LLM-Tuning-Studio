# Online serving benchmark (P1-3)

`v2_bench.py` measures **offline** `LLM.generate` batch throughput. This directory adds
the **online** serving path: stand up an OpenAI-compatible vLLM endpoint on a **real run
artifact** and measure true HTTP-streaming latency/throughput under concurrency. The two
are reported as **separate tables** — an offline batch number is not an online SLA.

| file | role |
|---|---|
| `serve_vllm.sh` | launch vLLM `api_server` on a real artifact + wait for `/health` |
| `client.py` | ask one question against the endpoint (streaming) |
| `v2_bench_serve.py` | concurrency sweep → TTFT/TPOT/ITL/throughput/goodput/failure p50·p95·p99 |
| `bench.sh` | readiness check → run `v2_bench_serve.py` → stop reminder |
| `troubleshooting.md` | model-load / OOM / CUDA-compat / cleanup |

## Clean-machine flow (start → health → sample → benchmark → stop)

```bash
# 0) from repo root, deps installed; GPU + the run's real artifact present.
cd pdf_qa_extraction

# 1) START + HEALTH — serve a real artifact (bf16). For INT4 (B/C) add: torchao
./quantization/serving/serve_vllm.sh runs/<run_id>/model/sft 8000
#   -> prints pid + "healthy after Ns — base_url http://127.0.0.1:8000"

# 2) SAMPLE request (sanity)
python quantization/serving/client.py \
  --base-url http://127.0.0.1:8000 \
  --model runs/<run_id>/model/sft \
  --context "광주는 대한민국 남서부의 광역시이다." \
  --question "광주는 어느 지역에 있는가?"

# 3) BENCHMARK — online concurrency sweep, raw JSON under the run dir (gitignored)
./quantization/serving/bench.sh \
  --model runs/<run_id>/model/sft \
  --concurrency 1,8,32 \
  --output runs/<run_id>/serving/raw/online.json

# 4) STOP the server + confirm the GPU is released
kill "$(cat /tmp/vllm_serve_8000.pid)"
nvidia-smi   # no python process should hold memory
```

## What it measures

Per concurrency level, over the requests it sends (scale `--num-prompts` to a few
hundred for stable tails): request rate, max concurrency, **TTFT / TPOT / ITL**,
request throughput, output-token throughput, **goodput** (fraction meeting an SLA:
`--sla-ttft-s`, `--sla-e2e-s`) and **failure rate**, each with **p50/p95/p99**. The
raw JSON records the server base URL, served model, vLLM version (queried from
`/version`) and the artifact SHA-256 for provenance.

## Offline vs online (two separate tables)

**Offline** (already produced — `quantization/results/vllm_throughput.json`, batch
`LLM.generate`): batch-size sweep 1…256, single-stream TTFT proxy via `max_tokens=1`,
crossover ≈ batch 16, clean weight VRAM bf16 15.27 / int4 6.05 GiB.

**Online** (this tool, HTTP streaming): **`planned`** — populated only when you run
`bench.sh` against a served artifact on a GPU. No online numbers are committed here,
and the offline `max_tokens=1` value is kept explicitly as a **TTFT proxy**, never
relabeled as an online TTFT.

## Notes / guardrails

- **CPU-testable transport.** `v2_bench_serve.py` talks plain OpenAI SSE, so its
  measurement logic is covered by `quantization/tests/test_bench_serve.py` against a
  fake in-process SSE server (no GPU, no vLLM). Real numbers still require a GPU serve.
- These commands **do not modify** `quantization/results/`; raw output goes under
  `runs/<run_id>/serving/raw/` (gitignored) or a path you pass.
- The PDF-native workflow **calls** this serving interface rather than duplicating the
  scripts under `workflows/` (one-way dependency; see the workflow README).
- Always **stop the server and deprovision** the VM/endpoint afterwards to stop billing.
