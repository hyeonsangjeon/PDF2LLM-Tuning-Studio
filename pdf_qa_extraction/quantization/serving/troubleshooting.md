# Serving troubleshooting (P1-3)

Failure → cause → fix for standing up the vLLM OpenAI-compatible server and running
the online benchmark. These are the concrete issues hit while producing the v2 results.

## Model load fails
- **`model dir not found` / wrong path** → pass a **real run artifact** (e.g.
  `runs/<run_id>/model/sft` or `quantization/artifacts/A_bf16_seed42`), never a
  placeholder. `serve_vllm.sh` exits `2` before launching if the path is missing.
- **INT4 (B/C) artifact won't load as bf16** → the PTQ/QAT artifacts are torchao
  tile-packed INT4. Serve them with `serve_vllm.sh <dir> <port> torchao`
  (`--quantization torchao`). Loading an INT4 dir without the quant flag fails or
  silently mis-dequantizes.
- **`safetensors`/serialization error on INT4** → torchao INT4 is saved with
  `safe_serialization=False`; ensure the artifact was exported that way (the v2
  pipeline does this) and that `torchao` matches the training version.

## OOM (CUDA out of memory)
- Lower `--gpu-memory-utilization` (env `GPU_MEM_UTIL`, default 0.85) to e.g. 0.80.
- Lower `--max-model-len` (env `MAX_MODEL_LEN`, default 4096). One KorQuAD prompt
  needs ~2049 tokens, so do **not** drop below ~2560.
- Serve one model at a time; the 3-way comparison runs **sequentially** on a single
  A100, not concurrently.

## CUDA / kernel compatibility
- **FlashInfer sampler JIT needs `nvcc`** (absent on lean images) → the launcher
  exports `VLLM_USE_FLASHINFER_SAMPLER=0` to use the native sampler. Override only
  if your image has the CUDA toolkit.
- **torch/torchao/vLLM version skew** → pin the trio the artifact was produced with
  (see `quantization/results/vllm_throughput.json` → `engine`: torch 2.11+cu130,
  torchao 0.17.0, vllm 0.23.0). Mismatched torchao can't read the INT4 layout.

## Server never becomes healthy
- `serve_vllm.sh` polls `/health`; if the process exits early it prints the last 30
  log lines. Inspect `/tmp/vllm_serve_<port>.log`.
- Port already in use → pass a different `PORT` (2nd arg).

## Benchmark issues
- **`server not healthy` from `bench.sh`** → start `serve_vllm.sh` first and wait for
  the `healthy` line; `bench.sh` refuses to run against a down server.
- **All requests fail / connection refused** → check `--base-url` matches the served
  `HOST:PORT`; check a firewall isn't blocking localhost.
- **`online TTFT` looks like the offline proxy** → they are different. The offline
  `v2_bench.py` `max_tokens=1` value is a **TTFT proxy**; only `v2_bench_serve.py`
  (HTTP streaming) reports a real online TTFT.

## Cleanup (always)
- Stop the server: `kill "$(cat /tmp/vllm_serve_<port>.pid)"`.
- Confirm the GPU is released: `nvidia-smi` shows no python process holding memory.
- Deprovision the VM/endpoint when done to stop billing (e.g. `az group delete ...`).
