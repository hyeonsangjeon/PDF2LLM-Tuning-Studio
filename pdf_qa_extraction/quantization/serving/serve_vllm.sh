#!/usr/bin/env bash
# P1-3: launch a vLLM OpenAI-compatible server on a REAL run artifact and wait
# until it is healthy. Example expects the artifact a run actually produced
# (not a placeholder), e.g. runs/<run_id>/model/sft or quantization/artifacts/A_bf16_seed42.
#
#   ./quantization/serving/serve_vllm.sh <model_dir> [PORT] [QUANT]
#     QUANT: omit for bf16; use "torchao" for the INT4 (B/C) tile-packed artifacts.
#
# Stop the server:  kill "$(cat "$PID_FILE")"   (PID printed on start)
set -euo pipefail

MODEL_DIR="${1:?usage: serve_vllm.sh <model_dir> [port] [quant]}"
PORT="${2:-8000}"
QUANT="${3:-}"
HOST="${HOST:-127.0.0.1}"
LOG_FILE="${LOG_FILE:-/tmp/vllm_serve_${PORT}.log}"
PID_FILE="${PID_FILE:-/tmp/vllm_serve_${PORT}.pid}"

if [[ ! -e "$MODEL_DIR" ]]; then
  echo "[serve] ERROR: model dir not found: $MODEL_DIR" >&2
  echo "[serve] pass a REAL run artifact (e.g. runs/<run_id>/model/sft), not a placeholder." >&2
  exit 2
fi

QUANT_ARG=()
[[ -n "$QUANT" ]] && QUANT_ARG=(--quantization "$QUANT")

# FlashInfer sampler JIT needs nvcc; disable to avoid a hard failure on lean images.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "[serve] starting vLLM on $HOST:$PORT for $MODEL_DIR ${QUANT:+(quant=$QUANT)}"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --host "$HOST" --port "$PORT" \
  --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  "${QUANT_ARG[@]}" \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[serve] pid $(cat "$PID_FILE")  log $LOG_FILE"

echo "[serve] waiting for /health ..."
for i in $(seq 1 "${HEALTH_TRIES:-180}"); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "[serve] healthy after ${i}s — base_url http://$HOST:$PORT"
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[serve] ERROR: process exited early; last log lines:" >&2
    tail -n 30 "$LOG_FILE" >&2 || true
    exit 3
  fi
  sleep 1
done
echo "[serve] ERROR: not healthy within timeout; see $LOG_FILE" >&2
exit 4
