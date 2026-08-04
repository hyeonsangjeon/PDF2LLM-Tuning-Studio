#!/usr/bin/env bash
# P1-3: run the ONLINE serving benchmark against a already-running vLLM endpoint.
# Verifies server readiness, runs a concurrency sweep, then reminds you to stop
# the server. Writes raw results under the run dir (gitignored) by default.
#
#   ./quantization/serving/bench.sh \
#       --model runs/<run_id>/model/sft \
#       --concurrency 1,8,32 \
#       --output runs/<run_id>/serving/raw/online.json
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL=""
CONCURRENCY="1,8,32"
OUTPUT=""
NUM_PROMPTS="0"
MAX_TOKENS="128"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --concurrency) CONCURRENCY="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2;;
    --max-tokens) MAX_TOKENS="$2"; shift 2;;
    *) echo "[bench] unknown arg: $1" >&2; exit 2;;
  esac
done

: "${MODEL:?--model is required (a REAL served artifact path/name)}"
: "${OUTPUT:?--output is required}"

echo "[bench] health check $BASE_URL/health"
if ! curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
  echo "[bench] ERROR: server not healthy at $BASE_URL — start it with serve_vllm.sh first." >&2
  exit 3
fi

# best-effort artifact provenance
ART_SHA=""
if [[ -d "$MODEL" ]]; then
  ART_SHA="$(find "$MODEL" -type f \( -name '*.safetensors' -o -name '*.bin' \) -print0 2>/dev/null \
    | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1 || true)"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[bench] running sweep concurrency=$CONCURRENCY -> $OUTPUT"
PYTHONPATH="$HERE" python -m quantization.serving.v2_bench_serve \
  --base-url "$BASE_URL" --model "$MODEL" \
  --concurrency "$CONCURRENCY" --num-prompts "$NUM_PROMPTS" \
  --max-tokens "$MAX_TOKENS" \
  ${ART_SHA:+--artifact-sha256 "$ART_SHA"} \
  --output "$OUTPUT"

echo "[bench] done. Remember to STOP the server:  kill \$(cat /tmp/vllm_serve_*.pid)"
