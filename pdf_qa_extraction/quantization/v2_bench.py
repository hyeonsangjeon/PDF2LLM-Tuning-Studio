"""v2 serving benchmark (W4 fix) — vLLM throughput sweep + TTFT/latency percentiles.

Runs on the REAL A/B/C artifacts (not base weights). For each method it:
  * loads the artifact under vLLM (graph mode / CUDA graphs; torchao int4 for B/C),
  * sweeps concurrency (1..256) with a fixed decode length to find the int4<->bf16
    throughput crossover (int4 wins memory-bound single-stream; bf16 wins compute-bound
    large batch),
  * reports TTFT and end-to-end latency p50/p99 from vLLM's per-request metrics.

vLLM spawns worker subprocesses, so this MUST run under ``if __name__ == '__main__'``.

Usage:
  python v2_bench.py --model-dir artifacts/A_bf16_seed42 --method A_bf16 --precision bf16 \
      --out results/bench_A.json
  python v2_bench.py --model-dir artifacts/B_int4_ptq_seed42 --method B_int4_ptq \
      --precision int4 --out results/bench_B.json
"""
import argparse
import json
import os
import statistics
import sys
import time


def build_prompts(n, model_dir):
    """Realistic Korean QA prompts (chat-templated) from the KorQuAD val slice."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from quantization.v2_pipeline import build_chat_prompt, FALLBACK_SYSTEM

    tok = AutoTokenizer.from_pretrained(model_dir)
    ds = load_dataset("KorQuAD/squad_kor_v1")["validation"].shuffle(seed=42).select(range(n))
    prompts = []
    for r in ds:
        try:
            p = build_chat_prompt(tok, FALLBACK_SYSTEM, r["context"], r["question"],
                                  None, False, True)
        except Exception:
            p = f"[문맥]\n{r['context']}\n\n[질문]\n{r['question']}\n답:"
        prompts.append(p)
    return prompts


def pct(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
    return round(vals[k], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--precision", choices=["bf16", "int4"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--batches", default="1,4,16,32,64,128,256")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--mode", choices=["sweep", "latency", "both"], default="both",
                    help="sweep=throughput batch sweep; latency=single-stream TTFT/e2e; both")
    ap.add_argument("--latency-n", type=int, default=24,
                    help="number of single-stream requests for latency percentiles")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(model=args.model_dir, dtype="bfloat16", enforce_eager=False,
                      gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
                      disable_log_stats=False)
    if args.precision == "int4":
        llm_kwargs["quantization"] = "torchao"
    llm = LLM(**llm_kwargs)

    prompts_pool = build_prompts(max(int(b) for b in args.batches.split(",")), args.model_dir)
    sp = SamplingParams(temperature=0.0, max_tokens=args.out_tokens, ignore_eos=True)

    # warmup (build CUDA graphs, page KV)
    llm.generate(prompts_pool[:4], sp)

    sweep = []
    if args.mode in ("sweep", "both"):
        for b in [int(x) for x in args.batches.split(",")]:
            prompts = (prompts_pool * ((b // len(prompts_pool)) + 1))[:b]
            t0 = time.time()
            outs = llm.generate(prompts, sp)
            elapsed = time.time() - t0
            gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
            ttfts, e2es = [], []
            for o in outs:
                m = getattr(o, "metrics", None)
                if m and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
                    ttfts.append(m.first_token_time - m.arrival_time)
                if m and getattr(m, "finished_time", None) and getattr(m, "arrival_time", None):
                    e2es.append(m.finished_time - m.arrival_time)
            row = {
                "batch": b, "elapsed_s": round(elapsed, 3), "gen_tokens": gen_tokens,
                "throughput_tok_s": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0.0,
                "ttft_p50_s": pct(ttfts, 0.50), "ttft_p99_s": pct(ttfts, 0.99),
                "e2e_p50_s": pct(e2es, 0.50), "e2e_p99_s": pct(e2es, 0.99),
            }
            sweep.append(row)
            print(f"[bench {args.method}] batch={b:4d} thr={row['throughput_tok_s']:9.1f} tok/s "
                  f"ttft_p50={row['ttft_p50_s']} p99={row['ttft_p99_s']} "
                  f"e2e_p50={row['e2e_p50_s']} p99={row['e2e_p99_s']}", flush=True)

    # single-stream latency (W4: TTFT + e2e p50/p99) — harness-timed, version-independent.
    # vLLM 0.23 V1 offline does not populate RequestOutput.metrics, so we measure the
    # client-observed wall-clock latency of batch=1 requests directly: TTFT = latency of a
    # max_tokens=1 request (prefill + first decode); e2e = latency of the full decode length.
    latency = None
    if args.mode in ("latency", "both"):
        lat_prompts = (prompts_pool * ((args.latency_n // len(prompts_pool)) + 1))[:args.latency_n]
        sp_ttft = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        sp_e2e = SamplingParams(temperature=0.0, max_tokens=args.out_tokens, ignore_eos=True)
        llm.generate(lat_prompts[:1], sp_e2e, use_tqdm=False)  # warmup
        ttfts, e2es = [], []
        for p in lat_prompts:
            t0 = time.time(); llm.generate([p], sp_ttft, use_tqdm=False); ttfts.append(time.time() - t0)
            t0 = time.time(); llm.generate([p], sp_e2e, use_tqdm=False); e2es.append(time.time() - t0)
        latency = {
            "n": len(lat_prompts), "concurrency": 1, "out_tokens": args.out_tokens,
            "ttft_p50_s": pct(ttfts, 0.50), "ttft_p99_s": pct(ttfts, 0.99),
            "ttft_mean_s": round(statistics.mean(ttfts), 4),
            "e2e_p50_s": pct(e2es, 0.50), "e2e_p99_s": pct(e2es, 0.99),
            "e2e_mean_s": round(statistics.mean(e2es), 4),
        }
        print(f"[bench {args.method}] latency single-stream n={latency['n']}: "
              f"ttft_p50={latency['ttft_p50_s']}s p99={latency['ttft_p99_s']}s "
              f"e2e_p50={latency['e2e_p50_s']}s p99={latency['e2e_p99_s']}s", flush=True)

    result = {"method": args.method, "precision": args.precision, "model_dir": args.model_dir,
              "out_tokens": args.out_tokens, "graph_mode": True, "sweep": sweep,
              "latency_single_stream": latency}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("[bench] wrote", args.out)


if __name__ == "__main__":
    main()
