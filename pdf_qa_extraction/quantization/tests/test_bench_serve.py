"""P1-3 tests — online serving benchmark measurement path against a fake SSE server.

No GPU / no vLLM: a tiny in-process HTTP server emits OpenAI-compatible streaming
chunks, so the TTFT/TPOT/throughput/goodput/percentile logic is fully exercised.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quantization.serving import v2_bench_serve as bs
from quantization.serving import client as serving_client

_TOKENS = ["광주", "는", " 남서부", "에", " 있다", "."]


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):  # /health, /version
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        elif self.path == "/version":
            body = json.dumps({"version": "fake-0.0"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if not self.path.endswith("/chat/completions"):
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        time.sleep(0.02)  # simulated prefill -> TTFT > 0
        for i, tok in enumerate(_TOKENS):
            send({"choices": [{"delta": {"content": tok}, "index": 0}]})
            time.sleep(0.005)  # inter-token latency
        send({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
              "usage": {"completion_tokens": len(_TOKENS)}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture(scope="module")
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    host, port = srv.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        srv.shutdown()


def test_stream_one_measures_ttft_and_tokens(fake_server):
    r = bs.stream_one(fake_server, "fake-model", "질문?", max_tokens=32)
    assert r.ok and r.status == 200
    assert r.ttft_s is not None and r.ttft_s > 0
    assert r.out_tokens == len(_TOKENS)          # from usage
    assert len(r.itl_s) == len(_TOKENS) - 1      # inter-token gaps
    assert r.e2e_s >= r.ttft_s
    assert r.tpot_s is not None and r.tpot_s > 0


def test_online_sweep_aggregates(fake_server):
    doc = bs.run_online_benchmark(
        fake_server, "fake-model", bs.default_prompts(4),
        concurrency_levels=[1, 4], num_prompts=8,
        sla_ttft_s=5.0, sla_e2e_s=10.0)
    assert doc["status"] == "live"
    assert [lvl["concurrency"] for lvl in doc["results"]] == [1, 4]
    for lvl in doc["results"]:
        assert lvl["requests"] == 8
        assert lvl["completed"] == 8 and lvl["failed"] == 0
        assert lvl["failure_rate"] == 0.0
        assert isinstance(lvl["request_throughput_rps"], (int, float)) and lvl["request_throughput_rps"] > 0
        assert lvl["output_token_throughput_tps"] > 0
        for p in ("p50", "p95", "p99"):
            assert lvl["ttft_s"][p] is not None and lvl["ttft_s"][p] > 0
            assert lvl["e2e_s"][p] is not None
        assert lvl["goodput"]["rate"] == 1.0     # generous SLA -> all good


def test_concurrency_speeds_up_wall_time(fake_server):
    """Higher concurrency should not serialize: 8-way wall < 8x single-request time."""
    doc = bs.run_online_benchmark(
        fake_server, "fake-model", bs.default_prompts(2),
        concurrency_levels=[8], num_prompts=8, sla_ttft_s=5.0, sla_e2e_s=10.0)
    lvl = doc["results"][0]
    per_req_e2e = lvl["e2e_s"]["p50"]
    assert lvl["wall_s"] < per_req_e2e * 8       # real parallelism


def test_failure_counted_on_bad_endpoint():
    r = bs.stream_one("http://127.0.0.1:9", "m", "q", timeout=1.0)  # nothing listening
    assert r.ok is False and r.error


def test_percentile_helper():
    assert bs.pct([], 0.5) is None
    assert bs.pct([1.0], 0.99) == 1.0
    assert bs.pct([1, 2, 3, 4], 0.5) in (2.0, 3.0)


def test_tight_sla_lowers_goodput(fake_server):
    # Impossible TTFT SLA -> goodput 0 even though all requests succeed.
    doc = bs.run_online_benchmark(
        fake_server, "fake-model", bs.default_prompts(2),
        concurrency_levels=[2], num_prompts=4,
        sla_ttft_s=1e-9, sla_e2e_s=1e-9)
    lvl = doc["results"][0]
    assert lvl["completed"] == 4
    assert lvl["goodput"]["rate"] == 0.0


def test_client_ask_streams(fake_server):
    ans = serving_client.ask(fake_server, "fake-model", "광주는?", context="광주는 남서부에 있다.")
    assert ans == "".join(_TOKENS)
