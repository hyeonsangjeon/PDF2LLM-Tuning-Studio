"""Smoke tests for the single-node local demo web app.

These only exercise the light metadata endpoints (``/``, ``/api/personas``,
``/api/device``, ``/api/providers``, ``/healthz``) plus input validation on
``/api/extract``. None of them import the heavy PDF/OCR or cloud-LLM stack, so
they run in a bare env. The test is skipped entirely when FastAPI (the optional
``webapp`` extra) is not installed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# The web app is an optional extra; skip cleanly when it (or the TestClient's
# httpx dependency) is unavailable rather than failing collection.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from webapp.app import app  # noqa: E402

client = TestClient(app)


def test_index_serves_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_personas_endpoint_lists_all_with_professor_default():
    resp = client.get("/api/personas")
    assert resp.status_code == 200
    data = resp.json()
    keys = [p["key"] for p in data["personas"]]

    assert data["default"] == "professor"
    # professor is always first; the ledger ships 7 personas.
    assert keys[0] == "professor"
    assert "memoirist" in keys
    assert "feynman" in keys
    assert len(keys) == 7
    # Every persona exposes a non-empty one-line method summary for the UI.
    assert all(p["method_summary"].strip() for p in data["personas"])
    # Each persona's method summary is distinct (personas are genuinely different).
    summaries = [p["method_summary"] for p in data["personas"]]
    assert len(set(summaries)) == len(summaries)


def test_device_endpoint_reports_gpu_readiness():
    resp = client.get("/api/device")
    assert resp.status_code == 200
    data = resp.json()
    # Device probing must never raise, even with no torch/GPU present.
    for field in ("gpu_ready", "summary", "onnxruntime_providers", "torch_installed"):
        assert field in data
    assert isinstance(data["gpu_ready"], bool)
    assert data["summary"]


def test_providers_endpoint_lists_three_backends():
    resp = client.get("/api/providers")
    assert resp.status_code == 200
    data = resp.json()
    names = {p["name"] for p in data["providers"]}
    assert names == {"azure", "openai", "bedrock"}
    assert all("configured" in p for p in data["providers"])


def test_extract_rejects_non_pdf():
    resp = client.post(
        "/api/extract",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"mode": "preview"},
    )
    assert resp.status_code == 400


def test_extract_rejects_unknown_persona():
    resp = client.post(
        "/api/extract",
        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        data={"mode": "preview", "persona": "does-not-exist"},
    )
    assert resp.status_code == 400
