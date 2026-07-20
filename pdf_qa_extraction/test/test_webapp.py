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


def test_providers_endpoint_lists_backends_including_local_ollama():
    resp = client.get("/api/providers")
    assert resp.status_code == 200
    data = resp.json()
    by_name = {p["name"]: p for p in data["providers"]}
    assert set(by_name) == {"azure", "openai", "bedrock", "ollama"}
    assert all("configured" in p for p in data["providers"])
    # Ollama is local / credential-free: always selectable, flagged as local.
    assert by_name["ollama"]["local"] is True
    assert by_name["ollama"]["configured"] is True


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


def test_meta_reports_bundled_sample():
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    data = resp.json()
    # The repo checkout ships pdf_qa_extraction/data/fsi_data.pdf, so the demo's
    # one-click sample must be advertised as available.
    assert data["sample_available"] is True
    assert data["sample_name"] == "fsi_data.pdf"
    assert data["sample_domain"]


def test_extract_requires_file_or_sample():
    # Neither an uploaded file nor use_sample -> a clear 400, not a 500.
    resp = client.post("/api/extract", data={"mode": "preview"})
    assert resp.status_code == 400


def test_extract_use_sample_still_validates_persona():
    # The use_sample branch must not bypass early persona validation.
    resp = client.post(
        "/api/extract",
        data={"mode": "preview", "use_sample": "true", "persona": "does-not-exist"},
    )
    assert resp.status_code == 400
