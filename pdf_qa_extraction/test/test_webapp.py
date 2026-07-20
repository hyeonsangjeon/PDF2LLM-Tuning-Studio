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


def test_settings_endpoint_grouped_and_secret_safe():
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    groups = {g["group"] for g in resp.json()["groups"]}
    assert {"core", "provider.azure", "provider.ollama"} <= groups
    for g in resp.json()["groups"]:
        for s in g["settings"]:
            assert "is_set" in s
            if s["secret"]:
                # Secret values are never exposed via the ledger endpoint.
                assert s["default"] == ""


def test_providers_reports_missing_required_vars(monkeypatch):
    # With no Azure endpoint, azure is unconfigured and names the missing var.
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    by_name = {p["name"]: p for p in client.get("/api/providers").json()["providers"]}
    assert by_name["azure"]["configured"] is False
    assert "AZURE_OPENAI_ENDPOINT" in by_name["azure"]["missing"]


def test_download_jsonl_is_a_file_attachment():
    pairs = [{"QUESTION": "q", "ANSWER": "a", "source": "text"}]
    resp = client.post("/api/download", json={"pairs": pairs, "name": "report.pdf"})
    assert resp.status_code == 200
    assert 'filename="report.qa.jsonl"' in resp.headers.get("content-disposition", "")
    assert "application/x-ndjson" in resp.headers.get("content-type", "")
    import json as _json

    assert _json.loads(resp.text.strip())["QUESTION"] == "q"


def test_download_manifest_exposes_figure_linkage():
    pairs = [
        {
            "source": "image",
            "image_path": "/f/fig-2.png",
            "page": 3,
            "section": "GDP",
            "figure_index": 2,
            "context_used": True,
        }
    ]
    resp = client.post(
        "/api/download", json={"pairs": pairs, "name": "d.pdf", "kind": "manifest"}
    )
    assert resp.status_code == 200
    assert 'filename="d.manifest.json"' in resp.headers.get("content-disposition", "")
    import json as _json

    m = _json.loads(resp.text)
    assert m["counts"]["image"] == 1
    assert m["figures"][0]["section"] == "GDP"
    assert m["figures"][0]["context_used"] is True


def test_download_requires_pairs():
    resp = client.post("/api/download", json={"name": "x"})
    assert resp.status_code == 400

