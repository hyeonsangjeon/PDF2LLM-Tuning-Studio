"""P2-1: public trust/security/data docs.

These assert the three cross-linked public docs exist, link to each other
(instead of duplicating policy), match the *actual* web-app upload behavior, and
never over-claim controls the code does not implement. They read the real repo
files, so a drift between docs and code (or a missing/renamed doc) fails CI.
"""

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SECURITY = _REPO / "SECURITY.md"
_TRUST = _REPO / "docs" / "TRUST_AND_DATA.md"
_LICENSES = _REPO / "docs" / "DATA_AND_LICENSES.md"
_APP = _REPO / "pdf_qa_extraction" / "webapp" / "app.py"


def _read(p: Path) -> str:
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert text.strip(), f"empty {p}"
    return text


def test_trust_docs_exist_and_nonempty():
    for p in (_SECURITY, _TRUST, _LICENSES):
        _read(p)


def test_trust_docs_cross_link_each_other():
    sec, trust, lic = _read(_SECURITY), _read(_TRUST), _read(_LICENSES)
    # SECURITY.md links the other two.
    assert "docs/TRUST_AND_DATA.md" in sec
    assert "docs/DATA_AND_LICENSES.md" in sec
    # TRUST_AND_DATA.md links SECURITY + LICENSES.
    assert "SECURITY.md" in trust
    assert "DATA_AND_LICENSES.md" in trust
    # DATA_AND_LICENSES.md links SECURITY + TRUST.
    assert "SECURITY.md" in lic
    assert "TRUST_AND_DATA.md" in lic


def test_upload_limits_documented_match_code():
    # The documented knobs/signature must actually exist in the web app, so the
    # docs describe real behavior rather than aspiration.
    app_src = _read(_APP)
    trust = _read(_TRUST)
    for token in ("PDFQA_MAX_UPLOAD_MB", "%PDF-"):
        assert token in app_src, f"{token} absent from app.py"
        assert token in trust, f"{token} absent from TRUST_AND_DATA.md"
    # The 413/415 guards the doc promises are wired in code.
    assert "status_code=413" in app_src
    assert "status_code=415" in app_src


def test_docs_do_not_overclaim_unimplemented_controls():
    trust = _read(_TRUST).lower()
    # Never assert compliance/auth as *implemented*.
    for bad in (
        "soc 2 certified",
        "iso 27001 certified",
        "hipaa compliant",
        "gdpr compliant",
        "fully authenticated",
    ):
        assert bad not in trust, f"over-claim found: {bad!r}"
    # Must frame the boundary honestly.
    assert "not supported" in trust
    normalized = trust.replace("*", "")
    assert "no authentication" in normalized


def test_license_ledger_records_known_unknowns():
    lic = _read(_LICENSES).lower()
    # The ledger must not silently omit the un-provenanced sample nor imply MIT
    # covers datasets/models.
    assert "fsi_data.pdf" in lic
    assert "known unknown" in lic or "known-unknown" in lic
    assert "korquad" in lic
    assert "apache" in lic  # base-model license recorded
