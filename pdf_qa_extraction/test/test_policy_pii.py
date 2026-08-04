"""Korean PII baseline detector + mechanical-fake canary validator tests.

scan-secrets: allow-file (fixtures below embed PII/secret *shapes* on purpose to
exercise the detector; none are real).
"""
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pdf_qa import pii  # noqa: E402
from pdf_qa import policy  # noqa: E402
from pdf_qa.policy import DocumentPolicy, EgressBlocked, guard_provider_call  # noqa: E402


# --------------------------------------------------------------------------- #
# PII baseline + mechanical-fake canary validator                             #
# --------------------------------------------------------------------------- #
def test_synthetic_canaries_detected_but_flagged_fake():
    # Luhn-invalid card, reserved example email, reserved-phone-block, impossible RRN date
    text = (
        "카드 4111-1111-1111-1112 이메일 hong@example.com "
        "전화 010-555-0123 주민 991315-1234567"
    )
    hits = pii.detect(text)
    kinds = {h.kind for h in hits}
    assert {"card", "email", "phone", "rrn"} <= kinds  # gate actually triggers
    assert all(h.mechanically_fake for h in hits if h.kind in {"card", "email", "phone", "rrn"})
    assert pii.has_real_pii(text) is False  # canaries are not real PII


def test_real_shaped_pii_is_not_flagged_fake():
    # Luhn-valid card + real-looking email domain => treated as real PII
    text = "카드 4111-1111-1111-1111 메일 user@company.co.kr"
    assert pii.has_real_pii(text) is True


def test_redact_masks_all_shapes():
    red = pii.redact("전화 010-1234-5678 메일 a@b.com")
    assert "010-1234-5678" not in red and "a@b.com" not in red
    assert "[REDACTED_PHONE]" in red and "[REDACTED_EMAIL]" in red


# --------------------------------------------------------------------------- #
# Fail-closed egress gate                                                      #
# --------------------------------------------------------------------------- #
def test_missing_classification_defaults_restricted():
    p = DocumentPolicy.from_dict({"license": "CC-BY-4.0"})
    assert p.classification == "restricted"


def test_restricted_blocks_cloud_before_object_creation():
    p = DocumentPolicy.from_dict({"classification": "restricted"})
    cloud_calls = {"n": 0}

    class FakeCloudProvider:  # must never be constructed
        def __init__(self):
            cloud_calls["n"] += 1

    def run_with_provider(provider_name):
        guard_provider_call(p, provider_name)  # gate BEFORE construction
        return FakeCloudProvider()

    with pytest.raises(EgressBlocked):
        run_with_provider("azure")
    assert cloud_calls["n"] == 0  # zero cloud egress


def test_local_providers_always_allowed():
    p = DocumentPolicy.from_dict({"classification": "restricted"})
    for prov in ("ollama", "replay", "recorded_replay", "local"):
        guard_provider_call(p, prov)  # must not raise


def test_public_allows_listed_cloud_only():
    p = DocumentPolicy.from_dict({
        "classification": "public",
        "raw_content_egress": "allowed",
        "allowed_providers": ["azure"],
    })
    guard_provider_call(p, "azure")  # allowed
    with pytest.raises(EgressBlocked):
        guard_provider_call(p, "openai")  # not in allow-list


def test_unknown_provider_blocked_by_default():
    p = DocumentPolicy.from_dict({"classification": "public", "raw_content_egress": "allowed"})
    with pytest.raises(EgressBlocked):
        guard_provider_call(p, "some-random-service")


# --------------------------------------------------------------------------- #
# PDF threat gate                                                             #
# --------------------------------------------------------------------------- #
def test_bad_magic_bytes_quarantined(tmp_path):
    f = tmp_path / "fake.pdf"
    f.write_bytes(b"not a pdf at all")
    with pytest.raises(policy.PDFQuarantined) as ei:
        policy.inspect_pdf(str(f))
    assert ei.value.reason_code == "bad_magic_bytes"


def test_empty_file_quarantined(tmp_path):
    f = tmp_path / "empty.pdf"
    f.write_bytes(b"")
    with pytest.raises(policy.PDFQuarantined) as ei:
        policy.inspect_pdf(str(f))
    assert ei.value.reason_code == "empty_file"


def test_size_limit_quarantined(tmp_path):
    f = tmp_path / "big.pdf"
    f.write_bytes(b"%PDF-1.4\n" + b"0" * 2048)
    with pytest.raises(policy.PDFQuarantined) as ei:
        policy.inspect_pdf(str(f), max_bytes=1024)
    assert ei.value.reason_code == "size_limit"
