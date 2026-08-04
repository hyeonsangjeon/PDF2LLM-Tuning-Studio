"""Data-boundary policy + fail-closed egress gate + PDF threat gate (P0-9).

Provider selection is a data-policy decision, not a convenience flag. A document
that is ``restricted`` (or whose classification is missing/unknown) must be
blocked from any cloud provider *before* a provider object is created. The gate
is fail-closed: unknown classification is treated as ``restricted``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Providers that keep raw content on the local machine.
LOCAL_PROVIDERS = {"ollama", "replay", "recorded_replay", "local", "none"}
# Providers that transmit raw content to an external service.
CLOUD_PROVIDERS = {"azure", "openai", "aws", "bedrock", "anthropic", "gemini", "vertex"}

_VALID_CLASSIFICATIONS = ("public", "internal", "restricted")

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 300


class EgressBlocked(Exception):
    """Raised when a provider call would violate the document policy."""


class PDFQuarantined(Exception):
    """Raised when a PDF fails the threat gate and must not be processed."""

    def __init__(self, reason_code: str, message: str = ""):
        super().__init__(message or reason_code)
        self.reason_code = reason_code


@dataclass
class DocumentPolicy:
    classification: str = "restricted"          # fail-closed default
    license: str = "unknown"
    allowed_providers: List[str] = field(default_factory=list)
    raw_content_egress: str = "denied"           # allowed | denied
    retention: str = "delete_after_run"

    @staticmethod
    def from_dict(d: Optional[dict]) -> "DocumentPolicy":
        d = d or {}
        cls = str(d.get("classification", "")).lower()
        if cls not in _VALID_CLASSIFICATIONS:
            cls = "restricted"  # missing/unknown => restricted
        return DocumentPolicy(
            classification=cls,
            license=str(d.get("license", "unknown")),
            allowed_providers=[str(p).lower() for p in d.get("allowed_providers", [])],
            raw_content_egress=str(d.get("raw_content_egress", "denied")).lower(),
            retention=str(d.get("retention", "delete_after_run")),
        )


def is_cloud_provider(provider: str) -> bool:
    return _norm(provider) in CLOUD_PROVIDERS


def _norm(provider: str) -> str:
    return (provider or "").lower().replace("-", "_")


def egress_decision(policy: DocumentPolicy, provider: str) -> tuple[bool, str]:
    """Return (allowed, reason). Fail-closed for restricted / denied egress."""
    provider = (provider or "").lower()
    pnorm = _norm(provider)
    allowed_norm = {_norm(p) for p in policy.allowed_providers}
    if pnorm in LOCAL_PROVIDERS:
        return True, "local provider (no raw-content egress)"
    if not is_cloud_provider(provider):
        # unknown provider name -> treat as external, block unless allow-listed
        if pnorm not in allowed_norm:
            return False, f"unknown provider {provider!r} not in allowed_providers"
    if policy.classification == "restricted":
        return False, "classification=restricted forbids cloud egress"
    if policy.raw_content_egress != "allowed":
        return False, "raw_content_egress != allowed"
    if allowed_norm and pnorm not in allowed_norm:
        return False, f"provider {provider!r} not in allowed_providers"
    return True, "permitted by policy"


def guard_provider_call(policy: DocumentPolicy, provider: str) -> None:
    """Raise :class:`EgressBlocked` BEFORE any provider object is created."""
    allowed, reason = egress_decision(policy, provider)
    if not allowed:
        raise EgressBlocked(f"egress blocked for provider={provider!r}: {reason}")


# --------------------------------------------------------------------------- #
# PDF threat gate                                                              #
# --------------------------------------------------------------------------- #
def inspect_pdf(path: str, *, max_bytes: int = MAX_PDF_BYTES, max_pages: int = MAX_PDF_PAGES) -> dict:
    """Validate a PDF is safe to parse; raise :class:`PDFQuarantined` otherwise.

    Returns a small report dict on success.
    """
    if not os.path.isfile(path):
        raise PDFQuarantined("missing_file", path)
    size = os.path.getsize(path)
    if size == 0:
        raise PDFQuarantined("empty_file", path)
    if size > max_bytes:
        raise PDFQuarantined("size_limit", f"{size} > {max_bytes}")

    with open(path, "rb") as fh:
        head = fh.read(5)
    if head[:4] != b"%PDF":
        raise PDFQuarantined("bad_magic_bytes", "not a %PDF file")

    encrypted = False
    n_pages = None
    has_js = False
    has_attachments = False
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(path)
        encrypted = bool(getattr(reader, "is_encrypted", False))
        if not encrypted:
            n_pages = len(reader.pages)
        root = reader.trailer.get("/Root", {}) if hasattr(reader, "trailer") else {}
        names = root.get("/Names", {}) if hasattr(root, "get") else {}
        if hasattr(names, "get"):
            has_js = "/JavaScript" in names or "/JS" in names
            has_attachments = "/EmbeddedFiles" in names
        if "/OpenAction" in (root or {}):
            has_js = True
    except PDFQuarantined:
        raise
    except Exception:  # noqa: BLE001 - parser hiccup is not fatal to the gate
        pass

    if encrypted:
        raise PDFQuarantined("encrypted_pdf", "encrypted PDFs are not processed")
    if has_js:
        raise PDFQuarantined("embedded_javascript", "PDF contains JavaScript / OpenAction")
    if has_attachments:
        raise PDFQuarantined("embedded_attachment", "PDF contains embedded files")
    if n_pages is not None and n_pages > max_pages:
        raise PDFQuarantined("page_limit", f"{n_pages} > {max_pages}")

    return {"ok": True, "size_bytes": size, "n_pages": n_pages}
