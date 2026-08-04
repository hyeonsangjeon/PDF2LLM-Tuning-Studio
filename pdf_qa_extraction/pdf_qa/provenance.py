"""Deterministic PDF provenance: text/table elements with immutable ids.

This is the *fast text parser* used by the credential-free replay path. It turns
a PDF into a flat list of elements, each carrying a parser-assigned immutable
``element_id``, page number, bounding box and verbatim text. Evidence can then
only reference ids the parser actually produced (a model cannot invent them).

PyMuPDF (``fitz``) is used for text blocks + bounding boxes and, when available,
table cells. The import is lazy so the module loads even where PyMuPDF is absent
(``parse_pdf`` then raises a clear error).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    """NFKC + collapse whitespace; used for quote matching and hashing."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_sha256(quote: str) -> str:
    return hashlib.sha256(normalize_text(quote).encode("utf-8")).hexdigest()


@dataclass
class Element:
    element_id: str
    page: int
    bbox: Tuple[float, float, float, float]
    text: str
    modality: str = "text"  # text | table | figure
    section_path: Optional[str] = None
    chunk_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "page": self.page,
            "bbox": [round(float(x), 2) for x in self.bbox],
            "text": self.text,
            "modality": self.modality,
            "section_path": self.section_path,
            "chunk_id": self.chunk_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "Element":
        bbox = tuple(d.get("bbox") or (0.0, 0.0, 0.0, 0.0))
        return Element(
            element_id=d["element_id"],
            page=int(d["page"]),
            bbox=bbox,  # type: ignore[arg-type]
            text=d.get("text", ""),
            modality=d.get("modality", "text"),
            section_path=d.get("section_path"),
            chunk_id=d.get("chunk_id"),
        )


@dataclass
class Document:
    path: str
    sha256: str
    version: Optional[str]
    n_pages: int
    elements: List[Element] = field(default_factory=list)

    def by_id(self) -> Dict[str, Element]:
        return {e.element_id: e for e in self.elements}

    def page_count(self) -> int:
        return self.n_pages

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "version": self.version,
            "n_pages": self.n_pages,
            "elements": [e.to_dict() for e in self.elements],
        }

    @staticmethod
    def from_dict(d: dict) -> "Document":
        return Document(
            path=d.get("path", ""),
            sha256=d["sha256"],
            version=d.get("version"),
            n_pages=int(d.get("n_pages", 0)),
            elements=[Element.from_dict(e) for e in d.get("elements", [])],
        )


_HEADING_RE = re.compile(r"^(제?\s*\d+[\.\)]|[0-9]+\.|[가-힣A-Za-z]{1,20}\s*$)")


def _looks_like_heading(text: str) -> bool:
    t = text.strip()
    return len(t) <= 40 and ("\n" not in t) and bool(_HEADING_RE.match(t))


def parse_pdf(path: str, version: Optional[str] = None) -> Document:
    """Parse ``path`` into a :class:`Document` of immutable-id elements."""
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyMuPDF (fitz) is required for PDF parsing. Install the 'pdf' extra."
        ) from exc

    sha = _sha256_file(path)
    elements: List[Element] = []
    with fitz.open(path) as doc:
        n_pages = doc.page_count
        for pno in range(n_pages):
            page = doc[pno]
            section: Optional[str] = None

            # 1) text blocks (sorted top-to-bottom, left-to-right)
            blocks = [b for b in page.get_text("blocks") if (b[4] or "").strip()]
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
            for bi, b in enumerate(blocks):
                x0, y0, x1, y1, raw = b[0], b[1], b[2], b[3], b[4]
                text = normalize_text(raw)
                if not text:
                    continue
                if _looks_like_heading(raw):
                    section = text
                elements.append(
                    Element(
                        element_id=f"p{pno + 1}-b{bi}",
                        page=pno + 1,
                        bbox=(x0, y0, x1, y1),
                        text=text,
                        modality="text",
                        section_path=section,
                    )
                )

            # 2) tables (best-effort; PyMuPDF >= 1.23)
            try:
                tables = page.find_tables()
                tabs = getattr(tables, "tables", tables)
            except Exception:  # noqa: BLE001
                tabs = []
            for ti, tab in enumerate(tabs):
                try:
                    rows = tab.extract()
                    tbbox = tuple(tab.bbox)
                except Exception:  # noqa: BLE001
                    continue
                for ri, row in enumerate(rows):
                    for ci, cell in enumerate(row):
                        cell_text = normalize_text(str(cell or ""))
                        if not cell_text:
                            continue
                        elements.append(
                            Element(
                                element_id=f"p{pno + 1}-table{ti}-r{ri}-c{ci}",
                                page=pno + 1,
                                bbox=tbbox,
                                text=cell_text,
                                modality="table",
                                section_path=section,
                            )
                        )
    return Document(path=path, sha256=sha, version=version, n_pages=n_pages, elements=elements)


def find_quote(doc: Document, quote: str) -> Optional[Element]:
    """Return the first element whose normalized text contains ``quote``."""
    nq = normalize_text(quote)
    if not nq:
        return None
    for el in doc.elements:
        if nq in el.text:
            return el
    return None
