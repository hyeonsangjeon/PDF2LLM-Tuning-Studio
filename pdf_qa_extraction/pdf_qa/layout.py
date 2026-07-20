"""Ordered document layout: rebuild the chunk <-> figure linkage.

``unstructured`` returns elements in reading ``order`` and tags each one with
``page_number``, ``coordinates`` (a bounding box) and -- for extracted figures
-- ``image_path``. The old pipeline threw that away: it read ``element.text``
for the text path and, separately, *globbed the PNGs off disk and sorted them by
filename*. A chart therefore reached the vision model **alone**, with no idea of
the paragraph, caption or section it belonged to, so it could only yield shallow
"read this number" questions.

This module reconstructs the linkage from the ordered element stream:

* :func:`build_document_layout` turns the raw elements into a
  :class:`DocumentLayout` with

  - ``text_chunks``  -- consecutive text regrouped under the nearest section
    heading up to a character budget (so text Q&A keeps section cohesion), and
  - ``figures``      -- every extracted figure paired with a
    :class:`FigureContext`: its section title, the paragraph(s) immediately
    **before** it, and the caption / paragraph(s) immediately **after** it.

The figure context is assembled by reading order, restricted to the figure's own
page, and -- when bounding boxes are available -- refined by vertical proximity
so a caption sitting directly under the chart is picked even in multi-column or
slightly mis-ordered layouts.

The module is deliberately dependency-free (it only reads duck-typed attributes
via :func:`getattr`), so it can be imported and unit-tested without the heavy
``unstructured`` / OCR stack installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Element category groupings (unstructured ``category`` / class names)
# ---------------------------------------------------------------------------
TITLE_CATEGORIES = {
    "Title", "Header", "Headline", "Subtitle", "SectionHeader", "PageHeader",
}
#: Text that may seed a figure's surrounding context (and the text chunks).
CONTEXT_CATEGORIES = {
    "NarrativeText", "Text", "ListItem", "UncategorizedText", "FigureCaption",
    "Caption", "Address", "Formula", "Footnote",
}
CAPTION_CATEGORIES = {"FigureCaption", "Caption"}
#: Categories whose ``.text`` is folded into the text chunks (Table text too,
#: unless the table is being sent to vision as an image -- see ``_is_figure``).
TEXT_BODY_CATEGORIES = TITLE_CATEGORIES | CONTEXT_CATEGORIES | {"Table"}

# Defaults for the figure context window (overridable via env).
_DEF_CTX_CHARS = 800
_DEF_CTX_ELEMS = 3
_DEF_AFTER_ELEMS = 2
_DEF_MAX_CHARS = 4000
_DEF_SOFT_LIMIT = 3800


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _table_images_enabled() -> bool:
    """Send table crops to vision too? Off by default (table *text* is richer)."""
    return str(os.environ.get("EXTRACT_TABLE_IMAGES", "")).strip().lower() in {
        "1", "true", "yes", "y", "on",
    }


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class DocElement:
    """A single normalised element in document reading order."""

    index: int
    category: str
    text: str
    page: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    image_path: Optional[str] = None


@dataclass
class TextChunk:
    """A cohesive block of body text, tagged with its section heading."""

    text: str
    section_title: str = ""
    page: Optional[int] = None


@dataclass
class FigureContext:
    """An extracted figure paired with the text that gives it meaning."""

    image_path: str
    figure_index: int
    kind: str = "image"  # "image" | "table"
    page: Optional[int] = None
    section_title: str = ""
    before_text: str = ""
    after_text: str = ""
    context_text: str = ""


@dataclass
class DocumentLayout:
    """The ordered layout: text chunks + figures with their context."""

    text_chunks: List[TextChunk] = field(default_factory=list)
    figures: List[FigureContext] = field(default_factory=list)
    elements: List[DocElement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _category(element) -> str:
    cat = getattr(element, "category", None)
    if cat:
        return str(cat)
    return type(element).__name__


def _extract_bbox(metadata) -> Optional[Tuple[float, float, float, float]]:
    """Best-effort (x0, y0, x1, y1) from an element's coordinates metadata."""
    coords = getattr(metadata, "coordinates", None)
    if coords is None:
        return None
    points = getattr(coords, "points", None)
    if not points:
        # Some versions expose a dict.
        if isinstance(coords, dict):
            points = coords.get("points")
    if not points:
        return None
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _to_doc(element, index: int) -> DocElement:
    metadata = getattr(element, "metadata", None)
    page = getattr(metadata, "page_number", None) if metadata is not None else None
    image_path = getattr(metadata, "image_path", None) if metadata is not None else None
    bbox = _extract_bbox(metadata) if metadata is not None else None
    text = getattr(element, "text", "") or ""
    return DocElement(
        index=index,
        category=_category(element),
        text=text,
        page=page,
        bbox=bbox,
        image_path=image_path or None,
    )


def _is_figure(doc: DocElement) -> bool:
    """A figure is an element that has a saved crop we can send to vision."""
    if not doc.image_path:
        return False
    if doc.category.lower().startswith("table"):
        return _table_images_enabled()
    return True


# ---------------------------------------------------------------------------
# Text chunking (our own, so image order is preserved by the caller)
# ---------------------------------------------------------------------------
def _build_text_chunks(
    docs: List[DocElement], max_chars: int, soft_limit: int
) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    buf: List[str] = []
    buf_len = 0
    section = ""
    chunk_section = ""
    chunk_page: Optional[int] = None

    def flush() -> None:
        nonlocal buf, buf_len, chunk_section, chunk_page
        if buf:
            body = "\n\n".join(buf).strip()
            if body:
                chunks.append(
                    TextChunk(text=body, section_title=chunk_section, page=chunk_page)
                )
        buf, buf_len = [], 0
        chunk_page = None

    for doc in docs:
        if _is_figure(doc):
            continue
        if doc.category not in TEXT_BODY_CATEGORIES:
            continue
        text = (doc.text or "").strip()
        if not text:
            continue
        is_title = doc.category in TITLE_CATEGORIES

        # Prefer to break right before a new section once the chunk is sizeable.
        if is_title and buf and buf_len >= soft_limit:
            flush()
        if is_title:
            section = text
        # Hard cap: never exceed the character budget.
        if buf and buf_len + len(text) > max_chars:
            flush()

        if not buf:
            chunk_section = section
            chunk_page = doc.page
        buf.append(text)
        buf_len += len(text) + 2

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Figure <-> context linking
# ---------------------------------------------------------------------------
def _nearest_title(docs: List[DocElement], i: int) -> str:
    for j in range(i - 1, -1, -1):
        if docs[j].category in TITLE_CATEGORIES:
            return (docs[j].text or "").strip()
    return ""


def _same_page(a: Optional[int], b: Optional[int]) -> bool:
    """Same-page guard that is permissive when a page number is missing."""
    if a is None or b is None:
        return True
    return a == b


def _collect_before(
    docs: List[DocElement], i: int, page: Optional[int], max_chars: int, max_elems: int
) -> List[DocElement]:
    picked: List[DocElement] = []
    total = 0
    for j in range(i - 1, -1, -1):
        doc = docs[j]
        if _is_figure(doc):
            break
        if doc.category in TITLE_CATEGORIES:
            break
        if not _same_page(page, doc.page):
            break
        if doc.category in CONTEXT_CATEGORIES and (doc.text or "").strip():
            picked.append(doc)
            total += len(doc.text)
            if len(picked) >= max_elems or total >= max_chars:
                break
    picked.reverse()
    return picked


def _collect_after(
    docs: List[DocElement], i: int, page: Optional[int], max_chars: int, max_elems: int
) -> List[DocElement]:
    picked: List[DocElement] = []
    total = 0
    for j in range(i + 1, len(docs)):
        doc = docs[j]
        if _is_figure(doc):
            break
        if doc.category in TITLE_CATEGORIES:
            break
        if not _same_page(page, doc.page):
            break
        if doc.category in CONTEXT_CATEGORIES and (doc.text or "").strip():
            picked.append(doc)
            total += len(doc.text)
            if len(picked) >= max_elems or total >= max_chars:
                break
    return picked


def _vgap(fig: Optional[Tuple], other: Optional[Tuple]) -> Optional[float]:
    """Vertical gap between two bboxes (0 if they overlap vertically)."""
    if not fig or not other:
        return None
    _, fy0, _, fy1 = fig
    _, oy0, _, oy1 = other
    if oy0 >= fy1:  # other is below the figure
        return oy0 - fy1
    if oy1 <= fy0:  # other is above the figure
        return fy0 - oy1
    return 0.0


def _spatial_caption(
    docs: List[DocElement], i: int, page: Optional[int]
) -> Optional[DocElement]:
    """Nearest text box directly **below** the figure on the same page.

    A chart caption almost always sits immediately under it; reading order can
    misplace it in multi-column PDFs, so when coordinates exist we pick the
    closest below-box explicitly. Returns ``None`` if bboxes are unavailable.
    """
    fig_bbox = docs[i].bbox
    if not fig_bbox:
        return None
    best: Optional[DocElement] = None
    best_gap: Optional[float] = None
    _, _, _, fy1 = fig_bbox
    for j, doc in enumerate(docs):
        if j == i or _is_figure(doc):
            continue
        if not _same_page(page, doc.page):
            continue
        if doc.category not in CONTEXT_CATEGORIES or not (doc.text or "").strip():
            continue
        if not doc.bbox:
            continue
        if doc.bbox[1] < fy1:  # must start below the figure's bottom
            continue
        gap = doc.bbox[1] - fy1
        if best_gap is None or gap < best_gap:
            best, best_gap = doc, gap
    return best


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def _assemble_context(section: str, before: str, after: str) -> str:
    parts = []
    if section:
        parts.append(f"[Section] {section}")
    if before:
        parts.append(f"[Context before the figure]\n{before}")
    if after:
        parts.append(f"[Caption / context after the figure]\n{after}")
    return "\n\n".join(parts)


def _build_figures(
    docs: List[DocElement], ctx_chars: int, ctx_elems: int, after_elems: int
) -> List[FigureContext]:
    figures: List[FigureContext] = []
    fig_index = 0
    for i, doc in enumerate(docs):
        if not _is_figure(doc):
            continue
        fig_index += 1
        section = _nearest_title(docs, i)
        before_docs = _collect_before(docs, i, doc.page, ctx_chars, ctx_elems)
        after_docs = _collect_after(docs, i, doc.page, ctx_chars, after_elems)

        # Spatial refinement: ensure the caption directly below is present.
        caption = _spatial_caption(docs, i, doc.page)
        if caption is not None and all(a.index != caption.index for a in after_docs):
            after_docs = [caption] + after_docs

        before_text = _clip("\n".join((d.text or "").strip() for d in before_docs), ctx_chars)
        after_text = _clip("\n".join((d.text or "").strip() for d in after_docs), ctx_chars)
        context_text = _assemble_context(section, before_text, after_text)

        figures.append(
            FigureContext(
                image_path=doc.image_path or "",
                figure_index=fig_index,
                kind="table" if doc.category.lower().startswith("table") else "image",
                page=doc.page,
                section_title=section,
                before_text=before_text,
                after_text=after_text,
                context_text=context_text,
            )
        )
    return figures


def build_document_layout(
    elements,
    *,
    max_chars: Optional[int] = None,
    soft_limit: Optional[int] = None,
    ctx_max_chars: Optional[int] = None,
    ctx_max_elems: Optional[int] = None,
    after_max_elems: Optional[int] = None,
) -> DocumentLayout:
    """Turn raw ``unstructured`` elements into an ordered :class:`DocumentLayout`.

    Text is regrouped into section-tagged chunks; every extracted figure is
    paired with the surrounding text (section + before + caption/after) so the
    vision model can interpret the chart in context. Character/element budgets
    fall back to env vars (``FIGURE_CONTEXT_MAX_CHARS`` etc.) then to defaults.
    """
    max_chars = max_chars or _env_int("TEXT_CHUNK_MAX_CHARS", _DEF_MAX_CHARS)
    soft_limit = soft_limit or _env_int("TEXT_CHUNK_SOFT_LIMIT", _DEF_SOFT_LIMIT)
    ctx_max_chars = ctx_max_chars or _env_int("FIGURE_CONTEXT_MAX_CHARS", _DEF_CTX_CHARS)
    ctx_max_elems = ctx_max_elems or _env_int("FIGURE_CONTEXT_MAX_ELEMS", _DEF_CTX_ELEMS)
    after_max_elems = after_max_elems or _env_int(
        "FIGURE_CONTEXT_AFTER_ELEMS", _DEF_AFTER_ELEMS
    )

    docs = [_to_doc(el, i) for i, el in enumerate(elements)]
    return DocumentLayout(
        text_chunks=_build_text_chunks(docs, max_chars, soft_limit),
        figures=_build_figures(docs, ctx_max_chars, ctx_max_elems, after_max_elems),
        elements=docs,
    )
