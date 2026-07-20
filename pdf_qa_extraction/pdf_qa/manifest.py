"""Run manifest + chart-linkage summaries (pure, dependency-free).

These helpers turn a list of Q&A pairs into (a) a per-figure linkage table that
surfaces the chart<->context provenance the pipeline now preserves, and (b) a
compact run manifest. They import nothing heavy, so both the web app and the
batch runner (``run_auto.py``) can share them without pulling in FastAPI or
unstructured.
"""

from __future__ import annotations

import os
from typing import List, Optional


def figure_linkage(pairs: List[dict]) -> List[dict]:
    """Aggregate image-derived Q&A by ``figure_index``.

    Each row records where a chart sat in the document (page, section), whether
    the surrounding text context was fed to the vision model, and how many
    questions it produced.
    """
    figures: dict = {}
    for item in pairs:
        if item.get("source") != "image":
            continue
        idx = item.get("figure_index")
        key = idx if idx is not None else item.get("image_path", "?")
        row = figures.get(key)
        if row is None:
            row = {
                "figure_index": idx,
                "page": item.get("page"),
                "section": item.get("section"),
                "image_path": os.path.basename(item.get("image_path") or "") or None,
                "context_used": bool(item.get("context_used")),
                "questions": 0,
            }
            figures[key] = row
        row["questions"] += 1
        row["context_used"] = row["context_used"] or bool(item.get("context_used"))
    return sorted(
        figures.values(),
        key=lambda r: (r["figure_index"] is None, r["figure_index"] or 0),
    )


def build_manifest(pairs: List[dict], meta: Optional[dict] = None) -> dict:
    """A compact, serialisable run manifest (counts + per-figure linkage)."""
    meta = meta or {}
    text_n = len([q for q in pairs if q.get("source") != "image"])
    image_n = len([q for q in pairs if q.get("source") == "image"])
    figures = figure_linkage(pairs)
    return {
        "document": meta.get("document"),
        "persona": meta.get("persona"),
        "provider": meta.get("provider"),
        "domain": meta.get("domain"),
        "device": meta.get("device"),
        "counts": {
            "total": len(pairs),
            "text": text_n,
            "image": image_n,
            "figures": len(figures),
            "figures_with_context": len([f for f in figures if f["context_used"]]),
        },
        "figures": figures,
    }
