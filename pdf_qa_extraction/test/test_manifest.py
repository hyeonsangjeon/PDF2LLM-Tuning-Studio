"""Unit tests for the run manifest + chart-linkage summaries.

Pure dict aggregation, so no heavy deps: they build synthetic Q&A pairs and
assert the chart<->context provenance is summarised correctly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pdf_qa.manifest import build_manifest, figure_linkage


def _pairs():
    return [
        {"QUESTION": "t1", "ANSWER": "a", "source": "text"},
        {"QUESTION": "t2", "ANSWER": "a", "source": "text"},
        {
            "QUESTION": "i1",
            "ANSWER": "a",
            "source": "image",
            "image_path": "/figs/fig-2.png",
            "page": 3,
            "section": "GDP growth",
            "figure_index": 2,
            "context_used": True,
        },
        {
            "QUESTION": "i2",
            "ANSWER": "a",
            "source": "image",
            "image_path": "/figs/fig-2.png",
            "page": 3,
            "section": "GDP growth",
            "figure_index": 2,
            "context_used": True,
        },
        {
            "QUESTION": "i3",
            "ANSWER": "a",
            "source": "image",
            "image_path": "/figs/fig-5.png",
            "page": 7,
            "section": "Trade",
            "figure_index": 5,
            "context_used": False,
        },
    ]


def test_figure_linkage_groups_by_figure_index():
    figs = figure_linkage(_pairs())
    assert [f["figure_index"] for f in figs] == [2, 5]
    fig2 = figs[0]
    assert fig2["questions"] == 2  # two questions collapsed onto one figure
    assert fig2["page"] == 3
    assert fig2["section"] == "GDP growth"
    assert fig2["image_path"] == "fig-2.png"  # basename only, no absolute path
    assert fig2["context_used"] is True
    assert figs[1]["context_used"] is False


def test_figure_linkage_ignores_text_pairs():
    text_only = [{"source": "text"}, {"QUESTION": "q"}]
    assert figure_linkage(text_only) == []


def test_build_manifest_counts_and_figures():
    m = build_manifest(
        _pairs(),
        {"document": "x.pdf", "persona": "professor", "provider": "ollama"},
    )
    assert m["document"] == "x.pdf"
    assert m["persona"] == "professor"
    assert m["counts"] == {
        "total": 5,
        "text": 2,
        "image": 3,
        "figures": 2,
        "figures_with_context": 1,
    }
    assert len(m["figures"]) == 2


def test_build_manifest_empty():
    m = build_manifest([])
    assert m["counts"]["total"] == 0
    assert m["figures"] == []
