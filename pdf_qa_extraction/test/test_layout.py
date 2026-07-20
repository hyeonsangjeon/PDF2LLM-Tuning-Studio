"""Unit tests for the ordered document layout + figure<->context linking.

Dependency-light: builds synthetic duck-typed elements (no unstructured), so it
runs in a bare venv and on the CPU image.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from pdf_qa.layout import (
    DocumentLayout,
    FigureContext,
    TextChunk,
    build_document_layout,
)


# --- synthetic unstructured-like elements ---------------------------------
class _Coords:
    def __init__(self, points):
        self.points = points


class _Meta:
    def __init__(self, page=None, image_path=None, points=None):
        self.page_number = page
        self.image_path = image_path
        self.coordinates = _Coords(points) if points else None


class _El:
    def __init__(self, category, text="", page=1, image_path=None, points=None):
        self.category = category
        self.text = text
        self.metadata = _Meta(page, image_path, points)


def _img(path="fig-1.png", page=1, points=None):
    return _El("Image", text="", page=page, image_path=path, points=points)


# --- figure linking --------------------------------------------------------
def test_figure_is_linked_to_section_before_and_after():
    els = [
        _El("Title", "환율 동향"),
        _El("NarrativeText", "원달러 환율이 상승했다."),
        _img("chart.png"),
        _El("FigureCaption", "그림 1. 원달러 환율 추이"),
        _El("Title", "다음 섹션"),
        _El("NarrativeText", "관련 없는 다음 문단."),
    ]
    layout = build_document_layout(els)
    assert len(layout.figures) == 1
    fig = layout.figures[0]
    assert fig.image_path == "chart.png"
    assert fig.figure_index == 1
    assert fig.section_title == "환율 동향"
    assert "원달러 환율이 상승했다." in fig.before_text
    assert "그림 1" in fig.after_text
    # The following section's paragraph must NOT leak into this figure.
    assert "관련 없는" not in fig.context_text
    assert "[Section] 환율 동향" in fig.context_text


def test_before_stops_at_title_and_respects_page():
    els = [
        _El("NarrativeText", "이전 페이지 문단", page=1),
        _El("Title", "섹션 A", page=2),
        _El("NarrativeText", "같은 페이지 문단", page=2),
        _img("c.png", page=2),
    ]
    fig = build_document_layout(els).figures[0]
    assert "같은 페이지 문단" in fig.before_text
    assert "이전 페이지 문단" not in fig.before_text  # different page + title barrier


def test_multiple_figures_increment_index():
    els = [
        _El("NarrativeText", "문단1"),
        _img("a.png"),
        _El("NarrativeText", "문단2"),
        _img("b.png"),
    ]
    figs = build_document_layout(els).figures
    assert [f.image_path for f in figs] == ["a.png", "b.png"]
    assert [f.figure_index for f in figs] == [1, 2]


# --- fallback: no image_path ----------------------------------------------
def test_image_without_path_is_not_a_figure():
    els = [_El("NarrativeText", "문단"), _El("Image", "", image_path=None)]
    layout = build_document_layout(els)
    assert layout.figures == []


# --- tables default to text, not vision -----------------------------------
def test_table_with_crop_is_text_by_default(monkeypatch):
    monkeypatch.delenv("EXTRACT_TABLE_IMAGES", raising=False)
    els = [_El("Table", "행1 열1", image_path="t.png")]
    layout = build_document_layout(els)
    assert layout.figures == []  # not sent to vision
    assert any("행1 열1" in c.text for c in layout.text_chunks)


def test_table_becomes_figure_when_enabled(monkeypatch):
    monkeypatch.setenv("EXTRACT_TABLE_IMAGES", "1")
    els = [_El("Table", "행1 열1", image_path="t.png")]
    layout = build_document_layout(els)
    assert len(layout.figures) == 1
    assert layout.figures[0].kind == "table"


# --- text chunking ---------------------------------------------------------
def test_text_chunks_carry_section_and_exclude_figures():
    els = [
        _El("Title", "제목1"),
        _El("NarrativeText", "본문 A"),
        _img("x.png"),
        _El("NarrativeText", "본문 B"),
    ]
    chunks = build_document_layout(els).text_chunks
    assert chunks and all(isinstance(c, TextChunk) for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    assert "본문 A" in joined and "본문 B" in joined
    assert "x.png" not in joined
    assert chunks[0].section_title == "제목1"


def test_hard_char_budget_splits_chunks():
    big = "가" * 3000
    els = [_El("NarrativeText", big), _El("NarrativeText", big)]
    chunks = build_document_layout(els, max_chars=4000).text_chunks
    assert len(chunks) == 2  # 3000 + 3000 exceeds 4000 -> split


# --- spatial caption tie-break --------------------------------------------
def test_spatial_caption_picks_box_below_figure():
    # Reading order puts an unrelated paragraph right after the figure, but the
    # real caption (by coordinates) sits directly below it -> must be included.
    els = [
        _img("chart.png", page=1, points=[(0, 0), (100, 100)]),
        # a box far below (not the caption)
        _El("NarrativeText", "먼 문단", page=1, points=[(0, 500), (100, 540)]),
        # the caption directly under the chart
        _El("FigureCaption", "그림. 캡션", page=1, points=[(0, 105), (100, 125)]),
    ]
    fig = build_document_layout(els).figures[0]
    assert "그림. 캡션" in fig.after_text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
