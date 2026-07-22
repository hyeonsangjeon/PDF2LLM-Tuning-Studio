"""Chart-context wiring tests (workstreams A + B), dependency-free.

Two levels:
* the prompt injection point (``build_image_instruction``) every provider shares;
* the pipeline threading the linked context into ``generate_image_qa`` and
  tagging each image Q&A with its chunk<->figure provenance.
Both use synthetic layout objects + a fake provider, so no unstructured/LLM.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pdf_qa.pipeline as pipeline
from pdf_qa.config import QAConfig
from pdf_qa.layout import DocumentLayout, FigureContext, TextChunk
from pdf_qa.prompts import build_image_instruction


# --- prompt-level: the shared context injection point ---------------------

def test_build_image_instruction_injects_context():
    ctx = "[Section] GDP growth\n[After] Figure 2 shows quarterly GDP."
    out = build_image_instruction("International Finance", "2", "professor", context=ctx)
    assert "FIGURE CONTEXT" in out
    assert "quarterly GDP" in out
    # the strict "numbers come from the image" rule must survive.
    assert "read from the image" in out.lower()


def test_build_image_instruction_empty_context_is_noop():
    assert "FIGURE CONTEXT" not in build_image_instruction("d", "1", "professor")
    assert "FIGURE CONTEXT" not in build_image_instruction(
        "d", "1", "professor", context="   \n  "
    )


# --- pipeline-level: context threading + linkage metadata -----------------

class _FakeProvider:
    name = "fake"

    def __init__(self):
        self.image_calls = []
        self.text_calls = []

    def generate_text_qa(self, text, domain, num, persona, language="auto"):
        self.text_calls.append({"text": text, "language": language})
        return [{"QUESTION": "tq", "ANSWER": "ta"}]

    def generate_image_qa(
        self, image_path, domain, num, persona, context="", language="auto"
    ):
        self.image_calls.append(
            {"image_path": image_path, "context": context, "language": language}
        )
        return [{"QUESTION": "iq", "ANSWER": "ia"}]


def test_pipeline_threads_context_and_tags_linkage(monkeypatch):
    monkeypatch.delenv("LEGACY_IMAGE_GLOB", raising=False)
    layout = DocumentLayout(
        text_chunks=[
            TextChunk(text="Narrative about GDP.", section_title="Overview", page=1)
        ],
        figures=[
            FigureContext(
                image_path="/figs/fig-1.png",
                figure_index=1,
                page=2,
                section_title="GDP",
                before_text="GDP rose in Q3.",
                after_text="Figure 1: quarterly GDP.",
                context_text="[Section] GDP\n[Before] GDP rose in Q3.\n"
                "[After] Figure 1: quarterly GDP.",
            )
        ],
        elements=[],
    )
    monkeypatch.setattr(pipeline, "extract_document_layout", lambda *a, **k: layout)

    provider = _FakeProvider()
    cfg = QAConfig(
        domain="International Finance",
        num_questions="1",
        num_img_questions="1",
        persona="professor",
        language="english",
    )
    pairs = pipeline.generate_qa_pairs("x.pdf", provider, cfg)

    # The figure's linked context actually reached the vision provider.
    assert provider.image_calls and provider.image_calls[0]["context"].startswith(
        "[Section] GDP"
    )
    # The configured output language was threaded to both text and image calls.
    assert provider.text_calls[0]["language"] == "english"
    assert provider.image_calls[0]["language"] == "english"

    image_pairs = [p for p in pairs if p.get("source") == "image"]
    assert len(image_pairs) == 1
    tagged = image_pairs[0]
    assert tagged["page"] == 2
    assert tagged["section"] == "GDP"
    assert tagged["figure_index"] == 1
    assert tagged["context_used"] is True
    # Text Q&A are still produced from the section-tagged chunks.
    assert any(p.get("source") != "image" for p in pairs)


def test_pipeline_context_used_false_when_no_context(monkeypatch):
    monkeypatch.delenv("LEGACY_IMAGE_GLOB", raising=False)
    layout = DocumentLayout(
        text_chunks=[],
        figures=[
            FigureContext(image_path="/figs/lone.png", figure_index=0, page=1)
        ],
        elements=[],
    )
    monkeypatch.setattr(pipeline, "extract_document_layout", lambda *a, **k: layout)
    provider = _FakeProvider()
    cfg = QAConfig(domain="d", num_questions="1", num_img_questions="1")
    pairs = pipeline.generate_qa_pairs("x.pdf", provider, cfg)
    img = [p for p in pairs if p.get("source") == "image"][0]
    assert img["context_used"] is False
    assert provider.image_calls[0]["context"] == ""
