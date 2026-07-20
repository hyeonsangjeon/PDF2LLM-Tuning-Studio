"""End-to-end orchestration: PDF -> elements -> provider -> JSONL.

This module is provider- and cloud-agnostic. Every entrypoint (local Docker,
Azure ML Job, SageMaker Processing Job) calls :func:`run_pipeline` with a
:class:`~pdf_qa.config.QAConfig` and a concrete
:class:`~pdf_qa.providers.base.LLMProvider`.
"""

from __future__ import annotations

import json
import os
from typing import List

from .config import QAConfig
from .device import probe_device
from .extract import (
    extract_document_layout,
    get_extracted_images,
)
from .prompts import get_persona
from .providers.base import LLMProvider


def _legacy_glob_forced() -> bool:
    return str(os.getenv("LEGACY_IMAGE_GLOB", "")).strip().lower() in {
        "1", "true", "yes", "y", "on",
    }


def generate_qa_pairs(
    pdf_path: str, provider: LLMProvider, config: QAConfig
) -> List[dict]:
    """Extract elements from ``pdf_path`` and generate Q&A pairs via ``provider``."""
    # Probe the accelerator once and reuse it for extraction (so the log shows
    # exactly which path -- GPU or CPU -- the heavy models take).
    device = probe_device()
    print(device.summary())

    # Ordered layout: section-tagged text chunks + figures paired with the
    # surrounding text that gives each chart its meaning (see pdf_qa.layout).
    layout = extract_document_layout(
        pdf_path,
        hi_res_model_name=config.table_model,
        figures_dir=config.figures_dir,
        strategy=config.strategy,
        gpu_boost=config.gpu_boost,
        device=device,
    )
    print(
        f"추출된 요소 수: {len(layout.elements)} "
        f"(텍스트 청크 {len(layout.text_chunks)}개, 도형 {len(layout.figures)}개)"
    )

    qa_pairs: List[dict] = []

    # --- text chunks (each carries its section heading) ---
    text_count = 0
    print("\n=== 텍스트 요소 처리 시작 ===")
    for chunk in layout.text_chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        try:
            response = provider.generate_text_qa(
                text, config.domain, config.num_questions, config.persona
            )
            qa_pairs.extend(response)
            text_count += 1
            print(f"텍스트 청크 {text_count} 처리 완료 - {len(response)}개 Q&A 생성")
        except Exception as exc:
            print(f"텍스트 청크 처리 에러: {exc}")
    print(f"텍스트 처리 완료: 총 {text_count}개 청크에서 {len(qa_pairs)}개 Q&A 생성")

    # --- figures (chart/image), each with its linked context ---
    print("\n=== 이미지 요소 처리 시작 ===")
    image_count = 0
    figures = [] if _legacy_glob_forced() else layout.figures
    if figures:
        print(f"문맥 연결된 도형: {len(figures)}개")
        for fig in figures:
            image_qa = provider.generate_image_qa(
                fig.image_path,
                config.domain,
                config.num_img_questions,
                config.persona,
                context=fig.context_text,
            )
            # Record the chunk<->figure linkage on every Q&A for provenance.
            for qa in image_qa:
                qa.setdefault("source", "image")
                qa["page"] = fig.page
                qa["section"] = fig.section_title
                qa["figure_index"] = fig.figure_index
                qa["context_used"] = bool(fig.context_text)
            qa_pairs.extend(image_qa)
            if image_qa:
                image_count += 1
    else:
        # Fallback: unstructured wrote crops but exposed no ``image_path`` on the
        # elements (older versions), or LEGACY_IMAGE_GLOB is set. Read images off
        # disk, context-free -- exactly the pre-linkage behaviour, so nothing
        # regresses when the ordered stream is unavailable.
        image_files = get_extracted_images(config.figures_dir)
        if image_files:
            print(f"발견된 이미지 파일(문맥 없음): {len(image_files)}개")
            for image_path in image_files:
                image_qa = provider.generate_image_qa(
                    image_path, config.domain, config.num_img_questions, config.persona
                )
                qa_pairs.extend(image_qa)
                if image_qa:
                    image_count += 1
        else:
            print("추출된 이미지가 없습니다.")
    image_qa_total = sum(1 for qa in qa_pairs if qa.get("source") == "image")
    print(f"이미지 처리 완료: 총 {image_count}개 이미지에서 {image_qa_total}개 Q&A 생성")

    return qa_pairs


def save_jsonl(qa_pairs: List[dict], output_path: str) -> None:
    """Write the Q&A pairs to ``output_path`` as UTF-8 JSON Lines."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in qa_pairs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_pipeline(
    pdf_path: str, output_path: str, provider: LLMProvider, config: QAConfig
) -> List[dict]:
    """Full run: generate Q&A pairs and persist them to ``output_path``."""
    print(f"PDF 처리 시작: {pdf_path}")
    persona = get_persona(config.persona)
    print(
        f"LLM 공급자: {provider.name} | 도메인: {config.domain} | "
        f"페르소나: {persona.label}({persona.key}) | "
        f"텍스트질문: {config.num_questions} | 이미지질문: {config.num_img_questions} | "
        f"전략: {config.strategy} | GPU부스트: {config.gpu_boost} | "
        f"테이블모델: {config.table_model or 'None'}"
    )

    qa_pairs = generate_qa_pairs(pdf_path, provider, config)
    save_jsonl(qa_pairs, output_path)

    text_n = len([qa for qa in qa_pairs if qa.get("source") != "image"])
    image_n = len([qa for qa in qa_pairs if qa.get("source") == "image"])
    print(f"\n[INFO] QA 생성 완료! 총 {len(qa_pairs)}개 Q&A가 {output_path}에 저장되었습니다.")
    print(f"- 텍스트에서 생성: {text_n}개")
    print(f"- 이미지에서 생성: {image_n}개")
    return qa_pairs
