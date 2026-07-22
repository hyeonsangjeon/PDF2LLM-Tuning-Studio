#!/usr/bin/env python3
"""SageMaker Processing Job entrypoint (thin wrapper over the ``pdf_qa`` core).

Kept at this path so the existing notebook (``ScriptProcessor(code="processing.py")``)
keeps working. The ``pdf_qa`` package is baked into the container image, so the
single uploaded script can import it. Defaults to AWS Bedrock (the execution
role supplies credentials), but ``--provider azure|openai`` also works.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

try:
    import pdf_qa  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_qa import QAConfig, get_provider, run_pipeline

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"


def _resolve_pdf(pdf_arg: str | None) -> str:
    """Locate the input PDF: explicit arg -> default name -> first *.pdf found."""
    if pdf_arg and os.path.isfile(pdf_arg):
        return pdf_arg
    pdf_dir = os.path.join(INPUT_DIR, "pdf")
    default = os.path.join(pdf_dir, "fsi_data.pdf")
    if os.path.isfile(default):
        return default
    matches = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"입력 PDF를 찾을 수 없습니다: {pdf_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF에서 QA 쌍을 생성하는 SageMaker Processing 스크립트"
    )
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "bedrock"),
                        help="LLM 공급자: bedrock(기본) | azure | openai")
    parser.add_argument("--domain", default="International Finance")
    parser.add_argument("--language", default=os.getenv("OUTPUT_LANGUAGE", "auto"),
                        help="출력 언어 고정: auto(원문 언어 자동 감지) | korean | english | japanese ...")
    parser.add_argument("--num_questions", default="5")
    parser.add_argument("--num_img_questions", default="1")
    parser.add_argument("--persona", default=os.getenv("PERSONA", "professor"),
                        help="Q&A 페르소나: professor(기본) | socratic | consultant | interviewer | analyst | feynman | memoirist (원장: pdf_qa/personas.yaml, PERSONA_FILE로 교체 가능)")
    parser.add_argument("--strategy", default=os.getenv("STRATEGY", "auto"),
                        help="추출 전략: auto(기본) | fast | hi_res | ocr_only. GPU 감지 시 auto는 hi_res로 승격")
    parser.add_argument("--gpu_boost", default=os.getenv("GPU_BOOST", "true"),
                        help="GPU 감지 시 레이아웃+표 모델을 GPU로 가속 (true/false)")
    parser.add_argument("--model_id", default=None,
                        help="공급자별 모델/디플로이먼트 ID (미지정 시 공급자 기본값)")
    parser.add_argument("--table_model", default=os.getenv("TABLE_MODEL", ""),
                        help="hi_res 레이아웃 모델 선택(unstructured의 hi_res_model_name): "
                             "yolox(기본) | yolox_quantized | detectron2_onnx ... "
                             "지정하면 strategy=auto가 hi_res로 승격돼 실제 적용됩니다. "
                             "비우면 auto/GPU 부스트가 결정")
    parser.add_argument("--pdf", default=None, help="입력 PDF 경로(선택)")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "qa_pairs.jsonl"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    config = QAConfig.from_args(args)
    provider = get_provider(args.provider, config=config)
    pdf_path = _resolve_pdf(args.pdf)
    run_pipeline(pdf_path, args.output, provider, config)


if __name__ == "__main__":
    main()
