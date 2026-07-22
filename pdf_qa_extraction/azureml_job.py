#!/usr/bin/env python3
"""Azure ML command-job entrypoint for PDF -> Q&A extraction.

Reads a PDF from ``--input-dir`` (an AML input mount / datastore path) and
writes ``qa_pairs.jsonl`` into ``--output-dir`` (an AML output mount). Defaults
to the Azure AI Foundry provider. Example (see ``azure/azureml_job.yml``)::

    python azureml_job.py --input-dir ${{inputs.pdf}} --output-dir ${{outputs.qa}}
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


def _resolve_pdf(input_dir: str, pdf_name: str | None) -> str:
    """Accept either a directly-mounted file or a directory of PDFs."""
    if os.path.isfile(input_dir):
        return input_dir
    if pdf_name:
        candidate = os.path.join(input_dir, pdf_name)
        if os.path.isfile(candidate):
            return candidate
    matches = sorted(glob.glob(os.path.join(input_dir, "**", "*.pdf"), recursive=True))
    if not matches:
        raise FileNotFoundError(f"입력 PDF를 찾을 수 없습니다: {input_dir}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure ML PDF->QA 배치 잡")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "azure"),
                        help="LLM 공급자: azure(기본) | bedrock | openai")
    parser.add_argument("--input-dir", dest="input_dir", default="data")
    parser.add_argument("--output-dir", dest="output_dir", default="outputs")
    parser.add_argument("--pdf-name", dest="pdf_name", default=None)
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
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--table_model", default=os.getenv("TABLE_MODEL", ""),
                        help="hi_res 레이아웃 모델 선택(unstructured의 hi_res_model_name): "
                             "yolox | detectron2_onnx ... 지정 시 auto→hi_res 승격")
    args = parser.parse_args()

    config = QAConfig.from_args(args)
    provider = get_provider(args.provider, config=config)
    pdf_path = _resolve_pdf(args.input_dir, args.pdf_name)
    output_path = os.path.join(args.output_dir, "qa_pairs.jsonl")
    run_pipeline(pdf_path, output_path, provider, config)


if __name__ == "__main__":
    main()
