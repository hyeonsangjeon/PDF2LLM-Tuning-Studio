"""FastAPI backend for the single-node local demo.

The app calls the :mod:`pdf_qa` core **in-process**. Light metadata endpoints
(``/api/personas``, ``/api/device``, ``/api/providers``) never import the heavy
PDF/OCR stack, so the page loads and the GPU/CPU badge works even in a bare
env. Only ``/api/extract`` pulls ``unstructured`` (lazily, inside the pipeline)
and -- in ``full`` mode -- a cloud LLM provider.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pdf_qa import (
    PERSONAS,
    QAConfig,
    get_persona,
    list_personas,
    probe_device,
)
from pdf_qa.prompts import DEFAULT_PERSONA, build_text_prompt

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Bundled demo document (International Finance) so a visitor can run a one-click
# offline preview with a real PDF and zero setup.
_SAMPLE_NAME = "fsi_data.pdf"
_SAMPLE_DOMAIN = "International Finance"

# Env vars that indicate a provider is likely usable (best-effort hint only).
_PROVIDER_ENV_HINTS = {
    "azure": (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_AI_PROJECT_CONNECTION_STRING",
        "PROJECT_CONNECTION_STRING",
    ),
    "openai": ("OPENAI_API_KEY",),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_DEFAULT_REGION", "AWS_REGION"),
    "ollama": ("OLLAMA_BASE_URL", "OLLAMA_HOST", "OLLAMA_MODEL"),
}

# Providers that need no cloud credentials (run locally). They are always
# selectable; the UI shows them as "local" rather than "needs credentials".
_CREDENTIAL_FREE = {"ollama"}


def _persona_payload() -> dict:
    """Persona keys + labels + a short one-line method summary for the UI."""
    personas = []
    for key in list_personas():
        p = PERSONAS[key]
        # First non-empty content line of the method block, sans the header.
        summary = ""
        for line in p.method.splitlines():
            stripped = line.strip().lstrip("-").strip()
            if stripped and not stripped.lower().startswith("method"):
                summary = stripped
                break
        personas.append({"key": key, "label": p.label, "method_summary": summary})
    return {"default": DEFAULT_PERSONA, "personas": personas}


def _device_payload() -> dict:
    report = probe_device()
    data = dataclasses.asdict(report)
    data["gpu_ready"] = report.gpu_ready
    data["summary"] = report.summary()
    return data


def _providers_payload() -> dict:
    providers = []
    for name, hints in _PROVIDER_ENV_HINTS.items():
        local = name in _CREDENTIAL_FREE
        # Local providers need no credentials, so they are always "configured".
        configured = True if local else any(os.environ.get(h) for h in hints)
        providers.append({"name": name, "configured": configured, "local": local})
    return {
        "default": os.environ.get("LLM_PROVIDER", "azure"),
        "providers": providers,
    }


def _sample_pdf_path() -> Optional[str]:
    """Absolute path to the bundled demo PDF, or ``None`` if unavailable.

    Resolves ``<app_root>/data/fsi_data.pdf`` -- which is ``pdf_qa_extraction/
    data`` in a repo checkout and ``/app/data`` in the container image -- and
    honours a ``SAMPLE_PDF`` env override.
    """
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.environ.get("SAMPLE_PDF"),
        os.path.join(app_root, "data", _SAMPLE_NAME),
        os.path.join(os.getcwd(), "data", _SAMPLE_NAME),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _meta_payload() -> dict:
    sample = _sample_pdf_path()
    return {
        "version": "0.2.0",
        "sample_available": bool(sample),
        "sample_name": _SAMPLE_NAME if sample else None,
        "sample_domain": _SAMPLE_DOMAIN,
    }


def _as_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_text_sample(elements: list, limit: int = 1200) -> str:
    for element in elements:
        text = getattr(element, "text", None)
        if text and text.strip():
            text = text.strip()
            return text if len(text) <= limit else text[:limit] + " …"
    return ""


def create_app() -> FastAPI:
    app = FastAPI(
        title="PDF2LLM Tuning Studio — Local Demo",
        description="Single-node demo: upload a PDF, pick a persona, run locally "
        "(auto GPU/CPU). The pipeline runs in-process.",
        version="0.2.0",
    )

    if os.path.isdir(_STATIC_DIR):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        index_html = os.path.join(_STATIC_DIR, "index.html")
        if not os.path.isfile(index_html):
            return JSONResponse({"detail": "UI not found"}, status_code=404)
        return FileResponse(index_html)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": "0.2.0"}

    @app.get("/api/personas")
    def api_personas():
        return _persona_payload()

    @app.get("/api/device")
    def api_device():
        return _device_payload()

    @app.get("/api/providers")
    def api_providers():
        return _providers_payload()

    @app.get("/api/meta")
    def api_meta():
        return _meta_payload()

    @app.post("/api/extract")
    def api_extract(
        file: Optional[UploadFile] = File(None),
        persona: str = Form(DEFAULT_PERSONA),
        domain: str = Form("International Finance"),
        num_questions: str = Form("3"),
        num_img_questions: str = Form("1"),
        provider: str = Form("azure"),
        strategy: str = Form("auto"),
        gpu_boost: str = Form("true"),
        table_model: str = Form(""),
        mode: str = Form("preview"),
        use_sample: str = Form("false"),
    ):
        """Run extraction (``preview``) or the full pipeline (``full``).

        ``preview`` partitions the PDF device-aware and returns element counts +
        the persona-rendered prompt, without calling any LLM (works offline).
        ``full`` additionally calls the selected provider to produce Q&A pairs.
        """
        # Validate the persona early so a typo fails fast with a clear message.
        try:
            resolved_persona = get_persona(persona)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Resolve the input document: an uploaded PDF, or the bundled sample
        # (one-click demo). An uploaded file always wins over the sample flag.
        has_upload = file is not None and bool((file.filename or "").strip())
        sample_path: Optional[str] = None
        if has_upload:
            if not (file.filename or "").lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400, detail="PDF 파일(.pdf)만 지원합니다."
                )
            input_name = file.filename or "upload.pdf"
        elif _as_bool(use_sample, default=False):
            sample_path = _sample_pdf_path()
            if not sample_path:
                raise HTTPException(
                    status_code=404,
                    detail="샘플 문서(fsi_data.pdf)를 찾을 수 없습니다. "
                    "PDF 파일을 직접 업로드하세요.",
                )
            input_name = _SAMPLE_NAME
        else:
            raise HTTPException(
                status_code=400,
                detail="PDF 파일을 업로드하거나 '샘플 문서로 시도'를 사용하세요.",
            )

        workdir = tempfile.mkdtemp(prefix="pdfqa_")
        pdf_path = os.path.join(workdir, "input.pdf")
        figures_dir = os.path.join(workdir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        try:
            if sample_path:
                shutil.copyfile(sample_path, pdf_path)
            else:
                with open(pdf_path, "wb") as handle:
                    handle.write(file.file.read())

            config = QAConfig(
                domain=domain,
                num_questions=num_questions,
                num_img_questions=num_img_questions,
                table_model=(table_model.strip() or None),
                figures_dir=figures_dir,
                persona=resolved_persona.key,
                strategy=strategy,
                gpu_boost=_as_bool(gpu_boost),
            )

            device = probe_device()

            # Heavy imports are deferred to here so the metadata endpoints and a
            # bare env never require the unstructured / provider stacks.
            try:
                from pdf_qa.extract import (
                    extract_elements_from_pdf,
                    get_extracted_images,
                    resolve_extraction_plan,
                )
            except Exception as exc:  # pragma: no cover - missing extra deps
                raise HTTPException(
                    status_code=500,
                    detail="추출 의존성(unstructured)이 설치되어 있지 않습니다. "
                    "컨테이너 이미지로 실행하거나 `pip install .`로 설치하세요. "
                    f"({exc})",
                ) from exc

            plan = resolve_extraction_plan(
                strategy=config.strategy,
                hi_res_model_name=config.table_model,
                gpu_boost=config.gpu_boost,
                device=device,
            )

            common = {
                "mode": mode,
                "input": {
                    "name": input_name,
                    "source": "sample" if sample_path else "upload",
                },
                "persona": {
                    "key": resolved_persona.key,
                    "label": resolved_persona.label,
                },
                "device": {
                    "gpu_ready": device.gpu_ready,
                    "summary": device.summary(),
                    "providers": device.onnxruntime_providers,
                    "onnxruntime_gpu_package": device.onnxruntime_gpu_package,
                    "cuda_device_name": device.cuda_device_name,
                },
                "plan": plan,
            }

            if mode == "preview":
                try:
                    elements = extract_elements_from_pdf(
                        pdf_path,
                        hi_res_model_name=config.table_model,
                        figures_dir=config.figures_dir,
                        strategy=config.strategy,
                        gpu_boost=config.gpu_boost,
                        device=device,
                    )
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"PDF 추출 중 오류: {exc}",
                    ) from exc
                sample_text = _first_text_sample(elements)
                images = get_extracted_images(config.figures_dir)
                table_n = sum(
                    1
                    for e in elements
                    if type(e).__name__.lower().startswith("table")
                )
                sample_prompt = (
                    build_text_prompt(
                        sample_text or "(본문 텍스트를 찾지 못했습니다)",
                        config.domain,
                        config.num_questions,
                        resolved_persona.key,
                    )
                )
                return {
                    **common,
                    "counts": {
                        "elements": len(elements),
                        "tables": table_n,
                        "images": len(images),
                    },
                    "sample_prompt": sample_prompt,
                }

            if mode == "full":
                from pdf_qa import generate_qa_pairs, get_provider

                try:
                    llm = get_provider(provider, config=config)
                except Exception as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{provider}' 공급자 초기화 실패 — 자격 증명(.env)을 "
                        f"확인하세요. ({exc})",
                    ) from exc

                try:
                    pairs: List[dict] = generate_qa_pairs(pdf_path, llm, config)
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Q&A 생성 중 오류: {exc}",
                    ) from exc

                jsonl = "\n".join(
                    __import__("json").dumps(item, ensure_ascii=False)
                    for item in pairs
                )
                text_n = len([q for q in pairs if q.get("source") != "image"])
                image_n = len([q for q in pairs if q.get("source") == "image"])
                return {
                    **common,
                    "counts": {"total": len(pairs), "text": text_n, "image": image_n},
                    "pairs": pairs,
                    "jsonl": jsonl,
                }

            raise HTTPException(
                status_code=400, detail=f"알 수 없는 mode: {mode} (preview|full)"
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return app


app = create_app()
