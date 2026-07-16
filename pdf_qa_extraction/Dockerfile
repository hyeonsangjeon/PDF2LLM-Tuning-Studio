# syntax=docker/dockerfile:1
# PDF2LLM-Tuning-Studio - PDF Q&A extraction image (slim, CPU, multi-stage)
# Public image (GHCR): one image serves Azure AI Foundry / AWS Bedrock / OpenAI.
#
# Why this is small (~6GB -> ~2.5GB):
#   * This container only *parses* PDFs (unstructured layout via onnxruntime +
#     tesseract OCR) and calls the LLM over the *network* (Azure/Bedrock/OpenAI
#     APIs). There is no torch.cuda / onnxruntime-gpu code path here, so CUDA is
#     never actually used. torch is pulled in only transitively by
#     `unstructured[pdf]`, and its default PyPI wheel drags ~3.5GB of bundled
#     NVIDIA CUDA libraries (cudnn/cublas/...) that this image never touches.
#   * So we install the CPU-only torch wheel (~200MB) and drop the CUDA base
#     image entirely. The heavy GPU workload (LoRA fine-tuning) runs in a
#     separate Azure ML / SageMaker training job, not in this extraction image.
#   * Multi-stage: compilers/headers live only in the builder; the final image
#     ships a stripped, self-contained virtualenv + a handful of runtime libs.
#
# Optional GPU build: install CUDA torch on the same slim base and run with
# `--gpus all` (libcuda is injected by nvidia-container-toolkit, no CUDA base
# image required):
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 -t pdf-qa-extractor .

# ============================  Stage 1: builder  ============================
ARG PYTHON_TAG=3.10-slim-bookworm
FROM python:${PYTHON_TAG} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Compilers/headers for the few deps that lack manylinux wheels. `binutils`
# (from build-essential) provides `strip`, used below to shrink native libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"
RUN python -m venv "$VIRTUAL_ENV"

# CPU-only torch keeps the image ~3.5GB smaller than the default CUDA wheel.
# Point this at .../whl/cu124 (and run with --gpus all) for a GPU build.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY pdf_qa ./pdf_qa
RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url "${TORCH_INDEX_URL}" torch && \
    pip install -r requirements.txt && \
    pip install --no-deps --no-build-isolation . && \
    # Trim: drop cached bytecode and strip debug symbols from native libs.
    find "$VIRTUAL_ENV" -type d -name __pycache__ -prune -exec rm -rf {} + ; \
    find "$VIRTUAL_ENV" -name '*.so*' -exec strip --strip-unneeded {} + 2>/dev/null || true

# ============================  Stage 2: runtime  ============================
ARG PYTHON_TAG=3.10-slim-bookworm
FROM python:${PYTHON_TAG}

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Runtime-only shared libraries: OCR (tesseract), PDF raster (poppler), the
# OpenCV/Pillow native deps (libGL, glib, X libs), OpenMP (torch/onnxruntime)
# and libmagic (unstructured filetype detection). No compilers, no CUDA.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libmagic1 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Bring in the fully-installed environment (pdf_qa + all provider SDKs).
COPY --from=builder /opt/venv /opt/venv

# Entrypoint scripts (local / Azure ML / SageMaker share the same image).
WORKDIR /app
COPY run_local.py processing.py azureml_job.py ./
COPY processing_local.py processing_local_openai.py ./

# LLM_PROVIDER env var selects the backend (azure | bedrock | openai).
CMD ["python", "run_local.py"]
