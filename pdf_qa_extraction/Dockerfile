# syntax=docker/dockerfile:1
# PDF2LLM-Tuning-Studio - PDF Q&A extraction image (slim, CPU, multi-stage)
# Public image (GHCR): one image serves Azure AI Foundry / AWS Bedrock / OpenAI.
#
# Why this is small (~6GB -> ~2GB):
#   * This DEFAULT build is CPU-only. The extraction *code* never calls CUDA
#     directly, so it works fine on CPU: unstructured parses PDFs (hi_res layout
#     via a PyTorch/ONNX model + tesseract OCR) and the LLM runs over the
#     *network* (Azure/Bedrock/OpenAI APIs). torch is only a transitive dep of
#     `unstructured[pdf]`, yet its default PyPI wheel drags ~3.5GB of bundled
#     NVIDIA CUDA libraries (cudnn/cublas/...). We install the CPU torch wheel
#     (~200MB) instead and drop the CUDA base image.
#   * NOTE on GPU: the DEFAULT (CPU) build runs everything on CPU. The GPU build
#     (:latest-gpu, or `--build-arg TORCH_INDEX_URL=.../cu124 ONNXRUNTIME_PACKAGE=onnxruntime-gpu`)
#     accelerates BOTH model families:
#       - Layout detection (YOLOX / detectron2_onnx) runs on onnxruntime; the
#         GPU build swaps CPU onnxruntime for onnxruntime-gpu -> CUDA provider.
#       - Table structure (Table Transformer) is PyTorch and auto-uses the GPU
#         via torch.cuda.is_available() when CUDA torch is present.
#     Tesseract OCR and pdfminer (fast) are always CPU. See README "PDF parsing
#     models - GPU/CPU". The heavy LoRA fine-tuning still runs in a separate
#     Azure ML / SageMaker training job.
#   * Multi-stage: compilers/headers live only in the builder; the final image
#     ships a self-contained virtualenv + a handful of runtime libs.
#
# Optional GPU build: CUDA torch + onnxruntime-gpu on the same slim base, run
# with `--gpus all` (libcuda is injected by nvidia-container-toolkit, no CUDA
# base image required):
#   docker build \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#     --build-arg ONNXRUNTIME_PACKAGE=onnxruntime-gpu \
#     -t pdf-qa-extractor-gpu .

# ============================  Stage 1: builder  ============================
ARG PYTHON_TAG=3.10-slim-bookworm
FROM python:${PYTHON_TAG} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Compilers/headers for the few deps that lack prebuilt manylinux wheels.
# Builder-stage only, so they never reach the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"
RUN python -m venv "$VIRTUAL_ENV"

# CPU-only torch keeps the image ~3.5GB smaller than the default CUDA wheel.
# Point this at .../whl/cu124 (and run with --gpus all) for a GPU build.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# `unstructured-inference` pulls the CPU `onnxruntime` (its ONNX layout models
# run on it). The GPU build overrides this with `onnxruntime-gpu` so those
# layout models (YOLOX / detectron2_onnx) use the CUDAExecutionProvider too.
ARG ONNXRUNTIME_PACKAGE=onnxruntime

WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY pdf_qa ./pdf_qa
RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url "${TORCH_INDEX_URL}" torch && \
    pip install -r requirements.txt && \
    pip install --no-deps --no-build-isolation . && \
    # GPU build: swap the CPU onnxruntime (pulled transitively above) for the
    # requested GPU package so the ONNX layout models run on CUDA.
    if [ "${ONNXRUNTIME_PACKAGE}" != "onnxruntime" ]; then \
        pip uninstall -y onnxruntime && \
        pip install "${ONNXRUNTIME_PACKAGE}"; \
    fi && \
    # Trim cached bytecode (regenerated lazily at runtime). NB: do NOT `strip`
    # site-packages *.so - it corrupts numpy/scipy's bundled OpenBLAS
    # ("ELF load command not page-aligned"); torch/onnxruntime ship pre-stripped.
    find "$VIRTUAL_ENV" -type d -name __pycache__ -prune -exec rm -rf {} + || true

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

# Make the venv's bundled NVIDIA CUDA/cuDNN libs (shipped inside CUDA torch and,
# for the GPU build, onnxruntime-gpu's nvidia-* deps) resolvable by the dynamic
# loader, so onnxruntime-gpu can load its CUDAExecutionProvider at runtime. The
# NVIDIA driver (libcuda) itself is injected by `--gpus all` at run time. No-op
# for the CPU image (it has no nvidia/* libs).
RUN find /opt/venv -type d \( -path '*/nvidia/*/lib' -o -path '*/torch/lib' \) \
        > /etc/ld.so.conf.d/venv-cuda.conf 2>/dev/null || true; \
    ldconfig

# Entrypoint scripts (local / Azure ML / SageMaker share the same image).
WORKDIR /app
COPY run_local.py processing.py azureml_job.py ./
COPY processing_local.py processing_local_openai.py ./
# Single-node local demo web app (run: `python run_webapp.py`, serves :8000).
COPY run_webapp.py ./
COPY webapp ./webapp
# Bundled sample document so the demo web app's one-click "try sample" works.
COPY data/fsi_data.pdf ./data/fsi_data.pdf

# The demo web app listens here; harmless metadata for the batch entrypoints.
EXPOSE 8000

# LLM_PROVIDER env var selects the backend (azure | bedrock | openai).
# Default CMD stays the batch run; override with `python run_webapp.py` for the UI.
CMD ["python", "run_local.py"]
