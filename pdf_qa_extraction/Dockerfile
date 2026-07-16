# syntax=docker/dockerfile:1
# PDF2LLM-Tuning-Studio - GPU PDF Q&A extraction image (lightweight, multi-stage)
# Public image (GHCR): one image serves Azure AI Foundry / AWS Bedrock / OpenAI.
#
# Why this is small:
#   * PyTorch's pip wheels bundle their own CUDA libraries (cublas / cudnn /
#     nvrtc / cuda-runtime), so the heavy `cuda:runtime` layer and the
#     `cuda-command-line-tools` dev package are redundant. We start from the
#     tiny `cuda:base` image (just the NVIDIA container-runtime hooks) and let
#     pip pull exactly the CUDA libs torch needs. `libcuda.so` (the driver) is
#     injected at run time by `--gpus all` / nvidia-container-toolkit.
#   * Multi-stage: compilers and headers live only in the builder; the final
#     image ships a self-contained virtualenv + a handful of runtime libraries.
#
# If a GPU path ever needs the full CUDA runtime, rebuild with:
#   docker build --build-arg CUDA_TAG=12.4.1-runtime-ubuntu22.04 -t pdf-qa-extractor .

# ============================  Stage 1: builder  ============================
ARG CUDA_TAG=12.4.1-base-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build toolchain + Python (Ubuntu 22.04 ships Python 3.10; no PPA needed).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# A self-contained venv is copied wholesale into the runtime stage.
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"
RUN python3 -m venv "$VIRTUAL_ENV"

WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY pdf_qa ./pdf_qa
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install --no-deps --no-build-isolation . && \
    # Strip cached bytecode that ships inside site-packages (regenerated lazily).
    find "$VIRTUAL_ENV" -type d -name __pycache__ -prune -exec rm -rf {} + || true

# ============================  Stage 2: runtime  ============================
ARG CUDA_TAG=12.4.1-base-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

# Runtime-only shared libraries: OCR (tesseract), PDF raster (poppler) and the
# OpenCV/Pillow native deps (libGL, glib, X libs). No compilers, no headers,
# no CUDA dev tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
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
