# PDF2LLM-Tuning-Studio - GPU PDF Q&A extraction image
# Public image (GHCR): one image serves Azure AI Foundry / AWS Bedrock / OpenAI.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 비대화형 모드 설정
ENV DEBIAN_FRONTEND=noninteractive

# -------------------------
# 1) 시스템 패키지 설치
# -------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        # NVIDIA 관련 유틸리티
        cuda-command-line-tools-12-4 \
        # OCR / PDF 처리 의존성
        tesseract-ocr \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-dev \
        cmake && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# 2) Python 3.11 설치
# -------------------------
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# 3) get-pip.py로 Python 3.11 pip 설치
# -------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    # 심볼릭 링크 설정 (기본 python, pip → Python3.11)
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3 && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

# -------------------------
# 4) NVIDIA 환경변수 설정
# -------------------------
ENV PATH="/usr/local/cuda-12.4/bin:${PATH}"

# -------------------------
# 5) 애플리케이션 설치
#    requirements.txt(전체 런타임 의존성) 설치 후 pdf_qa 코어 패키지를 설치합니다.
#    하나의 이미지가 Azure AI Foundry / AWS Bedrock / OpenAI 공급자를 모두 지원합니다.
# -------------------------
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY pdf_qa ./pdf_qa
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps --no-build-isolation .

# -------------------------
# 6) 엔트리포인트 스크립트 복사 (로컬 / Azure ML / SageMaker 공용)
# -------------------------
COPY run_local.py processing.py azureml_job.py ./
COPY processing_local.py processing_local_openai.py ./

# -------------------------
# 컨테이너 실행 시 기본 명령 (LLM_PROVIDER 환경변수로 공급자 선택)
# -------------------------
CMD ["python3", "run_local.py"]
