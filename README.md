# PDF2LLM-Tuning-Studio

PDF 문서에서 지식을 추출하고 대규모 언어 모델(LLM)을 효율적으로 파인튜닝하는 **Azure-first 멀티클라우드** 엔드투엔드 파이프라인입니다. 경량 CPU 컨테이너의 PDF 파싱과 최적화된 LLM 파인튜닝을 결합하여 문서 기반 질의응답 시스템을 구축합니다.

기본 백엔드는 **Azure AI Foundry**(Azure OpenAI / Foundry Agent Service)와 **Azure Machine Learning**이며, **AWS Bedrock + SageMaker** 경로도 그대로 지원합니다. 하나의 공개 컨테이너 이미지(GHCR)를 두 클라우드가 함께 사용합니다.

- 금융보안원 AI 개발 강의, 2025

## 📚 주요 기능

- **경량 PDF 추출**: Unstructured 라이브러리로 텍스트/표/이미지 추출 (CPU 슬림 이미지 `:latest` ~2GB, 레이아웃+표를 GPU로 가속하는 `:latest-gpu` ~8GB는 선택)
- **Q&A 자동 생성 (멀티 공급자)**: Azure AI Foundry(Azure OpenAI · Foundry Agent), OpenAI, Amazon Bedrock Claude, **로컬 Ollama**(자격 증명 불필요) 중 환경 변수 하나로 전환. Azure는 **Entra ID 키리스 인증으로 한 번에 기동**(키 없이 `DefaultAzureCredential`). Ollama는 텍스트·비전 모델을 **분리 지정**(`OLLAMA_VISION_MODEL`로 `qwen2.5vl`·`minicpm-v`·`llama3.2-vision` 등 멀티모달 태그)
- **출력 언어 고정(드리프트 방지)**: `OUTPUT_LANGUAGE`로 생성 Q&A 언어를 고정 — 기본 `auto`(원문 언어 자동 감지)로 영문 PDF→한국어 답변 같은 드리프트를 차단하고, `korean`/`english`/`japanese` 등으로 강제도 가능. 프롬프트에 언어 락 지시가 주입됨
- **데이터 품질 관리(검증·중복 제거)**: 데이터셋 저장 직전 빈 항목·너무 짧은 응답·질문=답변·모델 거부("제공된 정보로는 알 수 없…")를 제거하고, 정규화 기반 **정확/유사 중복 질문**을 제거. `VALIDATE_QA`·`DEDUP_QA`·`DEDUP_SIMILARITY` 등으로 조절하며 제거 사유 통계를 리포트(웹앱·manifest에 노출)
- **다중 페르소나**: `PERSONA`로 교수·소크라테스식 튜터·실무 컨설턴트·기술 면접관·리서치 분석가·파인만(쉬운 설명)·자서전 저자(1인칭 회고) 등 **서로 다른 방식**을 전환해 하나의 PDF로 여러 파인튜닝 데이터셋 생성. 페르소나는 `pdf_qa/personas.yaml` 원장에서 관리하며 `PERSONA_FILE`로 외부 파일 지정 가능
- **GPU 자동 가속(디바이스 인지)**: 실행 시 GPU를 점검해 감지되면 `hi_res` 레이아웃(onnxruntime-gpu)+표 구조(CUDA torch)를 자동으로 GPU에 태우고, CPU에서는 경량 경로 유지
- **클라우드 무관 코어 패키지**: `pdf_qa` 패키지 + 공급자 플러그인 구조로 코드 중복 제거, 런타임별 얇은 진입점
- **단일 공개 이미지**: `ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor` 하나로 로컬·Azure ML·SageMaker 실행
- **로컬 데모 웹앱(싱글 노드)**: 같은 이미지에서 `python run_webapp.py` → 브라우저에서 문서 업로드·페르소나 선택. 파이프라인을 **같은 프로세스에서 직접 호출(in-process)** 해 GPU 유무를 자동 감지하며, 미리보기 모드는 클라우드 자격 증명 없이 오프라인으로 동작(병렬 팬아웃은 advanced)
- **아주 쉬운 사용**: 한 줄 함수 API `extract_qa("report.pdf", out="qa.jsonl", provider="ollama")` + 모든 환경변수를 문서화한 원장 `pdf_qa/settings.yaml`(→ `.env.example` 생성/검증: `python -m pdf_qa.settings --check`)
- **차트-문맥 추출 파이프라인**: 차트/도표 PNG를 단독으로 넘기지 않고 **읽기 순서·좌표·이미지 경로**로 각 도표를 주변 절·문단·캡션과 연결해 비전 프롬프트에 문맥을 주입(숫자·라벨은 이미지에서만). 각 이미지 Q&A에 `page·section·figure_index·context_used` 출처 기록
- **폴더 일괄 처리 & Compose**: 레포 루트 `docker-compose.yml`로 웹앱/배치를 CPU·GPU 한 번에 구동, `run_auto.py`가 `INPUT_DIR`→`OUTPUT_DIR`로 파일별 JSONL + `all.qa.jsonl` + `manifest.json`(차트 연결 포함) 생성. 웹앱은 `<문서>.qa.jsonl`·`manifest.json` **서버 다운로드** 제공
- **메모리 효율적 파인튜닝**: Unsloth 최적화와 LoRA 어댑터로 제한된 GPU 환경에서도 대형 모델 학습

## 🏗️ 아키텍처

```
                        PDF 문서
                           │
             ┌─────────────▼──────────────┐
             │  pdf_qa 코어 패키지 (파싱·프롬프트·파이프라인)  │
             │  providers: azure · openai · bedrock · ollama │
             └─────────────┬──────────────┘
                           │  빌드/배포
        ghcr.io/.../pdf-qa-extractor  (공개 경량 컨테이너 이미지 1개)
                 │ pull                              │ pull
        ┌────────▼──── Azure (기본) ──┐     ┌────────▼──── AWS (also) ──┐
        │ Azure ML Command Job        │     │ SageMaker Processing Job  │
        │  └▶ Azure AI Foundry        │     │  └▶ Bedrock Claude        │
        │ Blob · Key Vault · Entra ID │     │ S3 · SSM · IAM            │
        └──────────────┬──────────────┘     └─────────────┬─────────────┘
                       └──────────────┬───────────────────┘
                                      ▼
                          qa_pairs.jsonl (Q&A 데이터셋)
                                      ▼
                     fine_tuning/ (Unsloth + LoRA 파인튜닝)
```

동일한 코드·이미지에서 `LLM_PROVIDER` 환경 변수만 바꾸면 백엔드가 교체됩니다.

## 🔍 프로젝트 구조

```
PDF2LLM-Tuning-Studio/
├── .github/workflows/
│   └── build-and-push.yml   # GHCR 컨테이너 자동 빌드/푸시 (공개 이미지)
│
├── azure/                   # ☁️ Azure ML + AI Foundry 자산 (Azure-first)
│   ├── azureml_job.yml               # Azure ML Command Job 스펙
│   ├── azureml_pdf_qa_extraction.ipynb   # Azure ML 배치 잡 제출 데모
│   ├── foundry_agent_quickstart.ipynb    # Foundry Agent Service 데모
│   └── README.md                     # Azure 설정/실행 가이드
│
├── assets/                  # 공통 리소스
│   ├── images/              # 다이어그램 및 이미지
│   └── utils/               # iam.py(AWS) · ssm.py(AWS) · keyvault.py(Azure)
│
├── pdf_qa_extraction/       # PDF 처리 및 Q&A 추출 모듈
│   ├── pdf_qa/              # ⭐ 클라우드 무관 코어 패키지
│   │   ├── config.py · parsing.py · prompts.py · extract.py · pipeline.py · device.py
│   │   ├── api.py           # 🟢 한 줄 함수 API (extract_qa)
│   │   ├── layout.py        # 📊 정렬된 레이아웃 + 차트↔문맥 연결
│   │   ├── manifest.py      # 실행 매니페스트 + 도표 연결 요약
│   │   ├── settings.py · settings.yaml   # 🗒️ 환경변수 원장 (.env 생성/검증)
│   │   ├── personas.yaml    # 🎭 페르소나 원장 (YAML, PERSONA_FILE로 외부 지정 가능)
│   │   └── providers/       # azure_foundry · openai · bedrock · ollama (+ base 팩토리)
│   ├── webapp/              # 🖥️ 로컬 데모 웹앱 (FastAPI, in-process 파이프라인)
│   │   ├── app.py           # /api/{personas,device,providers,settings,extract,download}
│   │   └── static/index.html    # 싱글 페이지 UI (GPU/CPU 배지, 페르소나, 차트 연결·다운로드)
│   ├── run_local.py         # 로컬/컨테이너 통합 진입점 (LLM_PROVIDER로 전환)
│   ├── run_auto.py          # 📁 폴더 일괄 처리 (INPUT_DIR→OUTPUT_DIR, JSONL+manifest)
│   ├── run_webapp.py        # 로컬 데모 웹앱 진입점 (uvicorn, 기본 :8000)
│   ├── azureml_job.py       # Azure ML 잡 진입점
│   ├── processing.py        # SageMaker Processing Job 진입점
│   ├── processing_local*.py # (하위호환) Bedrock/OpenAI 로컬 실행 shim
│   ├── pyproject.toml · requirements.txt
│   ├── Dockerfile · Dockerfile_event_eng
│   └── README.md · README_en.md
│
├── docker-compose.yml       # 🐳 웹앱/배치 원커맨드 구동 (CPU·GPU 프로필)
├── .env.example             # 🗒️ settings.yaml 원장에서 생성된 환경변수 예시
│
└── fine_tuning/             # LLM 파인튜닝 모듈
    ├── 01_setup_environment.ipynb
    ├── 02_data_preprocessing_and_analysis.ipynb  # Blob/S3/로컬 데이터 로딩
    ├── 03_train_unsloth_model.ipynb
    └── README.md
```

## 🚀 시작 가이드

### ⚡ 가장 쉬운 방법 (한 줄 / 원커맨드)

```python
# 파이썬 한 줄: PDF → Q&A JSONL (GPU/CPU·페르소나·차트 문맥 자동)
from pdf_qa import extract_qa
extract_qa("report.pdf", out="qa_pairs.jsonl", provider="ollama", persona="feynman")
```

```bash
# 또는 Docker Compose 원커맨드 (레포 루트)
cp .env.example .env                       # 공급자 자격 증명 입력 (Ollama는 불필요)
docker compose up webapp                   # 데모 웹앱 → http://localhost:8000
mkdir -p data/input && cp *.pdf data/input/
docker compose run --rm batch              # 폴더 일괄 처리 → data/output/*.qa.jsonl + manifest.json
# GPU: docker compose --profile gpu up webapp-gpu  /  --profile gpu run --rm batch-gpu
```

> 환경변수는 `pdf_qa/settings.yaml` 원장에서 관리합니다 — `python -m pdf_qa.settings --check azure`로 검증, `--write-env`로 `.env.example` 재생성.

### 0단계: 컨테이너 이미지 준비

공개 GHCR 이미지를 그대로 받거나 직접 빌드합니다.

```bash
# 공개 이미지 pull (권장)
docker pull ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest

# 또는 직접 빌드
cd pdf_qa_extraction
docker build -t pdf-qa-extractor -f Dockerfile .
```

### 1단계: PDF 텍스트 및 Q&A 추출

경량 컨테이너에서 PDF의 텍스트/표/이미지를 추출하고 고품질 Q&A 쌍을 생성합니다. 공급자는 `LLM_PROVIDER`로 선택합니다(기본 `azure`).

#### ☁️ Azure (기본 경로)

```bash
# Azure OpenAI (Foundry 모델 배포) 사용
docker run --rm -v $(pwd):/app -w /app --env-file .env \
  ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest \
  python run_local.py
```

`.env` 예시(Azure OpenAI):
```bash
LLM_PROVIDER=azure
AZURE_MODE=openai
AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-21
# 키 대신 Managed Identity/az login 사용 시 API_KEY 생략 → Entra ID로 한 번에 기동
PDF_PATH=data/fsi_data.pdf
DOMAIN=International Finance
NUM_QUESTIONS=5
NUM_IMG_QUESTIONS=1
# hi_res 레이아웃 모델 선택(=unstructured의 hi_res_model_name): yolox(기본)·detectron2_onnx 등
TABLE_MODEL=yolox
```

- **Azure ML 배치 잡**으로 실행: [`azure/README.md`](./azure/README.md), `azure/azureml_job.yml`, `azure/azureml_pdf_qa_extraction.ipynb`
- **Foundry Agent Service**(교수 페르소나 에이전트)로 생성: `azure/foundry_agent_quickstart.ipynb` (`AZURE_MODE=agent`)

#### 🖥️ 로컬 Ollama (also supported, 자격 증명 불필요)

```bash
# 로컬에서 Ollama 서버를 띄우고 모델을 받은 뒤(예: ollama pull llama3.1)
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1 \
  OLLAMA_BASE_URL=http://localhost:11434 python run_local.py

# 차트/도표를 읽으려면 멀티모달 태그를 따로 지정 (텍스트=llama3.1, 비전=qwen2.5vl)
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1 \
  OLLAMA_VISION_MODEL=qwen2.5vl python run_local.py   # 대안: minicpm-v, llama3.2-vision, llava
```

> **비전 태그 선택**: 한국어·차트·표 OCR에는 `qwen2.5vl`·`minicpm-v`가 강하고, `llama3.2-vision`·`llava`도 지원됩니다. `OLLAMA_VISION_MODEL`을 비우면 `OLLAMA_MODEL`을 그대로 재사용합니다.
> **참고**: `MAI`(MAI-DS-R1)는 **텍스트 전용**이며 Ollama 라이브러리에 없습니다(멀티모달 아님) — 텍스트 추론은 Azure Foundry로 사용하세요.

#### 🟧 AWS (also supported)

```bash
# AWS Bedrock 사용
LLM_PROVIDER=bedrock docker run --rm -v $(pwd):/app -w /app --env-file .env \
  ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest \
  python run_local.py
```

SageMaker Processing 배치 잡 실행은 [PDF Q&A 추출 가이드](./pdf_qa_extraction/README.md)와 `sagemaker_processingjob_pdf_qa_extraction.ipynb`를 참조하세요.

#### 🖥️ 로컬 데모 웹앱 (싱글 노드)

CLI 대신 브라우저로 체험하려면 같은 이미지에서 웹앱을 띄웁니다. 문서를 올리고 페르소나를 고른 뒤 실행하면 **호스트 GPU 유무를 자동 감지**합니다. 미리보기 모드는 자격 증명 없이 오프라인으로 추출·페르소나 프롬프트를 보여주고, 전체 모드는 선택한 공급자로 Q&A까지 생성합니다(자세한 내용은 [PDF Q&A 추출 가이드](./pdf_qa_extraction/README.md)의 "로컬 데모 웹앱" 절 참조).

![PDF2LLM 로컬 데모 웹앱 — 미리보기 예시](assets/images/webapp-demo.png)

```bash
# http://localhost:8000 접속 (GPU 호스트는 --gpus all 추가)
docker run --rm -p 8000:8000 --env-file .env \
  ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest \
  python run_webapp.py
```

> 처리 방식은 두 가지 중 선택합니다 — **단일 노드·인프로세스**(기본 `WORKERS=1`, GPU 데모 권장)와 **멀티 프로세스**(`WORKERS=N`, CPU 동시성↑). 파일 없이도 UI의 **📄 샘플 문서로 시도**로 번들 PDF를 바로 미리볼 수 있습니다.

### 2단계: 데이터 전처리 및 분석

생성된 Q&A 데이터를 파인튜닝에 적합한 형식으로 변환합니다. `fine_tuning/02_data_preprocessing_and_analysis.ipynb`에서 진행하며, 입력 데이터는 **Azure Blob / AWS S3 / 로컬**에서 로드할 수 있습니다(`DATA_SOURCE` 환경 변수).

- 데이터 품질 검증(중복/짧은 응답 제거), 통계 분석/시각화, 학습·검증 분할, 학습 포맷 변환

### 3단계: LLM 파인튜닝

Unsloth와 LoRA 어댑터로 메모리 효율적 파인튜닝을 수행합니다. `fine_tuning/03_train_unsloth_model.ipynb`에서 진행하며, Azure ML Compute 또는 AWS/온프레미스 GPU 어디서든 동작합니다. 자세한 내용은 [LLM 파인튜닝 가이드](./fine_tuning/README.md)를 참조하세요.

## 🔀 공급자 전환 (멀티클라우드)

| 목표 | `LLM_PROVIDER` | 추가 설정 |
|---|---|---|
| Azure OpenAI (기본) | `azure` | `AZURE_MODE=openai`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |
| Azure AI Foundry Agent | `azure` | `AZURE_MODE=agent`, `AZURE_AI_PROJECT_ENDPOINT`, `AZURE_AI_AGENT_MODEL` |
| OpenAI 직접 | `openai` | `OPENAI_API_KEY` |
| AWS Bedrock | `bedrock` | AWS 자격증명 / SageMaker 실행 역할, `MODEL_ID` |

## 💡 기술 스택

- **Q&A 생성 (기본)**: Azure AI Foundry — Azure OpenAI · Foundry Agent Service
- **오케스트레이션 (기본)**: Azure Machine Learning (Command Job)
- **Q&A 생성 (also)**: Amazon Bedrock Claude, OpenAI(GPT-4o 등)
- **오케스트레이션 (also)**: Amazon SageMaker Processing
- **컨테이너/레지스트리**: Docker, GitHub Container Registry(GHCR, 공개)
- **PDF 추출**: Unstructured, CUDA
- **모델 파인튜닝**: Unsloth, PyTorch, LoRA
- **지원 모델**: Llama, Mistral, Gemma, Qwen 등 오픈소스 LLM

## 📊 성능 및 요구사항

### 하드웨어 요구사항
- **PDF 추출**: CPU (경량 슬림 이미지, 스캔 문서 레이아웃/OCR 가속용 GPU는 선택)
- **파인튜닝**: 최소 8GB VRAM (16GB+ 권장)

### 최적화 팁
1. **PDF 처리**: 대용량 PDF(100MB+)는 분할 처리, `batch_size`로 메모리 조절
2. **모델 파인튜닝**: 4비트 양자화로 메모리 75% 감소, `gradient_checkpointing="unsloth"`로 추가 30% VRAM 절약

## 🔗 확장 가능성
- **추가 모델 지원**: 새로운 오픈소스 LLM 적용
- **다국어 지원**: 다양한 언어 문서 처리
- **Azure ML / SageMaker 통합**: 대규모 병렬 파인튜닝·추론
- **RAG 시스템 구축**: 문서 임베딩으로 검색 증강 생성

---

> 각 모듈에 대한 자세한 정보는 해당 디렉토리의 README를 참조하세요:
> - [Azure ML + Foundry 가이드](./azure/README.md)
> - [PDF Q&A 추출 가이드](./pdf_qa_extraction/README.md)
> - [LLM 파인튜닝 가이드](./fine_tuning/README.md)

## 📚 참고 자료

- [Azure AI Foundry Documentation](https://learn.microsoft.com/azure/ai-foundry/)
- [Azure AI Foundry Agent Service](https://learn.microsoft.com/azure/ai-services/agents/)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/azure/machine-learning/)
- [Unsloth: Accelerating LLM Fine-tuning](https://github.com/unslothai/unsloth)
- [Unstructured: Open-source PDF extraction](https://github.com/Unstructured-IO/unstructured)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [OpenAI Platform Documentation](https://platform.openai.com/docs)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Llama 3: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/llama/)
- [teddylee777: LangChain 한국어 튜토리얼](https://github.com/teddylee777/langchain-kr)
