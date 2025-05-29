# PDF2LLM-Tuning-Studio

PDF 문서에서 지식을 추출하고 대규모 언어 모델(LLM)을 효율적으로 파인튜닝하는 엔드투엔드 파이프라인입니다. GPU 가속 PDF 파싱과 최적화된 LLM 파인튜닝을 결합하여 문서 기반 질의응답 시스템을 구축합니다.

## 📚 주요 기능

- **GPU 가속 PDF 추출**: Unstructured 라이브러리와 NVIDIA GPU를 활용한 고속 텍스트 추출
- **Q&A 자동 생성**: Amazon Bedrock Claude 모델을 사용하여 고품질 질문-답변 쌍 생성
- **메모리 효율적 파인튜닝**: Unsloth 최적화와 LoRA 어댑터로 제한된 GPU 환경에서도 대형 모델 학습 가능
- **완전 자동화 파이프라인**: PDF 문서 입력부터 맞춤형 LLM 모델 출력까지 자동화

## 🔍 프로젝트 구조

```
PDF2LLM-Tuning-Studio/
├── assets/                  # 공통 리소스 (이미지, 유틸리티)
│   ├── images/              # 다이어그램 및 이미지
│   └── utils/               # 공통 유틸리티 함수
│
├── pdf_qa_extraction/       # PDF 처리 및 Q&A 추출 모듈
│   ├── Dockerfile           # GPU 지원 PDF 추출 컨테이너
|   |── Dockerfile_event_eng # AWS Event 실습플렛폼의 네트워크 패키지 경로로 인한 Dockerfile 대용 
│   ├── processing_local.py  # 로컬 처리 스크립트
│   ├── processing.py        #SageMaker Processing job entrypoint 배치잡 실행파일
│   ├── sagemaker_processingjob_pdf_qa_extraction.ipynb # SageMaker Processing을 활용한 PDF 기반 QA 데이터 생성 배치 파이프라인 자동화 데모
│   └── README.md            # PDF 추출 가이드
│
└── fine_tuning/             # LLM 파인튜닝 모듈
    ├── 01_setup_environment.ipynb        # 환경 설정
    ├── 02_data_preprocessing_and_analysis.ipynb  # 데이터 전처리
    ├── 03_train_unsloth_model.ipynb      # Unsloth 모델 훈련
    └── README.md            # 파인튜닝 가이드
```

## 🚀 시작 가이드

### 1단계: PDF 텍스트 및 Q&A 추출

Cuda 컨테이너를 빌드해보고, GPU인스턴스 터미널에서 PDF 문서에서 텍스트 블록을 추출하고 고품질 Q&A 쌍을 생성합니다.

#### PDF 텍스트 및 Q&A 추출

PDF 문서 처리 및 Q&A 추출을 위한 Docker 환경을 설정하고 실행합니다. 자세한 내용은 [PDF Q&A 추출 가이드](./pdf_qa_extraction/README.md)를 참조하세요.

```bash
cd pdf_qa_extraction
# Docker 빌드 및 실행 지침은 PDF Q&A 추출 가이드 참조
```

이 모듈에서 수행되는 작업:
- NVIDIA GPU 가속 PDF 텍스트 추출
- Amazon Bedrock Claude를 활용한 고품질 Q&A 쌍 생성
- 문서 도메인 기반 맞춤형 질문 생성
- SageMaker Processing 배치 작업으로 자동화된 PDF 문서 처리 배치 프로세스


### 2단계: 데이터 전처리 및 분석

생성된 Q&A 데이터를 파인튜닝에 적합한 형식으로 변환합니다. 이 단계에서는 데이터 품질을 검증하고 모델 학습에 최적화된 형태로 준비합니다.

데이터 전처리 및 분석은 `fine_tuning/02_data_preprocessing_and_analysis.ipynb` 노트북에서 진행됩니다.

자세한 내용은 [LLM 파인튜닝 가이드](./fine_tuning/README.md)를 참조하세요.

이 노트북에서 수행되는 작업:
- 문서에서 추출한 데이터 품질 검증 (중복/짧은 응답 제거)
- 통계 분석 및 시각화
- 학습/검증 데이터셋 분할
- 모델 학습용 입력 포맷으로 변환

### 3단계: LLM 파인튜닝

Unsloth와 LoRA 어댑터를 사용하여 메모리 효율적인 파인튜닝을 수행합니다.

파인튜닝은 `fine_tuning/03_train_unsloth_model.ipynb` 노트북에서 진행됩니다.
자세한 내용은 [LLM 파인튜닝 가이드](./fine_tuning/README.md)를 참조하세요.


주요 단계:
1. 모델 로드 및 양자화 설정
2. LoRA 어댑터 구성
3. 학습 데이터 준비
4. 모델 훈련
5. 추론 및 테스트

## 💡 기술 스택

- **PDF 추출**: Unstructured, CUDA, Docker
- **질문-답변 생성**: Amazon Bedrock Claude
- **모델 파인튜닝**: Unsloth, PyTorch, LoRA
- **지원 모델**: Llama, Mistral, Gemma, Qwen 등 다양한 오픈소스 LLM

## 📊 성능 및 요구사항

### 하드웨어 요구사항

- **PDF 추출**: NVIDIA GPU (CUDA 지원)
- **파인튜닝**: 최소 8GB VRAM (16GB+ 권장)

### 최적화 팁

1. **PDF 처리**:
   - 대용량 PDF(100MB+)는 분할 처리
   - `batch_size` 파라미터로 메모리 사용량 조절

2. **모델 파인튜닝**:
   - 4비트 양자화로 메모리 사용량 75% 감소
   - `gradient_checkpointing="unsloth"`로 30% 추가 VRAM 절약
   - `batch_size`와 `gradient_accumulation_steps` 조정으로 메모리-속도 균형

## 🔗 확장 가능성

- **추가 모델 지원**: 새로운 오픈소스 LLM 모델 적용
- **다국어 지원**: 다양한 언어 문서 처리
- **SageMaker 통합**: AWS 환경에서 대규모 파인튜닝
- **RAG 시스템 구축**: 문서 임베딩으로 검색 증강 생성 시스템 개발

---

> 각 모듈에 대한 자세한 정보는 해당 디렉토리의 README 파일을 참조하세요:
> - PDF Q&A 추출 가이드
> - LLM 파인튜닝 가이드



## 📚 참고 자료

- [Unsloth: Accelerating LLM Fine-tuning](https://github.com/unslothai/unsloth)
- [Unstructured: Open-source PDF extraction](https://github.com/Unstructured-IO/unstructured)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Llama 3: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/llama/)
- [Fine-Tuning Llama-3-1-8B for Function Calling using LoRA](https://medium.com/@gautam75/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060)
- [teddylee777: LangChain 한국어 튜토리얼](https://github.com/teddylee777/langchain-kr)