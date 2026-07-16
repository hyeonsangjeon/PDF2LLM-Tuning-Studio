# Azure ML + Azure AI Foundry 가이드 (Azure-first)

이 디렉터리는 PDF → Q&A 추출 파이프라인을 **Azure Machine Learning** + **Azure AI Foundry**
위에서 실행하기 위한 자산입니다. LLM 백엔드는 Foundry(Azure OpenAI 또는 Foundry Agent Service)를
기본으로 사용하며, 컨테이너 이미지는 **공개 GHCR 이미지 하나**를 Azure ML과 AWS SageMaker가 함께 사용합니다.

```
        PDF ─▶ [ghcr.io/.../pdf-qa-extractor  (공개 GPU 이미지)]
                 │ pull                              │ pull
        ┌────────▼──── Azure (lead) ──┐     ┌────────▼──── AWS (also) ──┐
        │ Azure ML Command Job        │     │ SageMaker Processing Job  │
        │  └▶ Azure AI Foundry        │     │  └▶ Bedrock Claude        │
        │ Blob · Key Vault · Entra ID │     │ S3 · SSM · IAM            │
        └─────────────────────────────┘     └───────────────────────────┘
```

## 구성 요소

| 파일 | 설명 |
|---|---|
| `azureml_job.yml` | Azure ML Command Job 스펙 (CLI `az ml job create -f` 용) |
| `azureml_pdf_qa_extraction.ipynb` | SDK(`azure-ai-ml`)로 배치 잡 제출/다운로드 데모 |
| `foundry_agent_quickstart.ipynb` | Foundry Agent Service(교수 페르소나) 활용 데모 |

관련 코어 코드는 `../pdf_qa_extraction/pdf_qa/`(클라우드 무관 패키지)와
`../assets/utils/keyvault.py`(SSM ↔ Key Vault 대응)입니다.

## 사전 준비

1. **Azure 구독** 및 리소스 그룹
2. **Azure AI Foundry** 프로젝트 + 모델 배포 (예: `gpt-4o`)
3. **Azure ML 워크스페이스** + 컴퓨트 클러스터 (`gpu-cluster` 권장, CPU면 `yolox` 사용)
4. 로컬: `az login` / Azure ML 컴퓨트: **Managed Identity**
5. 역할 부여 (RBAC): 실행 ID에
   - **Cognitive Services OpenAI User** (Azure OpenAI 사용 시)
   - **Azure AI User** (Foundry Agent 사용 시)

## 인증 (IAM → Entra ID / Managed Identity)

코드는 `DefaultAzureCredential`을 사용합니다.
- **로컬**: `az login` 자격증명
- **Azure ML 잡**: 컴퓨트의 Managed Identity (키 불필요, 권장)
- **키 방식**(선택): `AZURE_OPENAI_API_KEY` 설정 시 키 인증

## 시크릿 (SSM → Key Vault)

```python
import sys; sys.path.append("../assets/utils")
from keyvault import key_vault
kv = key_vault("https://<your-vault>.vault.azure.net/")
kv.put_params("PDF2LLM-REGION", "koreacentral")
print(kv.get_params("PDF2LLM-REGION"))
```

## 실행 방법

### A) CLI로 Command Job 제출
```bash
# azureml_job.yml 안의 AZURE_OPENAI_ENDPOINT/DEPLOYMENT를 먼저 수정
az ml job create -f azure/azureml_job.yml \
  --resource-group <rg> --workspace-name <ws>
```

### B) 노트북(SDK)으로 제출
`azureml_pdf_qa_extraction.ipynb` 참조 — PDF 업로드 → 잡 제출 → 결과 다운로드.

### C) 로컬 Docker로 실행 (Azure OpenAI)
```bash
docker pull ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest
docker run --rm --gpus all -v $(pwd):/app -w /app --env-file .env \
  ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest \
  python run_local.py
```

## 환경 변수 (Azure)

**Azure OpenAI 모드 (`AZURE_MODE=openai`, 기본)**
| 변수 | 예시 |
|---|---|
| `LLM_PROVIDER` | `azure` |
| `AZURE_MODE` | `openai` |
| `AZURE_OPENAI_ENDPOINT` | `https://<res>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` |
| `AZURE_OPENAI_API_KEY` | (선택, 미설정 시 Managed Identity) |

**Foundry Agent 모드 (`AZURE_MODE=agent`)**
| 변수 | 예시 |
|---|---|
| `LLM_PROVIDER` | `azure` |
| `AZURE_MODE` | `agent` |
| `AZURE_AI_PROJECT_ENDPOINT` | `https://<proj>.services.ai.azure.com/api/projects/<name>` |
| `AZURE_AI_AGENT_MODEL` | `gpt-4o` |

공통: `PDF_PATH`, `DOMAIN`, `NUM_QUESTIONS`, `NUM_IMG_QUESTIONS`, `TABLE_MODEL`.

## 공급자 전환 (멀티클라우드)

환경 변수만 바꾸면 동일 이미지·동일 코드로 백엔드가 교체됩니다.

| 목표 | `LLM_PROVIDER` | 추가 설정 |
|---|---|---|
| Azure OpenAI (기본) | `azure` | `AZURE_MODE=openai` |
| Foundry Agent | `azure` | `AZURE_MODE=agent`, `AZURE_AI_PROJECT_ENDPOINT` |
| OpenAI 직접 | `openai` | `OPENAI_API_KEY` |
| AWS Bedrock | `bedrock` | AWS 자격증명 / SageMaker 역할 |
