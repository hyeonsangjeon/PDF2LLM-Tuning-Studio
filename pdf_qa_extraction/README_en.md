# PDF QA Extraction

[English](README_en.md) | [한국어](README.md)

This tool extracts text/tables/images from PDF documents (layout detection via onnxruntime, OCR via tesseract — the default image is a CPU slim build) and uses an LLM to automatically generate high-quality question-answer pairs from the extracted content. Through this process, document knowledge is transformed into structured QA JSONL datasets that can be used for training, fine-tuning, or knowledge base construction.

The LLM provider is selected with a single environment variable `LLM_PROVIDER` — **`azure` (default, Azure AI Foundry / Azure OpenAI)**, `openai`, or `bedrock` (AWS). The core logic lives in the cloud-agnostic `pdf_qa` package, and thin per-runtime entrypoints (`run_local.py` local/container, `azureml_job.py` Azure ML, `processing.py` SageMaker) wrap it.

> For Azure ML/Foundry setup details, see [`../azure/README.md`](../azure/README.md).

[PDF QA Extraction Process Video Guide](https://assets.fsi.kr/videos/qna-extract.mp4)

[WorkshopStudio Manual](https://catalog.us-east-1.prod.workshops.aws/workshops/61cd351b-6326-4618-ad97-e318ed31472f/ko-KR)

## System Flow Diagram

![GPU Container Process](../assets/images/flow.png)

*The above diagram shows the complete process of extracting QA data from PDF documents. When a PDF document is input, it is converted into text blocks through the Unstructured partition extractor, and this data is processed into structured JSONL QA data using the selected LLM provider (Azure AI Foundry / OpenAI / Bedrock Claude).*

## Installation Guide

### Use the public container image (recommended)

You can use the public GHCR image directly without building. A single image is shared by local, Azure ML, and SageMaker runtimes.

```bash
# Default: CPU slim image (~2GB) — recommended for most cases
docker pull ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest

# GPU acceleration: CUDA torch + onnxruntime-gpu image (~8GB) — run with `--gpus all` on an NVIDIA host
docker pull ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest-gpu
```

> **CPU vs GPU image**: with `:latest-gpu` (CUDA torch + **onnxruntime-gpu**) run
> via `--gpus all`, **both** model families are GPU-accelerated: **layout
> detection (YOLOX/detectron2_onnx on onnxruntime-gpu)** and the
> **table-structure model (Table Transformer, PyTorch)**. **OCR (Tesseract)** and
> digital text extraction (pdfminer) are always CPU. See the per-model breakdown
> in [PDF parsing models — GPU/CPU](#pdf-parsing-models--gpucpu) below. If you
> only process digital PDFs or have no GPU, the default `:latest` (CPU) is enough.

### Building Unstructured CUDA Docker Image

Unstructured provides powerful tools for extracting and processing content from PDFs. This tool performs block-level text extraction from documents to convert data into structured format. To set up the Docker environment, follow these steps:

1. Ensure Docker is installed on your system.

2. Build Docker image:
     ```bash
     docker build -t pdf-qa-extractor -f Dockerfile .
     ```

     ```bash
     # Event Engine lab accounts have network restrictions and must use this Dockerfile_event_eng.
     docker build -t pdf-qa-extractor -f Dockerfile_event_eng .     
     ```

> **Lightweight image design (~6GB → ~2GB)**: the default image is **CPU-only**.
> The extraction *code* never calls CUDA directly, so it runs fine on CPU:
> unstructured parses PDFs (`hi_res` layout = PyTorch/ONNX model + tesseract OCR)
> and the LLM runs over the *network* (Azure/Bedrock/OpenAI). torch is only a
> transitive dep of `unstructured[pdf]`, yet its default PyPI wheel bundles
> ~3.5GB of NVIDIA CUDA libraries (cudnn/cublas/...). So the image installs the
> **CPU torch wheel (~200MB)** and drops the CUDA base image (multi-stage:
> compilers/headers stay in the builder).
>
> **For GPU acceleration**, use the `:latest-gpu` image above or build with CUDA
> torch + onnxruntime-gpu. `:latest-gpu` runs **both layout (onnxruntime-gpu)**
> and **table structure (CUDA torch)** on the GPU (run with `--gpus all`;
> `libcuda` is injected by nvidia-container-toolkit). See the per-model breakdown
> in [PDF parsing models — GPU/CPU](#pdf-parsing-models--gpucpu). The heavy LoRA
> fine-tuning still runs in a separate Azure ML / SageMaker job.
> ```bash
> docker build \
>   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
>   --build-arg ONNXRUNTIME_PACKAGE=onnxruntime-gpu \
>   -t pdf-qa-extractor:gpu -f Dockerfile .
> ```

### PDF parsing models — GPU/CPU

`unstructured.partition.pdf.partition_pdf` is not one model — it runs **different
engines per stage**. PyTorch stages ride the GPU with CUDA torch; ONNX stages ride
it with onnxruntime-gpu; Tesseract and pdfminer are separate. Actual acceleration
for our images (`:latest` = CPU torch + CPU onnxruntime, `:latest-gpu` = **CUDA
torch + onnxruntime-gpu**):

| Stage | Model / library | Engine | `:latest` (CPU) | `:latest-gpu` (GPU) |
|---|---|---|---|---|
| Text extraction (digital PDF, `strategy="fast"`) | pdfminer.six | pure Python | CPU | CPU (no model) |
| Layout detection (default `yolox`) | YOLOX | **onnxruntime** | CPU | **GPU** (onnxruntime-gpu) |
| Layout detection (alt `detectron2_onnx`, …) | Detectron2-ONNX | **onnxruntime** | CPU | **GPU** (onnxruntime-gpu) |
| Table structure (`infer_table_structure=True`) | Table Transformer (TATR) | **PyTorch / transformers** | CPU | **GPU** (CUDA torch) |
| OCR (scanned pages, `hi_res`/`ocr_only`) | Tesseract | Tesseract C++ | CPU | **CPU (always)** |
| Image block extraction/crop | Pillow · pdf2image · OpenCV | CPU | CPU | CPU |

**Key takeaways**
- `:latest-gpu` runs **both layout detection (onnxruntime-gpu)** and the
  **table-structure model (CUDA torch)** on the GPU. Layout runs automatically
  under the `hi_res` strategy (scanned PDFs, `extract_images_in_pdf`, …); the
  table model only loads when `infer_table_structure=True` — set the `TABLE_MODEL`
  env var (or `--table_model`) to enable it.
- GPU acceleration only kicks in with a **compatible NVIDIA driver + `--gpus
  all`**. `:latest-gpu` targets CUDA 12.x (cu124) by default; for an older driver,
  rebuild with `TORCH_INDEX_URL=cu121`/`cu118` and a matching onnxruntime-gpu. The
  CUDA/cuDNN runtime libs ship inside the torch and onnxruntime-gpu wheels, so
  they're baked into the image.
- **OCR (Tesseract)** and digital-PDF text extraction (pdfminer) are **always
  CPU**.
- The default `:latest` uses **CPU-only onnxruntime + CPU torch**, so every stage
  is CPU (and the image is small).

**Selecting models (choosing among several)**
- **Strategy**: `partition_pdf(strategy=...)` — `"fast"` (pdfminer, no model),
  `"hi_res"` (layout + optional OCR/table), `"ocr_only"` (Tesseract), `"auto"`
  (default; picks per document/options).
- **Layout model**: `hi_res_model_name=` arg or `UNSTRUCTURED_DEFAULT_MODEL_NAME`
  env. Options: `yolox` (default), `yolox_tiny`, `yolox_quantized`,
  `detectron2_onnx`, `detectron2_quantized`, `detectron2_mask_rcnn` (all ONNX →
  run on the GPU via onnxruntime-gpu in `:latest-gpu`).
- **Table**: `infer_table_structure=True` → Table Transformer (PyTorch). Enabled
  here via the `TABLE_MODEL` env var.

> **Why your 3080 lit up on the old image**: older unstructured used a **PyTorch
> detectron2** (layoutparser) layout model by default, so CUDA torch alone put
> the layout stage on the GPU. Current versions default layout to **ONNX
> (YOLOX)**, which uses the onnxruntime engine — so `:latest-gpu` ships
> **onnxruntime-gpu** to put layout back on the GPU (old behaviour + the table
> model on the GPU too).

> **Switching a CPU install to GPU by hand**: `onnxruntime` (CPU) and
> `onnxruntime-gpu` both provide the `onnxruntime` module and **conflict**, so
> remove the CPU one first — `pip uninstall -y onnxruntime && pip install
> onnxruntime-gpu`. (This repo does that automatically in the GPU build via the
> `ONNXRUNTIME_PACKAGE=onnxruntime-gpu` build-arg.) On a driver/CUDA mismatch it
> silently falls back to CPU.

### PDF Extractor Usage Guide

#### 1. Running Locally

The Unstructured Extractor extracts text/tables/images from PDF documents (layout via onnxruntime, OCR via tesseract — the **default image is a CPU slim build**, so it runs without a GPU). Use the unified entrypoint `run_local.py` and select the provider via the `LLM_PROVIDER` environment variable (default `azure`). Use the public image pulled above or a locally built `pdf-qa-extractor`.

```bash
# Image alias (when using the public image)
IMG=ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest

# Azure AI Foundry / Azure OpenAI (default, Linux/macOS)
docker run --rm -v $(pwd):/app -w /app --env-file .env $IMG python run_local.py

# AWS Bedrock (Linux/macOS)
LLM_PROVIDER=bedrock docker run --rm -v $(pwd):/app -w /app --env-file .env $IMG python run_local.py

# OpenAI (Linux/macOS)
LLM_PROVIDER=openai docker run --rm -v $(pwd):/app -w /app --env-file .env $IMG python run_local.py

# On Windows (PowerShell/CMD), use %cd% instead of $(pwd)
docker run --rm -v %cd%:/app -w /app --env-file .env %IMG% python run_local.py
```

> Backward compatible: `processing_local.py` (Bedrock) and `processing_local_openai.py` (OpenAI) still work; they set `LLM_PROVIDER` internally and call the same core as `run_local.py`.

> **Only when running a GPU build**, add `--gpus all` to the commands above and verify the host GPU is visible:
> ```bash
> docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
> ```

#### Environment Variable Configuration

Common: `PDF_PATH`, `DOMAIN`, `NUM_QUESTIONS`, `NUM_IMG_QUESTIONS`, `TABLE_MODEL` (e.g., yolox), `PERSONA` (Q&A style), `STRATEGY` (extraction strategy), `GPU_BOOST` (auto GPU acceleration on/off).

##### 🎭 Personas (choosing the Q&A style)

One PDF can seed **several different fine-tuning datasets**. The `PERSONA` env var (or `--persona`) swaps the role the model plays and its question/answer style. Each persona follows a **genuinely different method (방식)**, while the output JSON schema (`QUESTION`/`ANSWER`) stays the same.

| `PERSONA` | Role | Method |
|---|---|---|
| `professor` (default) | Teacher/Professor | Exam-setter method — broad, single-answer factual questions |
| `socratic` | Socratic tutor | "Why/how" prompts; answers reason 근거→과정→결론 step by step |
| `consultant` | Senior practitioner | Decision/risk/implication-oriented advisory Q&A |
| `interviewer` | Technical interviewer | Escalating interview questions with concise model answers |
| `analyst` | Research analyst | Synthesis/comparison across the document, drawing implications |
| `feynman` | Feynman (plain talk) | First-principles + everyday analogies, no jargon (Feynman technique) |
| `memoirist` | Autobiographer (1st person) | Recounts a life story in the first person ("나는…") — events, people, feelings, decisions and lessons — without inventing anything absent from the context |

> Personas live in a **YAML ledger (`pdf_qa/personas.yaml`)**, not in code. Edit that file to tweak wording/methods or add new personas, or point the `PERSONA_FILE` env var at your own external YAML to manage a separate ledger.

```bash
# e.g. build a Socratic study dataset from the same PDF
PERSONA=socratic LLM_PROVIDER=azure PDF_PATH=data/fsi_data.pdf python run_local.py

# e.g. turn an autobiography PDF into a first-person recollection dataset —
# capture a person's own voice in an SLM
PERSONA=memoirist DOMAIN="my father's life" PDF_PATH=data/memoir.pdf python run_local.py

# e.g. run your own persona ledger from an external file
PERSONA_FILE=/path/to/my_personas.yaml PERSONA=feynman python run_local.py
```

##### ⚡ Automatic GPU acceleration (device-aware logic)

At startup the pipeline **probes the device** and logs it ("디바이스 점검(GPU/CPU)"). When a GPU is actually reachable (`torch.cuda.is_available()` = NVIDIA driver present) it **routes the heavier, higher-quality path to the GPU automatically**:

- with `STRATEGY=auto` (default), a detected GPU **escalates to `hi_res`**, so the layout model (YOLOX/detectron2_onnx) runs on **onnxruntime-gpu**;
- **table-structure inference (Table Transformer, CUDA torch)** is turned on automatically (it is too slow to enable by default on CPU);
- on a CPU host the light path is kept → fast.

So running the `:latest-gpu` image with `--gpus all` lets this logic surface the GPU advantage on its own. Disable it with `GPU_BOOST=false`, or pin the strategy explicitly with `STRATEGY=fast|hi_res|ocr_only`.

```bash
# GPU host: probe device, then run hi_res layout + table structure on the GPU
docker run --rm --gpus all -e PERSONA=analyst \
  ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest-gpu
```


**☁️ For Azure AI Foundry / Azure OpenAI (default, `LLM_PROVIDER=azure`)**
- `AZURE_MODE`: `openai` (default) or `agent` (Foundry Agent Service)
- `AZURE_OPENAI_ENDPOINT`: e.g., `https://<res>.openai.azure.com/`
- `AZURE_OPENAI_DEPLOYMENT`: deployment name (e.g., gpt-4o)
- `AZURE_OPENAI_API_VERSION`: e.g., 2024-10-21
- `AZURE_OPENAI_API_KEY`: (optional) keyless via `DefaultAzureCredential` (Managed Identity / `az login`) when unset
- (agent mode) `AZURE_AI_PROJECT_ENDPOINT`, `AZURE_AI_AGENT_MODEL`

**🟧 For AWS Bedrock (`LLM_PROVIDER=bedrock`)**
- `AWS_REGION`: AWS region (e.g., us-east-1)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` (optional): credentials (use the execution role on SageMaker)
- `MODEL_ID`: Bedrock model ID (e.g., anthropic.claude-3-sonnet-20240229-v1:0)

**For OpenAI (`LLM_PROVIDER=openai`)**
- `OPENAI_API_KEY`: OpenAI API key

#### 2. 🖥️ Local Demo Web App (Single Node)

Parallel fan-out (SageMaker / Azure ML distributed processing) stays the **advanced** path; for a "this is how you use it locally" experience we ship a **single-node web app inside the same image**. Upload a document → pick a persona → click, and it **auto-detects the host GPU** and runs accordingly.

**Architecture — the container *is* the web app (in-process):** the app does **not** call a separate container as an API, nor spawn a process per request. It imports `pdf_qa` and calls it **in the same process**, so torch/onnxruntime run inline and a GPU passed via `--gpus all` is picked up automatically. It also exposes a REST endpoint (`POST /api/extract`), so it doubles as an API when you need one.

```bash
# Image alias
IMG=ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest      # CPU slim
# IMG=ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest-gpu  # GPU build

# Launch the web app → open http://localhost:8000
docker run --rm -p 8000:8000 --env-file .env $IMG python run_webapp.py

# GPU host: just add --gpus all → hi_res layout + table structure run on the GPU automatically
docker run --rm --gpus all -p 8000:8000 --env-file .env \
  ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest-gpu python run_webapp.py
```

Run locally (no container):

```bash
pip install ".[pdf,azure,webapp]"   # extraction + provider + web app deps
python run_webapp.py                 # HOST/PORT env vars change the bind (default 0.0.0.0:8000)
```

**Two modes**
- **Preview — offline, no cloud credentials:** partitions the PDF device-aware and shows **element/table/image counts + the actual device path + the persona-rendered prompt**. No LLM call, so you can see the **GPU acceleration strength and persona differences without any credentials**.
- **Full — credentials required:** additionally calls the selected provider (Azure/OpenAI/Bedrock) to **generate Q&A pairs**, returning a results table + a `JSONL` download.

> On load the UI shows a **GPU/CPU badge** (`/api/device`), the persona list with method summaries (`/api/personas`), and provider-configured hints (`/api/providers`). For large / distributed workloads, see the SageMaker / Azure ML parallel processing section below.

## Table Extraction Model Comparison

### Detailed Performance Comparison

| Model | Vendor | Accuracy | Speed | GPU Memory | Features |
|-------|--------|----------|-------|------------|----------|
| detectron2 | Meta | ⭐⭐⭐⭐⭐ | ⭐⭐ | High | Highest accuracy, research use |
| detectron2_onnx | Meta | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | ONNX optimized version |
| table-transformer | Microsoft | ⭐⭐⭐⭐⭐ | ⭐⭐ | High | Excellent for complex tables, Download issue in SageMaker Processing (2025-09-08) |
| tatr | Community | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | Balanced performance |
| yolox | Megvii | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | Fast processing |
| yolox_quantized | Megvii | ⭐⭐ | ⭐⭐⭐⭐⭐ | Very Low | Ultra-fast processing |
| paddle | Baidu | ⭐⭐⭐ | ⭐⭐⭐ | Medium | Chinese table specialized |
| chipper | Community | ⭐⭐ | ⭐⭐⭐⭐ | Low | Lightweight model |

### Recommended Models by GPU Memory

| GPU Memory | Recommended Model | Features |
|------------|-------------------|----------|
| Under 4GB | yolox_quantized | Ultra-lightweight |
| 4-8GB | yolox | Basic |
| 8-12GB | tatr | Balanced |
| 12-16GB | table-transformer | High performance |
| 16-24GB | detectron2_onnx | Meta optimized |
| 24GB+ | detectron2 | Meta highest performance |

### Recommended Models by Use Case

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Research paper analysis | detectron2 | Highest accuracy required |
| Reports | table-transformer | Many complex tables |
| General documents | tatr | Balanced choice |
| Real-time processing | yolox_quantized | Speed priority |
| Batch processing | detectron2_onnx | Large-scale processing |
| Chinese documents | paddle | Language specialized |
| IoT Edge | chipper | Lightweight |

```bash
# Create .env file
touch .env

# Open .env file with vi editor
vi .env
```

Press i to enter input mode
Copy and paste the content below:

**☁️ For Azure AI Foundry / Azure OpenAI (default):**
```bash
# App Setting
PDF_PATH=data/fsi_data.pdf
DOMAIN=International Finance
NUM_QUESTIONS=5
NUM_IMG_QUESTIONS=1
TABLE_MODEL=yolox

# LLM Provider
LLM_PROVIDER=azure
AZURE_MODE=openai

# Azure OpenAI (Foundry deployment)
AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-21
# Omit the next line to use keyless auth (Managed Identity / az login)
AZURE_OPENAI_API_KEY=your_azure_openai_key_here

# Press ESC and type :wq to save and exit
```

**For AWS Bedrock:**
```bash
# App Setting
PDF_PATH=data/fsi_data.pdf
DOMAIN=International Finance
NUM_QUESTIONS=5
NUM_IMG_QUESTIONS=1
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
TABLE_MODEL=yolox

# LLM Provider
LLM_PROVIDER=bedrock

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_SESSION_TOKEN=your_session_token_here

# Press ESC and type :wq to save and exit
```

**For OpenAI:**
```bash
# App Setting
PDF_PATH=data/fsi_data.pdf
DOMAIN=International Finance
NUM_QUESTIONS=5
NUM_IMG_QUESTIONS=1
TABLE_MODEL=yolox

# LLM Provider
LLM_PROVIDER=openai

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Press ESC and type :wq to save and exit
```

> **Note**: Use the `.env` file only for local testing purposes. In production, using Azure Managed Identity (or AWS IAM roles) is the reference architecture practice. Please be careful not to expose Azure/AWS/OpenAI keys externally.

#### Performance Optimization Tips

- Large PDF files (over 100MB) should be split before processing
- **Automatic GPU acceleration**: run the `:latest-gpu` image with `--gpus all` and the pipeline probes the device; when a GPU is detected it **escalates `STRATEGY=auto` to `hi_res`** and runs **layout (onnxruntime-gpu)** and **table structure (Table Transformer, CUDA torch)** on the GPU automatically. Turn it off with `GPU_BOOST=false` (OCR is always CPU — see the [GPU/CPU breakdown](#pdf-parsing-models--gpucpu) above)
- **Use personas**: run the same PDF several times with `PERSONA=professor|socratic|consultant|interviewer|analyst|feynman|memoirist` to accumulate datasets built with different methods. Edit `pdf_qa/personas.yaml` (or point `PERSONA_FILE` at your own YAML) to customize the ledger. (e.g. `memoirist` turns an autobiography PDF into first-person recollection Q&A that captures a person's voice in an SLM)
- Monitor memory usage and adjust the `batch_size` parameter if necessary (refer to partition_pdf in the code)

#### 2. ☁️ Running on Azure ML Command Job (default)

Run the same public image as an Azure Machine Learning batch job. Input PDFs are mounted from a Blob datastore and the resulting `qa_pairs.jsonl` is uploaded back to Blob.

```bash
# Review the image/endpoint in azure/azureml_job.yml, then submit
az ml job create -f ../azure/azureml_job.yml --resource-group <rg> --workspace-name <ws>
```

- SDK submit/download demo: `../azure/azureml_pdf_qa_extraction.ipynb`
- Foundry Agent Service demo: `../azure/foundry_agent_quickstart.ipynb`
- Workspace/compute/RBAC/Key Vault setup: [`../azure/README.md`](../azure/README.md)

The entrypoint is `azureml_job.py` (`--input-dir`/`--output-dir`); the provider is chosen via `LLM_PROVIDER` (default `azure`).

#### 3. 🟧 Running on SageMaker Processing Job (also supported)

The Unstructured pdf-qa-extractor image can also be run as a batch job through Amazon SageMaker Processing Jobs:

1. (Optional) Use the public GHCR image directly, or push to ECR if you need a private mirror:
    The following commands in the terminal perform ECR authentication, image tagging, repository creation, and image push processes respectively. They register the locally built Docker image to AWS ECR so it can be used in SageMaker.
     ```bash
     # ECR login - Perform AWS authentication
     aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
     # Tag local image for ECR
     docker tag pdf-qa-extractor <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/pdf-qa-extractor
     # Create ECR repository
     aws ecr create-repository --repository-name pdf-qa-extractor --region <your-region>
     # Push image to ECR
     docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/pdf-qa-extractor
     ```

2. Create SageMaker Processing Job:

     SageMaker Processing Job is a feature of AWS SageMaker for handling various stages of ML workflows such as data preprocessing, postprocessing, and model evaluation.
     For detailed examples on creating Unstructured Q&A Processing Jobs, refer to the `sagemaker_processingjob_pdf_qa_extraction.ipynb` notebook.
     
          ```python
          from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

          # Create processor object
          processor = Processor(
              role='your-iam-role',
              image_uri='your-container-image',
              instance_count=1,
              instance_type='ml.g5.xlarge',
              volume_size_in_gb=30
          )

          # Run processing job
          processor.run(
              inputs=[
                  ProcessingInput(
                      source='s3://your-bucket/input-data',
                      destination='/opt/ml/processing/input'
                  )
              ],
              outputs=[
                  ProcessingOutput(
                      source='/opt/ml/processing/output',
                      destination='s3://your-bucket/output-data'
                  )
              ],
              code='processing.py'
          )
          ```
          
          **Processing Job Configuration Explanation:**
          - `role`: IAM role ARN for SageMaker to access AWS resources
          - `image_uri`: pdf-qa-extractor container image URI in ECR (or public GHCR)
          - `instance_count`: Number of instances to run (increase for parallel processing)
          - `instance_type`: GPU instance type to use for processing job
          - `volume_size_in_gb`: EBS storage volume size allocated for processing job
          - `inputs`: Specify data path to import from S3 bucket to container (/opt/ml/processing/ is default)
          - `outputs`: Specify S3 path to store processing results (/opt/ml/processing/ is default)
          - `code`: Path to processing script to run inside container

This approach allows efficient management and scaling of large-scale PDF processing tasks.

### Advantages of Parallel Processing with SageMaker

SageMaker Processing Jobs enable parallel processing by running multiple independent jobs simultaneously, each handling different document ranges or types with flexible LLM provider selection.

![Parallel Document Processing with Different LLMs](../assets/images/diff_docs_swap_llm.png)

*The diagram above illustrates how multiple SageMaker Processing Jobs can run in parallel, each processing different document sets with different LLM providers simultaneously.*

#### Key Benefits:

**1. Multiple Job Parallel Execution**
- Launch multiple independent Processing Jobs simultaneously, each with `instance_count=1`
- Each job processes a specific document range, type, or category
- **Large Document Sets**: Even for the same document type, split large collections into multiple jobs for parallel processing
  - Example: 1,000 financial reports can be split into 10 jobs processing 100 documents each
  - Reduces processing time from 10 hours (sequential) to 1 hour (10 parallel jobs)
- Dramatically reduce total processing time for large document collections
- Jobs run independently without interfering with each other

**2. Flexible Job Configuration per Document Type**
- Assign different LLM providers to different jobs based on document characteristics
- Example:
  - Job 1: Financial reports → AWS Bedrock Claude (domain expertise)
  - Job 2: Technical papers → OpenAI GPT-4 (technical understanding)
  - Job 3: Legal documents → AWS Bedrock Claude with specialized prompts
- Optimize cost and quality by matching the right LLM to each document type

**3. Scalability & Cost Optimization**
- Scale horizontally by launching more jobs, not by increasing instance count per job
- Process hundreds of documents in parallel across multiple jobs
- Pay only for the compute time used during processing
- Auto-shutdown after each job completion prevents idle resource costs

**4. Fault Tolerance & Reliability**
- Failed jobs can be easily retried without affecting other running jobs
- Independent job execution ensures one document set failure doesn't impact others
- Complete audit trail and logging for all processing activities
- Easy to identify and reprocess specific document ranges if needed

#### Parallel Processing Example:

```python
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
import time

# Define different job configurations
jobs_config = [
    # Case 1: Different document types with different LLMs
    {
        'name': 'financial-docs-bedrock',
        'input': 's3://bucket/financial-reports/',
        'output': 's3://bucket/qa-results/financial/',
        'llm_provider': 'bedrock',
        'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'
    },
    {
        'name': 'technical-docs-openai',
        'input': 's3://bucket/technical-papers/',
        'output': 's3://bucket/qa-results/technical/',
        'llm_provider': 'openai',
        'model': 'gpt-4o'
    },
    {
        'name': 'legal-docs-bedrock',
        'input': 's3://bucket/legal-documents/',
        'output': 's3://bucket/qa-results/legal/',
        'llm_provider': 'bedrock',
        'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'
    },
    # Case 2: Same document type, split by document range (large volume)
    {
        'name': 'financial-reports-batch-1',
        'input': 's3://bucket/financial-reports/batch-001-100/',  # Documents 1-100
        'output': 's3://bucket/qa-results/financial-batch-1/',
        'llm_provider': 'bedrock',
        'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'
    },
    {
        'name': 'financial-reports-batch-2',
        'input': 's3://bucket/financial-reports/batch-101-200/',  # Documents 101-200
        'output': 's3://bucket/qa-results/financial-batch-2/',
        'llm_provider': 'bedrock',
        'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'
    },
    {
        'name': 'financial-reports-batch-3',
        'input': 's3://bucket/financial-reports/batch-201-300/',  # Documents 201-300
        'output': 's3://bucket/qa-results/financial-batch-3/',
        'llm_provider': 'bedrock',
        'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0'
    }
]

# Launch all jobs in parallel
for config in jobs_config:
    processor = Processor(
        role='your-iam-role',
        image_uri=f'your-qa-extractor-{config["llm_provider"]}-image',
        instance_count=1,  # Each job uses only 1 instance
        instance_type='ml.g5.xlarge',
        volume_size_in_gb=30
    )

    # Run job asynchronously (non-blocking)
    processor.run(
        job_name=f'qa-extraction-{config["name"]}-{int(time.time())}',
        inputs=[
            ProcessingInput(
                source=config['input'],
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                source='/opt/ml/processing/output',
                destination=config['output']
            )
        ],
        wait=False  # Don't wait for job completion, launch next job immediately
    )

print(f"Launched {len(jobs_config)} parallel processing jobs successfully!")
```

**Performance Comparison:**

*Example 1: Different Document Types*
- **Sequential Processing**: 3 document types × 30 min/type = 90 minutes
- **Parallel Processing**: max(30 min, 30 min, 30 min) = 30 minutes
- **Speed Improvement**: 3x faster

*Example 2: Large Volume Same Document Type (300 financial reports)*
- **Sequential Processing**: 300 documents × 2 min/doc = 600 minutes (10 hours)
- **Parallel Processing (3 jobs)**: 100 documents × 2 min/doc = 200 minutes (3.3 hours)
- **Speed Improvement**: 3x faster
- **Parallel Processing (10 jobs)**: 30 documents × 2 min/doc = 60 minutes (1 hour)
- **Speed Improvement**: 10x faster

**Cost Comparison:**
- Total compute time is the same (instance-minutes remain constant)
- Wall-clock time dramatically reduced (better time-to-insight)
- Ability to use spot instances for additional 70% cost savings
- No idle time between batches, maximizing resource utilization

This parallel job execution capability makes SageMaker Processing ideal for enterprise-scale document processing workflows where different document types require different handling strategies and LLM providers.

## Usage

This directory contains scripts for:
- PDF text extraction
- Content processing
- Question-answer pair generation

Please refer to individual script documentation for detailed usage of each tool.

## Dependencies

- Python 3.10+ (`pdf_qa` package, see `pyproject.toml`)
- Unstructured text/image extractor image (public GHCR `pdf-qa-extractor`, CPU slim)
- LLM provider SDKs: `azure` (azure-ai-projects · openai) / `openai` / `bedrock` (langchain-aws) — see extras in `requirements.txt`
- Execution runtime (optional): Azure ML Command Job or AWS SageMaker Processing Job
