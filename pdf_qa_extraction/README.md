# PDF QA 추출

이 디렉터리는 PDF 문서에서 질문-답변 쌍을 추출하는 도구를 포함하고 있습니다.

## 설치 안내

### Unstructured Docker 이미지 빌드하기

Unstructured는 PDF에서 콘텐츠를 추출하고 처리하기 위한 강력한 도구를 제공합니다. 이 도구는 문서의 블럭단위 text추출을 수행하여 구조화된 형식으로 데이터를 변환합니다. Docker 환경을 설정하려면 다음 단계를 따르세요:

1. 시스템에 Docker가 설치되어 있는지 확인하세요.

2. Docker 이미지 빌드:
     ```bash
     docker build -t qa-extractor -f Dockerfile .
     ```
3. 로컬 컨테이너에서 실행하는 방법:
     
     **환경 변수 설정 방법:**
     로컬에서 실행할 때는 python processing_local.py 과 동일 경로에  `.env` 파일을 사용하여 환경 변수를 설정할 수 있습니다. AWS 플랫폼(예: EC2, ECS, Lambda) 내에서 실행할 경우에는 액세스 키를 `.env` 파일에 저장하지 말고 IAM 역할을 사용하는 것이 보안 모범 사례입니다.

     -  `.env` 파일 생성 후 필요한 환경 변수 추가 예:
          ```
          AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
          AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
          AWS_REGION=USER_REGION
          PDF_PATH=data/fsi_data.pdf
          DOMAIN=International Finance
          NUM_QUESTIONS=5
          ```
     

     **Linux/macOS:**
     ```bash
     docker run --rm --gpus all \
          -v $(pwd):/app \
          -w /app \
          --env-file .env \
          qa-extractor \
          python processing_local.py

     #또는

     docker run --rm --gpus all \
         -v $(pwd):/app  \
         -w /app \
         -e AWS_REGION=us-east-1 \
         -e PDF_PATH=data/fsi_data.pdf \
         -e "DOMAIN=International Finance" \
         .....
         qa-extractor \
         python processing_local.py
     ```
     

     **Windows:**
     ```bash
     docker run --rm --gpus all ^
         -v %cd%:/app ^
         -w /app ^
         --env-file .env ^
         qa-extractor ^
         python processing_local.py
     
     #또는, 
     
     docker run --rm --gpus all ^
         -v %cd%:/app  ^
         -w /app ^
         -e AWS_REGION=us-east-1 ^
         -e PDF_PATH=data/fsi_data.pdf ^
         -e "DOMAIN=International Finance" ^
         .....
         qa-extractor ^
         python processing_local.py
     ```


### GPU기반 PDF Extractor 와 SageMaker Processing Jobs

Unstructured-qa-extractor 이미지는 Amazon SageMaker Processing Jobs를 통해 배치 작업으로 실행할 수 있습니다:

1. ECR에 이미지 푸시:
    터미널에서 아래 명령어들은 각각 ECR 인증, 이미지 태깅, 저장소 생성, 이미지 푸시 과정을 수행합니다. 로컬에서 빌드한 Docker 이미지를 AWS ECR에 등록하여 SageMaker에서 사용할 수 있게 합니다.```
     ```bash
     # ECR 로그인 - AWS 인증 수행
     aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
     # 로컬 이미지에 ECR 태그 지정
     docker tag qa-extractor <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/qa-extractor
     # ECR 저장소 생성
     aws ecr create-repository --repository-name qa-extractor --region <your-region>
     # 이미지를 ECR로 푸시
     docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/qa-extractor
     ```



2. SageMaker Processing Job 생성:

     SageMaker Processing Job은 데이터 전처리, 후처리, 모델 평가 등 ML 워크플로우의 다양한 단계를 처리하기 위한 AWS SageMaker의 기능입니다.
     Unstructured Q&A Processing Job 생성 방법에 대한 자세한 예제는 `sagemaker_processingjob_pdf_qa_extraction.ipynb` 노트북을 참조하세요. 
     

          ```python
          from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

          # 프로세서 객체 생성
          processor = Processor(
              role='your-iam-role',
              image_uri='your-container-image',
              instance_count=1,
              instance_type='ml.g5.xlarge',
              volume_size_in_gb=30
          )

          # 처리 작업 실행
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
              code='path/to/your/processing_script.py'
          )
          ```
          
          **Processing Job 설정 설명:**
          - `role`: SageMaker가 AWS 리소스에 접근할 수 있는 IAM 역할 ARN
          - `image_uri`: ECR에 업로드한 qa-extractor 컨테이너 이미지 URI
          - `instance_count`: 실행할 인스턴스 수 (병렬 처리 시 증가)
          - `instance_type`: 처리 작업에 사용할 gpu 인스턴스 유형 
          - `volume_size_in_gb`: 처리 작업에 할당할 EBS 저장 볼륨 크기
          - `inputs`: S3 버킷에서 컨테이너로 가져올 데이터 경로 지정 (/opt/ml/processing/ 은 default)
          - `outputs`: 처리 결과를 저장할 S3 경로 지정 (/opt/ml/processing/ 은 default)
          - `code`: 컨테이너 내부에서 실행할 처리 스크립트 경로
          
          
이 방식을 사용하면 대규모 PDF 처리 작업을 효율적으로 관리하고 확장할 수 있습니다.

## 사용법

이 디렉터리는 다음을 위한 스크립트를 포함합니다:
- PDF 텍스트 추출
- 콘텐츠 처리
- 질문-답변 쌍 생성

각 도구의 사용 방법에 대한 자세한 내용은 개별 스크립트 문서를 참조하세요.

## 의존성

- Python 3.8+
- Unstructured GPU TEXT Extractor Image 
- SageMaker Processin Job