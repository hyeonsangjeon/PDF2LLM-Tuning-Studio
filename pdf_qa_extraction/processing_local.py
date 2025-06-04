from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import json 
import boto3
import os

"""
local_processing.py

이 스크립트는 다음 단계로 이루어집니다:

1) .env 파일 로드
2) PDF 파티셔닝(텍스트/이미지/테이블 등 추출, GPU 사용여부 등)
3) Prompt Template 정의
4) AWS Bedrock(Claude) 클라이언트 설정
5) 체인 구성(Runnable 파이프라인)
6) PDF 요소들에 대해 체인 실행 & 질의응답 생성
7) 최종 JSONL 파일로 저장

환경 변수:
  - AWS_REGION
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY

사용 라이브러리:
  - dotenv, unstructured, langchain-core, langchain-aws, boto3 등
"""

# -------------------------------------------------------------------
# 1) .env 파일 로드
# .env 파일에서 환경 변수를 로드합니다.
# -------------------------------------------------------------------


load_dotenv()


# -------------------------------------------------------------------
# 2) PDF 파티셔닝 함수 정의
# PDF 파일에서 이미지, 테이블, 그리고 텍스트 조각을 추출합니다.
# -------------------------------------------------------------------

def extract_elements_from_pdf(filepath):
    """
    Extracts elements from a PDF file using specified partitioning strategies.
    
    Args:
        filepath (str): The path to the PDF file to be processed.

    Returns:
        list: A list of extracted elements from the PDF.

    Keyword Args:
        filename (str): The path to the PDF file to be processed.
        extract_images_in_pdf (bool): Whether to extract images from the PDF. This flag utilize GPU resources when available for improved performance in image recognition and structure inference.  Defaults to False. 
        infer_table_structure (bool): Whether to infer table structures in the PDF. This flag utilize GPU resources when available for improved performance in image recognition and structure inference. Defaults to False.
        chunking_strategy (str): The strategy to use for chunking text. Defaults to "by_title".
        max_characters (int): The maximum number of characters in a chunk. Defaults to 4000.
        new_after_n_chars (int): The number of characters after which a new chunk is created. Defaults to 3800.
        combine_text_under_n_chars (int): The number of characters under which text is combined into a single chunk. Defaults to 2000.
    """
    return partition_pdf(
        filename=filepath,
        extract_images_in_pdf=True, 
        infer_table_structure=True,  
        chunking_strategy="by_title",  #"by_page"
        #page_numbers=list(range(1, 7)),  # 1~6 페이지 명시
        max_characters=4000,  
        new_after_n_chars=3800, 
        combine_text_under_n_chars=2000, 
        #batch_size=10,  # 한 번에 처리할 페이지 수 (메모리 사용량 조절)
    )


# -------------------------------------------------------------------
# 3) 프롬프트 템플릿 정의
# -------------------------------------------------------------------
prompt = PromptTemplate.from_template(
    """Context information is below. You are only aware of this context and nothing else.
---------------------

{context}

---------------------
Given this context, generate only questions based on the below query.
You are an Teacher/Professor in {domain}. 
Your task is to provide exactly **{num_questions}** question(s) for an upcoming quiz/examination. 
You are not to provide more or less than this number of questions. 
The question(s) should be diverse in nature across the document. 
The purpose of question(s) is to test the understanding of the students on the context information provided.
You must also provide the answer to each question. The answer should be based on the context information provided only.

Restrict the question(s) to the context information provided only.
QUESTION and ANSWER should be written in Korean. response in JSON format which contains the `question` and `answer`.
DO NOT USE List in JSON format.
ANSWER should be a complete sentence.

#Format:
```json
{{
    "QUESTION": "테슬라가 공개한 차세대 로봇 '옵티머스 2.0'의 핵심 개선점 중 하나는 무엇입니까?",
    "ANSWER": "테슬라가 공개한 차세대 로봇 옵티머스 2.0의 핵심 개선점은 자체 설계한 근전도 센서를 활용해 정밀한 손동작을 구현한 것입니다."
}},
{{
    "QUESTION": "오픈AI가 발표한 GPT-5 연구 방향에서 가장 강조된 목표는 무엇입니까?",
    "ANSWER": "오픈AI가 발표한 GPT-5 연구 방향에서 가장 강조된 목표는 장기적 추론 능력 향상입니다."
}},
{{
    "QUESTION": "파이낸셜 타임즈 보고서에 따르면 2030년까지 글로벌 양자컴퓨팅 시장 규모는 얼마로 예상되나요?",
    "ANSWER": "파이낸셜 타임즈 보고서에 따르면 2030년까지 글로벌 양자컴퓨팅 시장 규모는 125억 달러로 예상됩니다."
}}
```
"""
)


#-------------------------------------------------------------------
# 4) AWS Bedrock(Claude) 클라이언트 설정
# Get AWS credentials from environment variables
# Extracting Sentences from PDF to JSONL with AWS Bedrock Claude which is a LLM Professor Personality prepared test questions.
#-------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")  # Default to us-east-1 if not specified
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
# Add debugging to check environment variables
print(f"AWS_REGION: {AWS_REGION}")
print(f"Extracting Sentences from PDF to JSONL with AWS Bedrock Claude which is a LLM Professor Personality prepared test questions.")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get('AWS_SESSION_TOKEN')

bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    aws_session_token=AWS_SESSION_TOKEN
)
#-------------------------------------------------------------------
# see https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
#--------------------------------------------------------------------
# Bedrock Claude 모델 설정
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",    
    #model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",    
    #model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
    client=bedrock_client,
    model_kwargs={
        "temperature": 0,
        "max_tokens": 2000,
    },
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

    
    
#-------------------------------------------------------------------
# 5) 체인 구성
# def format_docs(input_dict): """ 체인에 전달할 입력을 표준화합니다. """ 
# def custom_json_parser(response): """ ChatBedrock 모델의 응답(문자열) 중 JSON 포맷 부분을 찾아 파싱합니다. """ 
# chain : Runnable 파이프라인(체인) 정의, RunnablePassthrough -> Prompt -> LLM -> StrOutputParser -> custom_json_parser 
#-------------------------------------------------------------------

def custom_json_parser(response):
    if hasattr(response, 'content'):
        response = response.content
    
    try:
        start = response.find('```json') + 7 if '```json' in response else 0
        end = response.find('```', start) if '```' in response[start:] else len(response)
        json_text = response[start:end].strip()
        json_text = json_text.rstrip(',')
        json_text = f'[{json_text}]'
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 에러: {e}")
        print(f"파싱 시도한 텍스트: {json_text}")
        return []

def format_docs(input_dict):
    return {
        "context": input_dict["context"],
        "domain": input_dict["domain"],
        "num_questions": input_dict["num_questions"]
    }

chain = (
    RunnablePassthrough(format_docs)
    | prompt
    | llm
    | StrOutputParser()
    | custom_json_parser
)


def main():
    
    # Get PDF path from environment variable or use default
    pdf_path = os.environ.get("PDF_PATH", "data/fsi_data.pdf")
    print(f"Processing PDF from: {pdf_path}")
    
    # Get domain from environment variable or use default
    domain = os.environ.get("DOMAIN", "International Finance")
    print(f"Using domain: {domain}")
    
    # Get number of questions from environment variable or use default
    num_questions = os.environ.get("NUM_QUESTIONS", "5")
    print(f"Number of questions to generate: {num_questions}")
    
    
    # 6-1) PDF 파일에서 요소 추출     
    elements = extract_elements_from_pdf(pdf_path)
    print(f"추출된 요소 수: {len(elements)}")
    
    # 6-2) 추출된 요소 각각에 대해 chain 실행
    qa_pair = []
    #for element in elements[1:]:
    for element in elements:
        if element.text:
            response = chain.invoke({
                "context": element.text,                
                "domain": domain,
                #"domain": "AI",
                #"domain": "Finance",
                "num_questions": num_questions
            })
            # 여러 문서 조각 합치기
            qa_pair.extend(response)
            
    # 6-3) JSONL로 결과 저장
    with open("data/qa_pairs.jsonl", "w", encoding='utf-8') as f:
        for item in qa_pair:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("[INFO] QA 생성 완료! data/qa_pairs.jsonl 파일에 저장되었습니다.")


if __name__ == "__main__":
    
    main()