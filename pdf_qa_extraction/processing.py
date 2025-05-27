# processing.py - SageMaker Processing Job용 스크립트
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
import argparse  # 추가: 명령줄 인자 처리를 위한 모듈

"""
SageMaker Processing Job용 PDF 파싱 및 질문 생성 스크립트

이 스크립트는 다음 단계로 이루어집니다:
1) PDF 파티셔닝(텍스트/이미지/테이블 등 추출)
2) Prompt Template 정의
3) AWS Bedrock(Claude) 클라이언트 설정
4) 체인 구성(Runnable 파이프라인)
5) PDF 요소들에 대해 체인 실행 & 질의응답 생성
6) 최종 JSONL 파일로 S3에 저장

processing_local.py와 다른점은 LLM 객체 생성 부분을 전역에서 함수 내부로 이동하고 domain, num_questions, model_id을 인자로 받을 수 있도록 argparse 및 main 함수 수정

"""

# -------------------------------------------------------------------
# 1) SageMaker Processing Job용 경로 설정
# -------------------------------------------------------------------
# SageMaker Processing Job의 기본 경로
INPUT_DIR = '/opt/ml/processing/input'
OUTPUT_DIR = '/opt/ml/processing/output'

# 입출력 파일 경로 설정
PDF_FILE_PATH = os.path.join(INPUT_DIR, 'pdf/fsi_data.pdf')
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'qa_pairs.jsonl')

# 출력 디렉토리가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"입력 디렉토리: {INPUT_DIR}")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print(f"PDF 파일 경로: {PDF_FILE_PATH}")
print(f"출력 파일 경로: {OUTPUT_FILE_PATH}")

# -------------------------------------------------------------------
# 2) PDF 파티셔닝 함수 정의
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
        chunking_strategy="by_title",  #see : https://docs.unstructured.io/api-reference/partition/chunking
        #page_numbers=list(range(1, 7)),  # 1~6 페이지 명시
        max_characters=4000,  
        new_after_n_chars=3800, 
        combine_text_under_n_chars=2000, 
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

# -------------------------------------------------------------------
# 4) 유틸리티 함수 정의
# -------------------------------------------------------------------
# SageMaker는 실행 역할의 권한을 사용하므로 별도의 자격증명이 필요 없음

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
print(f"AWS 리전: {AWS_REGION}")

def custom_json_parser(response):
    """
    응답에서 JSON 형식의 텍스트를 추출하고 파싱합니다.
    
    Args:
        response (str|obj): LLM의 응답
        
    Returns:
        list: 파싱된 JSON 객체 목록
    """
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
    """
    체인에 전달할 입력을 형식화합니다.
    
    Args:
        input_dict (dict): 입력 딕셔너리
        
    Returns:
        dict: 형식화된 입력 딕셔너리
    """
    return {
        "context": input_dict["context"],
        "domain": input_dict["domain"],
        "num_questions": input_dict["num_questions"]
    }







# -------------------------------------------------------------------
# 6) 메인 실행 함수
# -------------------------------------------------------------------
def main(domain="International Finance", num_questions="5", model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"): #for arguments parsing
    print("PDF 질문 생성 작업 시작...")
    print(f"도메인: {domain}, 질문 수: {num_questions}, 모델: {model_id}")
    
    try:
        # AWS Bedrock 클라이언트 설정
        print("AWS Bedrock 클라이언트 설정 중...")
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION
        )
        # Bedrock Claude 모델 설정
        llm = ChatBedrock(
            model_id=model_id,  # 전달받은 model_id 사용            
            client=bedrock_client,
            model_kwargs={
                "temperature": 0,
                "max_tokens": 2000,
            },
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # -------------------------------------------------------------------
        # 5) 체인 구성 : Runnable 파이프라인 정의
        # -------------------------------------------------------------------

        # 체인 구성
        chain = (
            RunnablePassthrough(format_docs)
            | prompt
            | llm
            | StrOutputParser()
            | custom_json_parser
        )
        
        # 1) PDF 파일에서 요소 추출
        elements = extract_elements_from_pdf(PDF_FILE_PATH)
        print(f"추출된 요소 수: {len(elements)}")
        
        # 2) 추출된 요소 각각에 대해 chain 실행
        qa_pairs = []
        for i, element in enumerate(elements):
            if hasattr(element, 'text') and element.text:
                print(f"요소 {i+1}/{len(elements)} 처리 중... (텍스트 길이: {len(element.text)})")
                try:
                    response = chain.invoke({
                        "context": element.text,
                        "domain": "International Finance",  # 외환정책
                        "num_questions": "5"
                    })
                    qa_pairs.extend(response)
                    print(f"요소 {i+1} 처리 완료 - {len(response)}개 QA 쌍 생성")
                except Exception as e:
                    print(f"요소 {i+1} 처리 중 오류 발생: {str(e)}")
        
        print(f"총 {len(qa_pairs)}개의 QA 쌍 생성 완료")
        
        # 3) JSONL로 결과 저장
        with open(OUTPUT_FILE_PATH, "w", encoding='utf-8') as f:
            for item in qa_pairs:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"[INFO] QA 생성 완료! 결과 파일이 {OUTPUT_FILE_PATH}에 저장되었습니다.")
        
    except Exception as e:
        print(f"처리 중 오류 발생: {str(e)}")
        raise




if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='PDF에서 QA 쌍을 생성하는 스크립트')
    parser.add_argument('--domain', type=str, default='International Finance',
                        help='QA 생성을 위한 도메인 (예: "International Finance", "외환정책")')
    parser.add_argument('--num_questions', type=str, default='5',
                        help='각 PDF 요소마다 생성할 질문 수')
    parser.add_argument('--model_id', type=str, default='anthropic.claude-3-5-sonnet-20240620-v1:0',
                        help='사용할 Bedrock 모델 ID')
    
    args = parser.parse_args()
    
    # 파싱된 인자를 main 함수에 전달
    main(domain=args.domain, num_questions=args.num_questions, model_id=args.model_id)
