import json
import os
import base64
import glob

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler

"""
local_processing_openai.py

This script consists of the following steps:

1) Load .env file
2) PDF partitioning (extract text/images/tables, GPU usage, etc.)
3) Define Prompt Templates
4) Set up OpenAI client
5) Configure chain (Runnable pipeline)
6) Execute chain on PDF elements & generate Q&A
7) Save final results to JSONL file

Environment Variables:
  - OPENAI_API_KEY

Libraries Used:
  - dotenv, unstructured, langchain-core, langchain-openai, etc.
"""

# -------------------------------------------------------------------
# 1) Load .env file
# Load environment variables from .env file.
# -------------------------------------------------------------------


load_dotenv()


# -------------------------------------------------------------------
# 2) Define PDF partitioning function
# Extract images, tables, and text chunks from PDF files.
# -------------------------------------------------------------------

def extract_elements_from_pdf(filepath, table_model=None):
    """
    Extracts elements from a PDF file using specified partitioning strategies.

    Args:
        filepath (str): The path to the PDF file to be processed.
        table_model (str, optional): The table detection model to use.
                                   Options: "yolox", "table-transformer", "tatr"
                                   If None, infer_table_structure will be disabled.

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
    # Set table model options
    partition_kwargs = {
        "filename": filepath,
        "extract_images_in_pdf": True,
        "chunking_strategy": "by_title",  #"by_page"
        #"page_numbers": list(range(1, 7)),  # Specify pages 1-6
        "max_characters": 4000,
        "new_after_n_chars": 3800,
        "combine_text_under_n_chars": 2000,
        #"batch_size": 10,  # Number of pages to process at once (control memory usage)
        "extract_image_block_output_dir": "figures",  # Specify image extraction directory
        "strategy":"auto" # Automatically select layout and text extraction method, operates as fast (general pdf) or high_res (object detection layout & ocr for scanned pdf)
    }

    # Enable table structure inference only when table model is specified
    if table_model:
        partition_kwargs["infer_table_structure"] = True
        partition_kwargs["table_model"] = table_model
    else:
        partition_kwargs["infer_table_structure"] = False

    return partition_pdf(**partition_kwargs)

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64.

    Args:
        image_path (str): Path to the image file to encode

    Returns:
        str: Base64 encoded image data
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Image encoding error {image_path}: {e}")
        return None

def get_extracted_images(figures_dir="figures"):
    """
    Find extracted image files in the figures directory.

    Args:
        figures_dir (str): Directory where images are stored

    Returns:
        list: List of image file paths
    """
    if not os.path.exists(figures_dir):
        return []

    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    image_files = []

    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(figures_dir, extension)))

    return sorted(image_files)


# -------------------------------------------------------------------
# 3) Define Prompt Templates
# -------------------------------------------------------------------
# Text prompt template
text_prompt = PromptTemplate.from_template(
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

# Image prompt message generation function
def create_image_prompt_message(image_base64, domain, num_img_questions, image_path=""):
    """
    Generate a prompt message for image analysis.

    Args:
        image_base64 (str): Base64 encoded image data
        domain (str): Field/domain
        num_questions (str): Number of questions to generate
        image_path (str): Image file path (for format detection)

    Returns:
        HumanMessage: Message containing image and text
    """
    # Detect format from image extension
    image_format = "png"  # Default value
    if image_path:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            image_format = "jpeg"
        elif ext in ['.png']:
            image_format = "png"
        elif ext in ['.gif']:
            image_format = "gif"
        elif ext in ['.bmp']:
            image_format = "bmp"

    return HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
Analyze this image and generate question-answer pairs.

You are a professor/teacher in the {domain} field.
Your task is to create exactly **{num_img_questions}** questions for an upcoming quiz/exam.
You must not create more or fewer questions than this number.

**MANDATORY RULES - VIOLATION WILL RESULT IN FAILURE:**
1. **EXACT DATA ONLY**: Use ONLY the exact numbers, dates, and text visible in the image. Do NOT interpret, convert, or modify any values.
2. **PRECISE READING**: Read dates, numbers, and labels character-by-character as they appear. For example, if you see "12.3일", it means December 3rd, NOT November 13th.
3. **NO ASSUMPTIONS**: Do not assume relationships, trends, or meanings beyond what is explicitly shown.
4. **VERIFY BEFORE WRITING**: Before writing each answer, mentally point to the exact location in the image where that information appears.
5. **CONSERVATIVE APPROACH**: If you cannot clearly read a specific value or date, do not create a question about it.

**DATA ACCURACY REQUIREMENTS:**
- Charts/Graphs: Only reference data points where both X-axis (date/time) AND Y-axis (value) are clearly visible
- Tables: Only reference cells where both row and column headers are clear
- Text: Only reference text that is completely legible
- Numbers: Copy numbers exactly as shown (including decimal points, units like bp, %, etc.)

**FORBIDDEN ACTIONS:**
- Converting date formats (e.g., 12.3 ≠ 11.13)
- Estimating values between data points
- Creating questions about unclear or partially visible content
- Using information from chart legends if the actual data is unclear

**Question Types to Focus On:**
- Direct reading of clearly visible data points
- Identification of clearly labeled chart/table elements
- Reading of section titles, page numbers, or menu items
- Comparison of clearly visible values (highest, lowest, specific dates)

Write questions and answers in Korean and respond in JSON format.
Do not use arrays/lists in the JSON format.


#Format:
```json
{{
    "QUESTION": "CDS 프리미엄 차트에서 12월 17일의 수치는 얼마입니까?",
    "ANSWER": "CDS 프리미엄 차트에서 12월 17일의 수치는 36.3bp입니다."
}},
{{
    "QUESTION": "목차에서 개선 방안의 첫 번째 항목은 무엇입니까?",
    "ANSWER": "목차에서 개선 방안의 첫 번째 항목은 '건전성 규제 완화'입니다."
}},
{{
    "QUESTION": "9월 전후 지표 악화 차트에서 외환 차익거래유인 최고점은 언제 기록되었습니까?",
    "ANSWER": "9월 전후 지표 악화 차트에서 외환 차익거래유인 최고점은 10월 2일경에 기록되었습니다."
}}
```
"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_format};base64,{image_base64}"
            }
        }
    ])


#-------------------------------------------------------------------
# 4) Set up OpenAI client
# Get OpenAI API key from environment variables
# Extracting Sentences from PDF to JSONL with OpenAI which is a LLM Professor Personality prepared test questions.
#-------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
print(f"Extracting Sentences from PDF to JSONL with OpenAI which is a LLM Professor Personality prepared test questions.")

#-------------------------------------------------------------------
# see https://platform.openai.com/docs/models
#--------------------------------------------------------------------
# OpenAI model setup
llm = ChatOpenAI(
    model="gpt-4o",  # or "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo" etc.
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=2000,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)



#-------------------------------------------------------------------
# 5) Configure chain
# def format_docs(input_dict): """ Standardize input passed to the chain. """
# def custom_json_parser(response): """ Find and parse JSON format part in LLM model's response (string). """
# chain : Define Runnable pipeline (chain), RunnablePassthrough -> Prompt -> LLM -> StrOutputParser -> custom_json_parser
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
        print(f"JSON parsing error: {e}")
        print(f"Attempted to parse text: {json_text}")
        return []

def format_docs(input_dict):
    return {
        "context": input_dict["context"],
        "domain": input_dict["domain"],
        "num_questions": input_dict["num_questions"]
    }

# Text processing chain
text_chain = (
    RunnablePassthrough(format_docs)
    | text_prompt
    | llm
    | StrOutputParser()
    | custom_json_parser
)

# Image processing function
def process_image_with_llm(image_path, domain, num_img_questions):
    """
    Process image with LLM to generate Q&A

    Args:
        image_path (str): Image file path
        domain (str): Field/domain
        num_img_questions (str): Number of questions to generate

    Returns:
        list: List of generated Q&A pairs
    """
    try:
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return []

        # Generate image prompt message
        message = create_image_prompt_message(image_base64, domain, num_img_questions, image_path)

        # Process image with LLM
        response = llm.invoke([message])

        # Parse response to JSON
        parsed_response = custom_json_parser(response)

        # Add image source information
        for qa in parsed_response:
            qa['source'] = 'image'
            qa['image_path'] = os.path.basename(image_path)

        print(f"Image processing completed: {os.path.basename(image_path)} - {len(parsed_response)} Q&A generated")
        return parsed_response

    except Exception as e:
        print(f"Image processing error {image_path}: {e}")
        return []


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

    # Get number of image questions from environment variable or use default
    num_img_questions = os.environ.get("NUM_IMG_QUESTIONS", "1")

    # Get table model from environment variable or use default (None)
    table_model = os.environ.get("TABLE_MODEL", None)
    print(f"Table model: {table_model if table_model else 'None (disabled)'}")

    # 6-1) Extract elements from PDF file
    elements = extract_elements_from_pdf(pdf_path, table_model=table_model)
    print(f"Number of extracted elements: {len(elements)}")

    # 6-2) Execute text_chain on each extracted text element
    qa_pairs = []
    text_count = 0

    print("\n=== Starting text element processing ===")
    for element in elements:
        if element.text and element.text.strip():  # Exclude empty text
            try:
                response = text_chain.invoke({
                    "context": element.text,
                    "domain": domain,
                    "num_questions": num_questions
                })
                qa_pairs.extend(response)
                text_count += 1
                print(f"Text element {text_count} processed - {len(response)} Q&A generated")
            except Exception as e:
                print(f"Text element processing error: {e}")

    print(f"Text processing completed: {len(qa_pairs)} Q&A generated from {text_count} elements")

    # 6-3) Process extracted images
    print("\n=== Starting image element processing ===")
    image_files = get_extracted_images("figures")
    image_count = 0

    if image_files:
        print(f"Image files found: {len(image_files)}")
        for image_path in image_files:
            image_qa = process_image_with_llm(image_path, domain, num_img_questions)
            qa_pairs.extend(image_qa)
            if image_qa:
                image_count += 1
    else:
        print("No extracted images.")

    print(f"Image processing completed: {sum(1 for qa in qa_pairs if qa.get('source') == 'image')} Q&A generated from {image_count} images")

    # 6-4) Save results to JSONL
    os.makedirs("data", exist_ok=True)  # Create data directory if it doesn't exist

    with open("data/qa_pairs.jsonl", "w", encoding='utf-8') as f:
        for item in qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n[INFO] Q&A generation completed! Total {len(qa_pairs)} Q&A saved to data/qa_pairs.jsonl file.")
    print(f"- Generated from text: {len([qa for qa in qa_pairs if qa.get('source') != 'image'])}")
    print(f"- Generated from images: {len([qa for qa in qa_pairs if qa.get('source') == 'image'])}")


if __name__ == "__main__":

    main()
