"""AWS Bedrock (Claude) provider.

Credentials resolution:
  * If ``AWS_ACCESS_KEY_ID`` is set (local / .env), explicit keys are used.
  * Otherwise boto3's default chain is used (e.g. the SageMaker execution role).
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..extract import encode_image_to_base64
from ..parsing import custom_json_parser
from ..prompts import build_image_instruction, build_text_prompt, detect_image_format
from .base import LLMProvider

_DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"


class BedrockProvider(LLMProvider):
    """Generate Q&A pairs with Anthropic Claude via Amazon Bedrock."""

    name = "aws-bedrock"

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        streaming: bool = True,
    ) -> None:
        import boto3
        from langchain_aws import ChatBedrock
        from langchain_core.callbacks import StreamingStdOutCallbackHandler

        self.model_id = model_id or os.getenv("MODEL_ID") or _DEFAULT_MODEL
        region = region_name or os.getenv("AWS_REGION", "us-east-1")

        client_kwargs = {"service_name": "bedrock-runtime", "region_name": region}
        if os.getenv("AWS_ACCESS_KEY_ID"):
            client_kwargs.update(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            )
        bedrock_client = boto3.client(**client_kwargs)

        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        self._llm = ChatBedrock(
            model_id=self.model_id,
            client=bedrock_client,
            model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
            streaming=streaming,
            callbacks=callbacks,
        )
        print(f"[BedrockProvider] region={region} model={self.model_id}")

    def generate_text_qa(
        self,
        context: str,
        domain: str,
        num_questions: str,
        persona: str = "professor",
        language: str = "auto",
    ) -> List[dict]:
        prompt = build_text_prompt(context, domain, num_questions, persona, language)
        response = self._llm.invoke(prompt)
        return custom_json_parser(response)

    def generate_image_qa(
        self,
        image_path: str,
        domain: str,
        num_img_questions: str,
        persona: str = "professor",
        context: str = "",
        language: str = "auto",
    ) -> List[dict]:
        from langchain_core.messages import HumanMessage

        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return []

        image_format = detect_image_format(image_path)
        instruction = build_image_instruction(
            domain, num_img_questions, persona, context, language
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": instruction},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image_format}",
                        "data": image_base64,
                    },
                },
            ]
        )
        try:
            response = self._llm.invoke([message])
            parsed = custom_json_parser(response)
            self.tag_image_source(parsed, image_path)
            print(
                f"이미지 처리 완료: {os.path.basename(image_path)} - {len(parsed)}개 Q&A 생성"
            )
            return parsed
        except Exception as exc:
            print(f"이미지 처리 에러 {image_path}: {exc}")
            return []
