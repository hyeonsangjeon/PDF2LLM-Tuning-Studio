"""OpenAI (platform.openai.com) provider.

Uses the multimodal ``gpt-4o`` family by default. Vision is delivered as a
base64 ``data:`` URL, which is also the format Azure OpenAI understands.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..extract import encode_image_to_base64
from ..parsing import custom_json_parser
from ..prompts import build_image_instruction, build_text_prompt, detect_image_format
from .base import LLMProvider

_DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(LLMProvider):
    """Generate Q&A pairs with OpenAI chat models."""

    name = "openai"

    def __init__(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        streaming: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        from langchain_core.callbacks import StreamingStdOutCallbackHandler
        from langchain_openai import ChatOpenAI

        self.model_id = model_id or os.getenv("OPENAI_MODEL") or _DEFAULT_MODEL
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        self._llm = ChatOpenAI(
            model=self.model_id,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            callbacks=callbacks,
        )
        print(f"[OpenAIProvider] model={self.model_id}")

    def generate_text_qa(
        self, context: str, domain: str, num_questions: str
    ) -> List[dict]:
        prompt = build_text_prompt(context, domain, num_questions)
        response = self._llm.invoke(prompt)
        return custom_json_parser(response)

    def generate_image_qa(
        self, image_path: str, domain: str, num_img_questions: str
    ) -> List[dict]:
        from langchain_core.messages import HumanMessage

        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return []

        image_format = detect_image_format(image_path)
        instruction = build_image_instruction(domain, num_img_questions)
        message = HumanMessage(
            content=[
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{image_base64}"
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
