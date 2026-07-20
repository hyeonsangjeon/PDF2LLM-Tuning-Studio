"""Ollama provider - run the Q&A generation against a **local** model.

Ollama (https://ollama.com) serves open models (Llama, Qwen, Gemma, Mistral,
llava ...) over a local HTTP endpoint, so this backend needs **no cloud
credentials** - ideal for the single-node demo and fully offline runs.

Configuration
-------------
* ``OLLAMA_MODEL`` (or ``MODEL_ID``): model tag, default ``llama3.1``. Use a
  multimodal tag (e.g. ``llama3.2-vision`` / ``llava``) for image Q&A.
* ``OLLAMA_BASE_URL`` (or ``OLLAMA_HOST``): server URL, default
  ``http://localhost:11434``.

Requires the ``ollama`` extra (``pip install .[ollama]`` -> ``langchain-ollama``)
and a running Ollama server with the model pulled (``ollama pull llama3.1``).
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..extract import encode_image_to_base64
from ..parsing import custom_json_parser
from ..prompts import build_image_instruction, build_text_prompt, detect_image_format
from .base import LLMProvider

_DEFAULT_MODEL = "llama3.1"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Generate Q&A pairs with a local Ollama model (no credentials)."""

    name = "ollama"

    def __init__(
        self,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        num_predict: int = 2000,
        streaming: bool = True,
    ) -> None:
        from langchain_core.callbacks import StreamingStdOutCallbackHandler
        from langchain_ollama import ChatOllama

        self.model_id = model_id or os.getenv("OLLAMA_MODEL") or os.getenv("MODEL_ID") or _DEFAULT_MODEL
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or _DEFAULT_BASE_URL
        )
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        self._llm = ChatOllama(
            model=self.model_id,
            base_url=self.base_url,
            temperature=temperature,
            num_predict=num_predict,
            callbacks=callbacks,
        )
        print(f"[OllamaProvider] base_url={self.base_url} model={self.model_id}")

    def generate_text_qa(
        self, context: str, domain: str, num_questions: str, persona: str = "professor"
    ) -> List[dict]:
        prompt = build_text_prompt(context, domain, num_questions, persona)
        response = self._llm.invoke(prompt)
        return custom_json_parser(response)

    def generate_image_qa(
        self, image_path: str, domain: str, num_img_questions: str, persona: str = "professor"
    ) -> List[dict]:
        from langchain_core.messages import HumanMessage

        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return []

        image_format = detect_image_format(image_path)
        instruction = build_image_instruction(domain, num_img_questions, persona)
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
