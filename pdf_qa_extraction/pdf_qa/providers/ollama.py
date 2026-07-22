"""Ollama provider - run the Q&A generation against a **local** model.

Ollama (https://ollama.com) serves open models (Llama, Qwen, Gemma, Mistral,
llava ...) over a local HTTP endpoint, so this backend needs **no cloud
credentials** - ideal for the single-node demo and fully offline runs.

Configuration
-------------
* ``OLLAMA_MODEL`` (or ``MODEL_ID``): text model tag, default ``llama3.1``.
* ``OLLAMA_VISION_MODEL``: **separate multimodal tag used only for figure/chart
  Q&A** so you can pair a strong text model with a dedicated vision model, e.g.
  ``OLLAMA_MODEL=llama3.1`` + ``OLLAMA_VISION_MODEL=qwen2.5vl``. When unset, the
  text model is reused for images (so a single multimodal tag like
  ``llama3.2-vision`` / ``llava`` still works for both).
* ``OLLAMA_BASE_URL`` (or ``OLLAMA_HOST``): server URL, default
  ``http://localhost:11434``.

Multimodal (vision) Ollama tags that read charts/tables well: ``qwen2.5vl`` and
``minicpm-v`` (excellent OCR + chart/table + Korean), ``llama3.2-vision``,
``llava`` / ``llava-llama3``, ``bakllava``, ``moondream``. (Microsoft's *MAI*
models are text-only and not in the Ollama library — use them via the ``azure``
provider on Foundry instead.)

Requires the ``ollama`` extra (``pip install .[ollama]`` -> ``langchain-ollama``)
and a running Ollama server with the model(s) pulled (``ollama pull llama3.1``,
``ollama pull qwen2.5vl``).
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
    """Generate Q&A pairs with a local Ollama model (no credentials).

    Text Q&A uses ``model_id``; figure/chart Q&A uses ``vision_model_id`` when
    given (env ``OLLAMA_VISION_MODEL``), otherwise it falls back to ``model_id``.
    """

    name = "ollama"

    def __init__(
        self,
        model_id: Optional[str] = None,
        vision_model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        num_predict: int = 2000,
        streaming: bool = True,
    ) -> None:
        from langchain_core.callbacks import StreamingStdOutCallbackHandler
        from langchain_ollama import ChatOllama

        self.model_id = model_id or os.getenv("OLLAMA_MODEL") or os.getenv("MODEL_ID") or _DEFAULT_MODEL
        # A dedicated multimodal model for images; reuse the text model if unset.
        self.vision_model_id = (
            vision_model_id or os.getenv("OLLAMA_VISION_MODEL") or self.model_id
        )
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or _DEFAULT_BASE_URL
        )
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None

        def _client(model: str):
            return ChatOllama(
                model=model,
                base_url=self.base_url,
                temperature=temperature,
                num_predict=num_predict,
                callbacks=callbacks,
            )

        self._llm = _client(self.model_id)
        # Only spin up a second client when the vision model actually differs.
        self._vision_llm = (
            self._llm
            if self.vision_model_id == self.model_id
            else _client(self.vision_model_id)
        )
        vision_note = (
            f" vision={self.vision_model_id}"
            if self.vision_model_id != self.model_id
            else ""
        )
        print(
            f"[OllamaProvider] base_url={self.base_url} model={self.model_id}{vision_note}"
        )

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
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{image_base64}"
                    },
                },
            ]
        )
        try:
            response = self._vision_llm.invoke([message])
            parsed = custom_json_parser(response)
            self.tag_image_source(parsed, image_path)
            print(
                f"이미지 처리 완료: {os.path.basename(image_path)} - {len(parsed)}개 Q&A 생성"
            )
            return parsed
        except Exception as exc:
            print(f"이미지 처리 에러 {image_path}: {exc}")
            return []
