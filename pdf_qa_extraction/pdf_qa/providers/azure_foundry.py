"""Azure AI Foundry provider - the Azure-first default backend.

Two interchangeable modes (env ``AZURE_MODE``):

* ``openai`` (default): calls an **Azure OpenAI** deployment through
  ``AzureChatOpenAI``. Rock-solid for text and vision.
* ``agent``: calls the **Azure AI Foundry Agent Service** - a persistent,
  versioned "professor" agent - through ``azure-ai-projects`` /
  ``azure-ai-agents``. Showcases Foundry's agentic capabilities.

Authentication - **Entra ID first, one-shot bring-up**
------------------------------------------------------
Keyless **Microsoft Entra ID** is the default and recommended path: when no
``AZURE_OPENAI_API_KEY`` is set, a single process-wide
``DefaultAzureCredential`` (Managed Identity on Azure, ``az login`` locally) is
resolved once and **shared** by both the Azure OpenAI token provider and the
Foundry Agent client - so one sign-in / token cache boots the whole app in one
go. Set ``AZURE_OPENAI_API_KEY`` to opt into API-key auth instead. The token
scope can be overridden with ``AZURE_OPENAI_TOKEN_SCOPE``.

Required environment variables
------------------------------
openai mode:
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT (or MODEL_ID),
  AZURE_OPENAI_API_VERSION (optional), AZURE_OPENAI_API_KEY (optional -> keyless)
agent mode:
  AZURE_AI_PROJECT_ENDPOINT, AZURE_AI_AGENT_MODEL (or MODEL_ID)
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..extract import encode_image_to_base64
from ..parsing import custom_json_parser
from ..prompts import build_image_instruction, build_text_prompt, detect_image_format
from .base import LLMProvider

_DEFAULT_DEPLOYMENT = "gpt-4o"
_DEFAULT_API_VERSION = "2024-10-21"
# Entra ID token audience for Azure OpenAI / Cognitive Services.
_DEFAULT_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"

# Persona-neutral base instruction for the Foundry Agent; the per-request prompt
# carries the concrete role/persona (see pdf_qa.prompts) and the task/format.
_AGENT_INSTRUCTIONS = (
    "You generate high-quality Q&A pairs from the material the user provides. "
    "Adopt the role and follow the formatting instructions in each user message "
    "exactly, always answer in Korean, and return only the requested JSON block."
)

# Process-wide credential, resolved lazily and reused everywhere.
_SHARED_CREDENTIAL = None


def azure_credential():
    """Return a single, process-wide ``DefaultAzureCredential`` (Entra ID).

    Reused by both the Azure OpenAI token provider and the Foundry Agent client
    so one sign-in / Managed Identity token cache serves the whole app. This is
    what lets the provider come up keyless in a single shot.
    """
    global _SHARED_CREDENTIAL
    if _SHARED_CREDENTIAL is None:
        from azure.identity import DefaultAzureCredential

        _SHARED_CREDENTIAL = DefaultAzureCredential()
    return _SHARED_CREDENTIAL


def _token_scope() -> str:
    return os.getenv("AZURE_OPENAI_TOKEN_SCOPE", _DEFAULT_TOKEN_SCOPE)


def _bearer_token_provider():
    """Entra ID bearer-token provider bound to the shared credential + scope."""
    from azure.identity import get_bearer_token_provider

    return get_bearer_token_provider(azure_credential(), _token_scope())


class AzureFoundryProvider(LLMProvider):
    """Generate Q&A pairs with Azure AI Foundry (Azure OpenAI or Agent Service)."""

    name = "azure-foundry"

    def __init__(
        self,
        model_id: Optional[str] = None,
        mode: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        streaming: bool = True,
    ) -> None:
        self.mode = (mode or os.getenv("AZURE_MODE", "openai")).strip().lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.auth_mode = "entra-id"

        if self.mode == "agent":
            self.model_id = (
                model_id or os.getenv("AZURE_AI_AGENT_MODEL") or os.getenv("MODEL_ID")
                or _DEFAULT_DEPLOYMENT
            )
            self._agent = _FoundryAgentBackend(self.model_id, _AGENT_INSTRUCTIONS)
            self.name = "azure-foundry-agent"
        else:
            self.model_id = (
                model_id or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("MODEL_ID")
                or _DEFAULT_DEPLOYMENT
            )
            self._llm = self._build_azure_openai(streaming)
            self.name = "azure-foundry-openai"

    # ------------------------------------------------------------------
    # openai mode
    # ------------------------------------------------------------------
    def _build_azure_openai(self, streaming: bool):
        from langchain_core.callbacks import StreamingStdOutCallbackHandler
        from langchain_openai import AzureChatOpenAI

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            # Fail fast with an actionable message so bring-up is one-shot.
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT is required for the Azure Foundry provider "
                "(e.g. https://<resource>.openai.azure.com/). Set it in your .env "
                "or environment."
            )
        api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or _DEFAULT_API_VERSION
        )
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        common = dict(
            azure_deployment=self.model_id,
            azure_endpoint=endpoint,
            api_version=api_version,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=streaming,
            callbacks=callbacks,
        )

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if api_key:
            self.auth_mode = "api-key"
            print(
                f"[AzureFoundryProvider:openai] endpoint={endpoint} "
                f"deployment={self.model_id} auth=api-key"
            )
            return AzureChatOpenAI(api_key=api_key, **common)

        # Keyless: shared Entra ID credential (Managed Identity / az login).
        self.auth_mode = "entra-id"
        print(
            f"[AzureFoundryProvider:openai] endpoint={endpoint} "
            f"deployment={self.model_id} auth=entra-id "
            f"(DefaultAzureCredential, scope={_token_scope()})"
        )
        return AzureChatOpenAI(azure_ad_token_provider=_bearer_token_provider(), **common)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def generate_text_qa(
        self, context: str, domain: str, num_questions: str, persona: str = "professor"
    ) -> List[dict]:
        prompt = build_text_prompt(context, domain, num_questions, persona)
        if self.mode == "agent":
            return custom_json_parser(self._agent.run_text(prompt))
        return custom_json_parser(self._llm.invoke(prompt))

    def generate_image_qa(
        self,
        image_path: str,
        domain: str,
        num_img_questions: str,
        persona: str = "professor",
        context: str = "",
    ) -> List[dict]:
        instruction = build_image_instruction(domain, num_img_questions, persona, context)
        try:
            if self.mode == "agent":
                raw = self._agent.run_image(instruction, image_path)
                parsed = custom_json_parser(raw)
            else:
                parsed = custom_json_parser(self._invoke_vision(instruction, image_path))
            self.tag_image_source(parsed, image_path)
            print(
                f"이미지 처리 완료: {os.path.basename(image_path)} - {len(parsed)}개 Q&A 생성"
            )
            return parsed
        except Exception as exc:
            print(f"이미지 처리 에러 {image_path}: {exc}")
            return []

    def _invoke_vision(self, instruction: str, image_path: str):
        from langchain_core.messages import HumanMessage

        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return ""
        image_format = detect_image_format(image_path)
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
        return self._llm.invoke([message])

    def close(self) -> None:
        """Release Foundry Agent resources (no-op in openai mode)."""
        if self.mode == "agent":
            self._agent.close()


class _FoundryAgentBackend:
    """Thin wrapper over the Azure AI Foundry Agent Service.

    A single agent is created and reused; each request runs on a fresh thread so
    Q&A generation stays stateless. Requires ``azure-ai-projects`` and
    ``azure-ai-agents``.
    """

    def __init__(self, model: str, instructions: str) -> None:
        from azure.ai.projects import AIProjectClient

        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or os.getenv("PROJECT_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "AZURE_AI_PROJECT_ENDPOINT is required when AZURE_MODE=agent"
            )
        # Same shared Entra ID credential as the openai path -> one-shot bring-up.
        print(
            f"[AzureFoundryProvider:agent] endpoint={endpoint} model={model} "
            f"auth=entra-id (DefaultAzureCredential)"
        )
        self._project = AIProjectClient(
            endpoint=endpoint, credential=azure_credential()
        )
        self._agents = self._project.agents
        self._agent = self._agents.create_agent(
            model=model,
            name="pdf-qa-professor",
            instructions=instructions,
        )

    def _run(self, content) -> str:
        from azure.ai.agents.models import ListSortOrder

        thread = self._agents.threads.create()
        self._agents.messages.create(thread_id=thread.id, role="user", content=content)
        run = self._agents.runs.create_and_process(
            thread_id=thread.id, agent_id=self._agent.id
        )
        if getattr(run, "status", None) == "failed":
            print(f"Foundry Agent run 실패: {getattr(run, 'last_error', None)}")
            return ""

        messages = self._agents.messages.list(
            thread_id=thread.id, order=ListSortOrder.ASCENDING
        )
        answer = ""
        for message in messages:
            if getattr(message, "role", None) == "assistant" and message.text_messages:
                answer = message.text_messages[-1].text.value
        return answer

    def run_text(self, prompt: str) -> str:
        return self._run(prompt)

    def run_image(self, instruction: str, image_path: str) -> str:
        from azure.ai.agents.models import (
            FilePurpose,
            MessageImageFileParam,
            MessageInputImageFileBlock,
            MessageInputTextBlock,
        )

        uploaded = self._agents.files.upload_and_poll(
            file_path=image_path, purpose=FilePurpose.AGENTS
        )
        content = [
            MessageInputTextBlock(text=instruction),
            MessageInputImageFileBlock(
                image_file=MessageImageFileParam(file_id=uploaded.id, detail="high")
            ),
        ]
        try:
            return self._run(content)
        finally:
            try:
                self._agents.files.delete(uploaded.id)
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._agents.delete_agent(self._agent.id)
        except Exception:
            pass
