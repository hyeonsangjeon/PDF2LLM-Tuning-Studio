"""Provider registry / factory.

``get_provider("azure")`` returns a ready-to-use backend. Concrete provider
modules (and their SDKs) are imported lazily so selecting one backend never
requires the others' dependencies to be installed.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMProvider

# Map user-facing names (env: LLM_PROVIDER) to concrete implementations.
_ALIASES = {
    "azure": "azure_foundry",
    "foundry": "azure_foundry",
    "azure_openai": "azure_foundry",
    "azure-openai": "azure_foundry",
    "azureopenai": "azure_foundry",
    "bedrock": "bedrock",
    "aws": "bedrock",
    "openai": "openai",
}


def get_provider(name: Optional[str] = None, config=None, **kwargs) -> LLMProvider:
    """Instantiate an :class:`LLMProvider`.

    Args:
        name: ``azure`` (default) | ``bedrock`` | ``openai`` and aliases. Falls
            back to the ``LLM_PROVIDER`` env var, then ``azure``.
        config: Optional :class:`~pdf_qa.config.QAConfig`; its ``model_id`` is
            used when not passed explicitly.
    """
    name = (name or os.getenv("LLM_PROVIDER", "azure")).strip().lower()
    key = _ALIASES.get(name)
    if key is None:
        valid = "azure | bedrock | openai"
        raise ValueError(f"Unknown LLM provider '{name}'. Valid options: {valid}")

    if "model_id" not in kwargs and config is not None:
        kwargs["model_id"] = getattr(config, "model_id", None)

    if key == "azure_foundry":
        from .azure_foundry import AzureFoundryProvider

        return AzureFoundryProvider(**kwargs)
    if key == "bedrock":
        from .bedrock import BedrockProvider

        return BedrockProvider(**kwargs)
    if key == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(**kwargs)

    raise ValueError(f"Unhandled provider key: {key}")  # pragma: no cover


__all__ = ["LLMProvider", "get_provider"]
