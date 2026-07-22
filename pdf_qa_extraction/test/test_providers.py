"""Provider-factory + auth wiring tests (no heavy SDKs required).

These exercise the parts that must work without ``langchain_ollama`` /
``azure-identity`` / ``langchain-openai`` installed: the provider registry,
aliases, and the Azure Foundry Entra ID token-scope helper. Concrete provider
construction is covered by the container smoke test in CI.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from pdf_qa.providers import _ALIASES, get_provider


def test_unknown_provider_lists_ollama():
    with pytest.raises(ValueError) as exc:
        get_provider("does-not-exist")
    msg = str(exc.value)
    for name in ("azure", "bedrock", "openai", "ollama"):
        assert name in msg


def test_aliases_resolve_ollama_and_azure():
    assert _ALIASES["ollama"] == "ollama"
    assert _ALIASES["local"] == "ollama"
    # Azure aliases all fold to the single Foundry backend.
    for a in ("azure", "foundry", "azure_openai", "azure-openai", "azureopenai"):
        assert _ALIASES[a] == "azure_foundry"


def test_azure_token_scope_default_and_override():
    from pdf_qa.providers import azure_foundry as af

    os.environ.pop("AZURE_OPENAI_TOKEN_SCOPE", None)
    assert af._token_scope() == "https://cognitiveservices.azure.com/.default"
    os.environ["AZURE_OPENAI_TOKEN_SCOPE"] = "https://custom.example/.default"
    try:
        assert af._token_scope() == "https://custom.example/.default"
    finally:
        os.environ.pop("AZURE_OPENAI_TOKEN_SCOPE", None)


def test_ollama_provider_reads_env(monkeypatch):
    # Skip cleanly when the optional langchain-ollama extra isn't installed.
    pytest.importorskip("langchain_ollama")
    from pdf_qa.providers.ollama import OllamaProvider

    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.delenv("OLLAMA_VISION_MODEL", raising=False)
    provider = OllamaProvider(streaming=False)
    assert provider.model_id == "qwen2.5"
    assert provider.base_url == "http://ollama.internal:11434"
    assert provider.name == "ollama"
    # With no vision override, the text model is reused for images (one client).
    assert provider.vision_model_id == "qwen2.5"
    assert provider._vision_llm is provider._llm


def test_ollama_separate_vision_model(monkeypatch):
    pytest.importorskip("langchain_ollama")
    from pdf_qa.providers.ollama import OllamaProvider

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1")
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "qwen2.5vl")
    provider = OllamaProvider(streaming=False)
    # Text and image Q&A use distinct clients / model tags.
    assert provider.model_id == "llama3.1"
    assert provider.vision_model_id == "qwen2.5vl"
    assert provider._vision_llm is not provider._llm
    assert provider._llm.model == "llama3.1"
    assert provider._vision_llm.model == "qwen2.5vl"
