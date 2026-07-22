"""PDF2LLM Tuning Studio - PDF -> Q&A extraction core.

A cloud-agnostic pipeline that extracts elements from PDFs with `unstructured`
and generates Q&A pairs through a pluggable LLM provider (Azure AI Foundry,
AWS Bedrock, OpenAI, or a local Ollama model). The output language is locked
via ``OUTPUT_LANGUAGE`` (``auto`` matches the source document), and every
generated pair is validated + de-duplicated before the dataset is written.
"""

from __future__ import annotations

from .api import extract_qa
from .config import QAConfig
from .device import DeviceReport, probe_device
from .layout import DocumentLayout, FigureContext, TextChunk, build_document_layout
from .pipeline import generate_qa_pairs, run_pipeline, save_jsonl
from .prompts import (
    PERSONAS,
    Persona,
    get_persona,
    language_directive,
    list_personas,
    reload_personas,
    resolve_language,
)
from .providers import get_provider
from .providers.base import LLMProvider
from .settings import (
    load_settings,
    provider_configured,
    render_dotenv_example,
    validate_env,
)
from .validate import clean_qa_pairs, validate_pair

__all__ = [
    "QAConfig",
    "LLMProvider",
    "get_provider",
    "extract_qa",
    "run_pipeline",
    "generate_qa_pairs",
    "save_jsonl",
    "probe_device",
    "DeviceReport",
    "Persona",
    "PERSONAS",
    "get_persona",
    "list_personas",
    "reload_personas",
    "resolve_language",
    "language_directive",
    "DocumentLayout",
    "FigureContext",
    "TextChunk",
    "build_document_layout",
    "load_settings",
    "validate_env",
    "provider_configured",
    "render_dotenv_example",
    "clean_qa_pairs",
    "validate_pair",
]

__version__ = "0.2.0"
