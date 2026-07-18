"""PDF2LLM Tuning Studio - PDF -> Q&A extraction core.

A cloud-agnostic pipeline that extracts elements from PDFs with `unstructured`
and generates Korean Q&A pairs through a pluggable LLM provider
(Azure AI Foundry, AWS Bedrock, or OpenAI).
"""

from __future__ import annotations

from .config import QAConfig
from .device import DeviceReport, probe_device
from .pipeline import generate_qa_pairs, run_pipeline, save_jsonl
from .prompts import PERSONAS, Persona, get_persona, list_personas
from .providers import get_provider
from .providers.base import LLMProvider

__all__ = [
    "QAConfig",
    "LLMProvider",
    "get_provider",
    "run_pipeline",
    "generate_qa_pairs",
    "save_jsonl",
    "probe_device",
    "DeviceReport",
    "Persona",
    "PERSONAS",
    "get_persona",
    "list_personas",
]

__version__ = "0.2.0"
