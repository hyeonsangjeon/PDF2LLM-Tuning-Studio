"""Abstract interface every LLM backend implements.

A provider turns a chunk of context (text) or an image into a list of
``{"QUESTION": ..., "ANSWER": ...}`` dictionaries. Keeping this contract tiny is
what lets the pipeline stay identical across Azure Foundry, Bedrock and OpenAI.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List


class LLMProvider(ABC):
    """Base class for pluggable Q&A generation backends."""

    #: Short, human-readable provider name (overridden by subclasses).
    name: str = "base"

    @abstractmethod
    def generate_text_qa(
        self, context: str, domain: str, num_questions: str
    ) -> List[dict]:
        """Generate Q&A pairs from a text chunk."""

    @abstractmethod
    def generate_image_qa(
        self, image_path: str, domain: str, num_img_questions: str
    ) -> List[dict]:
        """Generate Q&A pairs from a single image file."""

    @staticmethod
    def tag_image_source(qa_list: List[dict], image_path: str) -> List[dict]:
        """Annotate image-derived Q&A pairs with their source metadata."""
        for qa in qa_list:
            qa["source"] = "image"
            qa["image_path"] = os.path.basename(image_path)
        return qa_list
