"""Robust extraction of the JSON Q&A block from an LLM response.

Shared by every provider so parsing behaviour is identical regardless of the
backend (Azure Foundry, Bedrock, OpenAI, ...).
"""

from __future__ import annotations

import json
from typing import Any, List


def custom_json_parser(response: Any) -> List[dict]:
    """Find the ```json ...``` block in ``response`` and parse it to a list.

    Accepts either a raw string or an object exposing a ``.content`` attribute
    (e.g. a LangChain message). Always returns a list; on any failure it logs
    the problem and returns an empty list instead of raising.
    """
    if hasattr(response, "content"):
        response = response.content

    if not isinstance(response, str) or not response.strip():
        return []

    json_text = ""
    try:
        start = response.find("```json") + 7 if "```json" in response else 0
        end = response.find("```", start) if "```" in response[start:] else len(response)
        json_text = response[start:end].strip()
        # Tolerate a trailing comma so the objects can be wrapped into an array.
        json_text = json_text.rstrip(",").strip()
        if not json_text:
            return []

        # The prompt asks the model for comma-separated objects (no enclosing
        # array), but models frequently return a proper JSON array or a single
        # object instead. Accept all three shapes.
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = json.loads(f"[{json_text}]")

        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError as exc:
        print(f"JSON 파싱 에러: {exc}")
        print(f"파싱 시도한 텍스트: {json_text}")
        return []
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"응답 파싱 실패: {exc}")
        return []
