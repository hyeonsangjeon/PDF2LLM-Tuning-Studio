"""Prompt templates and the persona ledger for text/image Q&A generation.

The *content* of the prompt is identical across providers; only the transport
envelope (how an image is attached to a chat message) differs, and that lives
in each provider module. Placeholders are filled with ``str.replace`` so the
literal JSON braces in the few-shot examples need no escaping.

**Personas are managed in a YAML ledger** (``personas.yaml`` next to this
module) so they can be edited/extended without touching Python. Each persona
carries a genuinely different *method* (방식) for turning context into Q&A pairs
-- exam setting, Socratic questioning, advisory framing, technical interview,
analytical synthesis, the Feynman technique -- while the output JSON schema
stays identical so every persona yields a compatible fine-tuning dataset.

Point the ``PERSONA_FILE`` environment variable at your own copy to manage a
custom ledger.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import yaml

# ---------------------------------------------------------------------------
# Personas (loaded from the YAML ledger)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Persona:
    """A role + *method* preset applied to the Q&A prompts.

    Attributes:
        key: Stable identifier used by config/env/CLI (e.g. ``"professor"``).
        label: Short human-readable name (used in logs, Korean-friendly).
        text_role: The ``You are ...`` sentence for the text prompt. May contain
            ``{domain}``.
        artifact: The kind of deliverable the questions belong to (fills
            ``for an upcoming {artifact}``), e.g. ``"quiz/examination"``.
        method: The distinct text methodology -- how to ask questions and how to
            write answers. May contain ``{domain}``.
        image_role: The ``You are ...`` sentence for the image prompt. May
            contain ``{domain}``.
        image_method: The distinct image methodology (persona angle applied
            within the mandatory data-accuracy rules).
    """

    key: str
    label: str
    text_role: str
    artifact: str
    method: str
    image_role: str
    image_method: str


#: Location of the persona ledger; override with the ``PERSONA_FILE`` env var to
#: manage personas outside the package.
DEFAULT_PERSONA_FILE = os.path.join(os.path.dirname(__file__), "personas.yaml")


def _persona_file() -> str:
    return os.environ.get("PERSONA_FILE") or DEFAULT_PERSONA_FILE


_REQUIRED_FIELDS = ("label", "text_role", "artifact", "method", "image_role", "image_method")


def load_personas(path: str | None = None) -> Tuple[Dict[str, Persona], str]:
    """Read the YAML ledger and return ``(personas, default_key)``.

    Raises a clear ``ValueError`` on a malformed ledger so misconfiguration
    fails loudly rather than silently dropping personas.
    """
    ledger_path = path or _persona_file()
    with open(ledger_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    raw = data.get("personas")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Persona ledger '{ledger_path}' has no 'personas' mapping.")

    personas: Dict[str, Persona] = {}
    for key, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Persona '{key}' in '{ledger_path}' must be a mapping.")
        missing = [f for f in _REQUIRED_FIELDS if not spec.get(f)]
        if missing:
            raise ValueError(
                f"Persona '{key}' in '{ledger_path}' is missing field(s): {', '.join(missing)}"
            )
        personas[str(key).lower()] = Persona(
            key=str(key).lower(),
            label=str(spec["label"]),
            text_role=str(spec["text_role"]),
            artifact=str(spec["artifact"]),
            method=str(spec["method"]).strip(),
            image_role=str(spec["image_role"]),
            image_method=str(spec["image_method"]).strip(),
        )

    default = str(data.get("default", next(iter(personas)))).lower()
    if default not in personas:
        raise ValueError(
            f"Default persona '{default}' is not defined in '{ledger_path}'."
        )
    return personas, default


# Loaded once at import time; call reload_personas() to pick up edits.
PERSONAS: Dict[str, Persona]
DEFAULT_PERSONA: str
PERSONAS, DEFAULT_PERSONA = load_personas()


def reload_personas(path: str | None = None) -> Dict[str, Persona]:
    """Re-read the ledger (e.g. after editing ``personas.yaml``) and return it."""
    global PERSONAS, DEFAULT_PERSONA
    PERSONAS, DEFAULT_PERSONA = load_personas(path)
    return PERSONAS


def list_personas() -> List[str]:
    """Return the available persona keys, in ledger order."""
    return list(PERSONAS.keys())


def get_persona(persona: Union[str, "Persona", None]) -> Persona:
    """Resolve a persona key (or object) to a :class:`Persona`.

    Falls back to the default persona for ``None``/empty; raises ``ValueError``
    for an unknown key so misconfiguration fails loudly.
    """
    if isinstance(persona, Persona):
        return persona
    key = (persona or DEFAULT_PERSONA).strip().lower()
    if key not in PERSONAS:
        valid = ", ".join(PERSONAS)
        raise ValueError(f"Unknown persona '{persona}'. Valid options: {valid}")
    return PERSONAS[key]


# ---------------------------------------------------------------------------
# Text prompt
# ---------------------------------------------------------------------------
_TEXT_PROMPT = """Context information is below. You are only aware of this context and nothing else.
---------------------

{context}

---------------------
Given this context, generate only questions based on the below query.
{persona_role}
Your task is to provide exactly **{num_questions}** question(s) for an upcoming {artifact}.
You are not to provide more or less than this number of questions.

# Persona method ({persona_label}) - follow this approach:
{persona_method}

The question(s) should be diverse in nature across the document.
You must also provide the answer to each question. The answer should be based on the context information provided only.

Restrict the question(s) to the context information provided only.
QUESTION and ANSWER should be written in Korean. response in JSON format which contains the `question` and `answer`.
DO NOT USE List in JSON format.
ANSWER should be a complete sentence.

#Format:
```json
{
    "QUESTION": "테슬라가 공개한 차세대 로봇 '옵티머스 2.0'의 핵심 개선점 중 하나는 무엇입니까?",
    "ANSWER": "테슬라가 공개한 차세대 로봇 옵티머스 2.0의 핵심 개선점은 자체 설계한 근전도 센서를 활용해 정밀한 손동작을 구현한 것입니다."
},
{
    "QUESTION": "오픈AI가 발표한 GPT-5 연구 방향에서 가장 강조된 목표는 무엇입니까?",
    "ANSWER": "오픈AI가 발표한 GPT-5 연구 방향에서 가장 강조된 목표는 장기적 추론 능력 향상입니다."
},
{
    "QUESTION": "파이낸셜 타임즈 보고서에 따르면 2030년까지 글로벌 양자컴퓨팅 시장 규모는 얼마로 예상되나요?",
    "ANSWER": "파이낸셜 타임즈 보고서에 따르면 2030년까지 글로벌 양자컴퓨팅 시장 규모는 125억 달러로 예상됩니다."
}
```
"""

# ---------------------------------------------------------------------------
# Image prompt (instruction portion; the image itself is attached by provider)
# ---------------------------------------------------------------------------
_IMAGE_INSTRUCTION = """
Analyze this image and generate question-answer pairs.

{persona_image_role}
Your task is to create exactly **{num_img_questions}** questions for an upcoming {artifact}.
You must not create more or fewer questions than this number.

# Persona method ({persona_label}) - apply this angle within the rules below:
{persona_image_method}
{figure_context}
**MANDATORY RULES - VIOLATION WILL RESULT IN FAILURE:**
1. **EXACT DATA ONLY**: Use ONLY the exact numbers, dates, and text visible in the image. Do NOT interpret, convert, or modify any values.
2. **PRECISE READING**: Read dates, numbers, and labels character-by-character as they appear. For example, if you see "12.3일", it means December 3rd, NOT November 13th.
3. **NO ASSUMPTIONS**: Do not assume relationships, trends, or meanings beyond what is explicitly shown.
4. **VERIFY BEFORE WRITING**: Before writing each answer, mentally point to the exact location in the image where that information appears.
5. **CONSERVATIVE APPROACH**: If you cannot clearly read a specific value or date, do not create a question about it.

**DATA ACCURACY REQUIREMENTS:**
- Charts/Graphs: Only reference data points where both X-axis (date/time) AND Y-axis (value) are clearly visible
- Tables: Only reference cells where both row and column headers are clear
- Text: Only reference text that is completely legible
- Numbers: Copy numbers exactly as shown (including decimal points, units like bp, %, etc.)

**FORBIDDEN ACTIONS:**
- Converting date formats (e.g., 12.3 ≠ 11.13)
- Estimating values between data points
- Creating questions about unclear or partially visible content
- Using information from chart legends if the actual data is unclear

**Question Types to Focus On:**
- Direct reading of clearly visible data points
- Identification of clearly labeled chart/table elements
- Reading of section titles, page numbers, or menu items
- Comparison of clearly visible values (highest, lowest, specific dates)

Write questions and answers in Korean and respond in JSON format.
Do not use arrays/lists in the JSON format.


#Format:
```json
{
    "QUESTION": "CDS 프리미엄 차트에서 12월 17일의 수치는 얼마입니까?",
    "ANSWER": "CDS 프리미엄 차트에서 12월 17일의 수치는 36.3bp입니다."
},
{
    "QUESTION": "목차에서 개선 방안의 첫 번째 항목은 무엇입니까?",
    "ANSWER": "목차에서 개선 방안의 첫 번째 항목은 '건전성 규제 완화'입니다."
},
{
    "QUESTION": "9월 전후 지표 악화 차트에서 외환 차익거래유인 최고점은 언제 기록되었습니까?",
    "ANSWER": "9월 전후 지표 악화 차트에서 외환 차익거래유인 최고점은 10월 2일경에 기록되었습니다."
}
```
"""


def build_text_prompt(
    context: str,
    domain: str,
    num_questions: str,
    persona: Union[str, "Persona", None] = None,
) -> str:
    """Render the text Q&A prompt for the given context, domain and persona."""
    p = get_persona(persona)
    role = p.text_role.replace("{domain}", str(domain))
    method = p.method.replace("{domain}", str(domain))
    return (
        _TEXT_PROMPT.replace("{context}", str(context))
        .replace("{persona_role}", role)
        .replace("{artifact}", p.artifact)
        .replace("{persona_label}", p.label)
        .replace("{persona_method}", method)
        .replace("{num_questions}", str(num_questions))
    )


def build_image_instruction(
    domain: str,
    num_img_questions: str,
    persona: Union[str, "Persona", None] = None,
    context: str = "",
) -> str:
    """Render the image Q&A instruction text (without the image payload).

    ``context`` is the surrounding-document text linked to this figure (section
    heading + the paragraphs/caption immediately around it, assembled by
    :mod:`pdf_qa.layout`). When provided it is injected as a FIGURE CONTEXT block
    so the model can interpret *what the chart shows and why it matters* -- while
    the strict data-accuracy rules still force every number/label to come from
    the image itself. Empty context renders nothing (identical to the old
    behaviour), so charts with no detectable surrounding text still work.
    """
    p = get_persona(persona)
    role = p.image_role.replace("{domain}", str(domain))
    method = p.image_method.replace("{domain}", str(domain))
    context_block = ""
    context = (context or "").strip()
    if context:
        context_block = (
            "\n**FIGURE CONTEXT (surrounding document text near this figure — "
            "use it to understand what the figure shows and why it matters, and "
            "to phrase questions naturally):**\n"
            f"{context}\n\n"
            "Use this context ONLY to frame the meaning/significance of the "
            "figure. Every number, date, and label in your answers must still be "
            "read from the image itself, per the rules below.\n"
        )
    return (
        _IMAGE_INSTRUCTION.replace("{persona_image_role}", role)
        .replace("{artifact}", p.artifact)
        .replace("{persona_label}", p.label)
        .replace("{persona_image_method}", method)
        .replace("{figure_context}", context_block)
        .replace("{num_img_questions}", str(num_img_questions))
    )


def detect_image_format(image_path: str = "") -> str:
    """Map a file extension to a media subtype (png/jpeg/gif/bmp)."""
    image_format = "png"
    if image_path:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            image_format = "jpeg"
        elif ext == ".png":
            image_format = "png"
        elif ext == ".gif":
            image_format = "gif"
        elif ext == ".bmp":
            image_format = "bmp"
    return image_format
