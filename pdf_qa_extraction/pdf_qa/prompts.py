"""Shared prompt templates and personas for text/image Q&A generation.

The *content* of the prompt is identical across providers; only the transport
envelope (how an image is attached to a chat message) differs, and that lives
in each provider module. Placeholders are filled with ``str.replace`` so the
literal JSON braces in the few-shot examples need no escaping.

A **persona** swaps the role the model plays and the style of the generated
questions/answers, so one PDF can seed several *different* fine-tuning datasets
(exam questions, Socratic dialogue, consulting advice, interview drills,
analyst reviews) without changing the output JSON schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Union

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Persona:
    """A role/style preset applied to the Q&A prompts.

    Attributes:
        key: Stable identifier used by config/env/CLI (e.g. ``"professor"``).
        label: Short human-readable name (used in logs, Korean-friendly).
        text_role: The ``You are ...`` sentence for the text prompt. May contain
            ``{domain}``.
        artifact: The kind of deliverable the questions belong to (fills
            ``for an upcoming {artifact}``), e.g. ``"quiz/examination"``.
        goal: One sentence describing what the questions should accomplish and
            in what style (fills the "purpose of the questions" line).
        image_role: The ``You are ...`` sentence for the image prompt. May
            contain ``{domain}``.
    """

    key: str
    label: str
    text_role: str
    artifact: str
    goal: str
    image_role: str


# The default persona reproduces the original professor/exam behaviour exactly.
PERSONAS: Dict[str, Persona] = {
    "professor": Persona(
        key="professor",
        label="교수/출제자",
        text_role="You are a Teacher/Professor in {domain}.",
        artifact="quiz/examination",
        goal=(
            "The purpose of the question(s) is to test the students' "
            "understanding of the context information provided."
        ),
        image_role="You are a professor/teacher in the {domain} field.",
    ),
    "socratic": Persona(
        key="socratic",
        label="소크라테스식 튜터",
        text_role="You are a Socratic tutor guiding a learner through {domain}.",
        artifact="guided study dialogue",
        goal=(
            "The purpose of the question(s) is to provoke deeper reasoning: ask "
            "'why' and 'how' style questions, and make each answer explain the "
            "underlying cause or step-by-step process, using only the context."
        ),
        image_role="You are a Socratic tutor discussing this figure with a learner in {domain}.",
    ),
    "consultant": Persona(
        key="consultant",
        label="실무 컨설턴트",
        text_role="You are a senior practitioner and consultant in {domain}.",
        artifact="advisory Q&A session",
        goal=(
            "The purpose of the question(s) is to surface practical, "
            "decision-oriented insights a professional would ask; each answer "
            "should give actionable guidance grounded only in the context."
        ),
        image_role="You are a senior {domain} consultant reviewing this figure for a client.",
    ),
    "interviewer": Persona(
        key="interviewer",
        label="기술 면접관",
        text_role="You are a technical interviewer assessing a candidate in {domain}.",
        artifact="job interview",
        goal=(
            "The purpose of the question(s) is to evaluate the candidate's "
            "grasp of the material; write interview-style questions with clear, "
            "model answers based only on the context."
        ),
        image_role="You are a technical interviewer using this figure to probe a candidate in {domain}.",
    ),
    "analyst": Persona(
        key="analyst",
        label="리서치 분석가",
        text_role="You are a research analyst in {domain}.",
        artifact="analytical review",
        goal=(
            "The purpose of the question(s) is to require synthesis and "
            "comparison across the material; answers should connect ideas and "
            "summarise implications using only the context."
        ),
        image_role="You are a research analyst interpreting this figure in {domain}.",
    ),
}

#: The persona used when none is specified (original behaviour).
DEFAULT_PERSONA = "professor"


def list_personas() -> List[str]:
    """Return the available persona keys."""
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
The question(s) should be diverse in nature across the document.
{persona_goal}
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
    goal = p.goal.replace("{domain}", str(domain))
    return (
        _TEXT_PROMPT.replace("{context}", str(context))
        .replace("{persona_role}", role)
        .replace("{artifact}", p.artifact)
        .replace("{persona_goal}", goal)
        .replace("{num_questions}", str(num_questions))
    )


def build_image_instruction(
    domain: str,
    num_img_questions: str,
    persona: Union[str, "Persona", None] = None,
) -> str:
    """Render the image Q&A instruction text (without the image payload)."""
    p = get_persona(persona)
    role = p.image_role.replace("{domain}", str(domain))
    return (
        _IMAGE_INSTRUCTION.replace("{persona_image_role}", role)
        .replace("{artifact}", p.artifact)
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
