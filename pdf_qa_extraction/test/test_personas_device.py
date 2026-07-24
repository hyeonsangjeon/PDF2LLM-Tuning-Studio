"""Unit tests for personas and the GPU/CPU device-aware extraction plan.

These are deliberately dependency-light (no torch / onnxruntime / unstructured),
so they run in CI on the CPU image and in a bare venv.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from pdf_qa.device import DeviceReport
from pdf_qa.extract import resolve_extraction_plan
from pdf_qa.prompts import (
    DEFAULT_PERSONA,
    PERSONAS,
    build_image_instruction,
    build_text_prompt,
    get_persona,
    list_personas,
    load_personas,
    reload_personas,
)

_TEXT_PLACEHOLDERS = (
    "{persona_role}",
    "{artifact}",
    "{persona_label}",
    "{persona_method}",
    "{context}",
    "{num_questions}",
)
_IMAGE_PLACEHOLDERS = (
    "{persona_image_role}",
    "{artifact}",
    "{persona_label}",
    "{persona_image_method}",
    "{num_img_questions}",
)


# --------------------------- personas (YAML ledger) ---------------------------
def test_persona_registry_has_expected_keys():
    # feynman is appended by the ledger refactor; professor must stay first so
    # the container smoke test (list_personas()[0] == "professor") holds.
    assert list_personas() == [
        "professor",
        "socratic",
        "consultant",
        "interviewer",
        "analyst",
        "feynman",
        "memoirist",
    ]
    assert DEFAULT_PERSONA == "professor"
    assert list_personas()[0] == "professor"


def test_default_persona_reproduces_professor_wording():
    prompt = build_text_prompt("CTX", "International Finance", "5")
    assert "You are a Teacher/Professor in International Finance." in prompt
    assert "for an upcoming quiz/examination." in prompt
    # persona method block is rendered with its Korean label header
    assert "# Persona method (교수/출제자) - follow this approach:" in prompt
    assert "Method — Examination setting:" in prompt
    # every placeholder is filled
    for token in _TEXT_PLACEHOLDERS:
        assert token not in prompt
    assert "CTX" in prompt


@pytest.mark.parametrize(
    "persona,role_needle,artifact,method_needle",
    [
        (
            "socratic",
            "You are a Socratic tutor guiding a learner through 금융.",
            "guided study dialogue",
            "Method — Socratic questioning:",
        ),
        (
            "consultant",
            "You are a senior practitioner and consultant in 금융.",
            "advisory session",
            "Method — Advisory framing:",
        ),
        (
            "interviewer",
            "You are a technical interviewer assessing a candidate in 금융.",
            "job interview",
            "Method — Technical interview:",
        ),
        (
            "analyst",
            "You are a research analyst in 금융.",
            "analytical review",
            "Method — Analytical synthesis:",
        ),
        (
            "feynman",
            "You are Richard Feynman explaining 금융 to a curious beginner.",
            "plain-language explainer",
            "Method — Feynman technique:",
        ),
        (
            "memoirist",
            "You are the narrator of this autobiography, retelling your own life (금융) in the first person and in your own original voice and register.",
            "life-story memoir",
            "Method — First-person life recollection with VOICE PRESERVATION (v4):",
        ),
    ],
)
def test_text_persona_swaps_role_artifact_and_method(persona, role_needle, artifact, method_needle):
    prompt = build_text_prompt("CTX", "금융", "3", persona)
    assert role_needle in prompt
    assert f"for an upcoming {artifact}." in prompt
    assert method_needle in prompt


def test_feynman_uses_analogy_and_first_principles():
    prompt = build_text_prompt("CTX", "금융", "2", "feynman")
    assert "everyday analogy" in prompt
    assert "first-principles" in prompt


def test_memoirist_is_first_person_and_faithful():
    prompt = build_text_prompt("CTX", "아버지의 생애", "3", "memoirist")
    # role weaves the domain in and asks for a first-person voice
    assert "the narrator of this autobiography" in prompt
    assert "first person" in prompt
    assert '"나는 …"' in prompt
    # the memory-preservation guardrail must be present
    assert "NEVER invent" in prompt
    # voice-preservation doctrine v4: register anchor restored + 존댓말 explicitly forbidden
    assert "VOICE PRESERVATION (v4)" in prompt
    assert "VOICE / REGISTER — preserve exactly" in prompt
    assert "Keep the PLAIN literary register" in prompt
    assert "convert to a polite/존댓말 register" in prompt
    assert "ORTHOGRAPHY — spelling ONLY" in prompt
    assert "Connective / final / negation endings are VOICE" in prompt
    # questions modern + grounded/non-leading; faithfulness outranks question count
    assert "Write QUESTIONS in natural present-day Korean" in prompt
    assert "FORBID sweeping/leading questions" in prompt
    assert "Prefer FEWER faithful pairs" in prompt
    assert "Build each answer PRIMARILY from the narrator's actual words" in prompt
    img = build_image_instruction("아버지의 생애", "1", "memoirist")
    assert '"나는 …"' in img
    assert "never" in img.lower()
    assert "preserved PLAIN literary voice" in img
    assert "평서형 문어체" in img
    assert "only spelling" in img


def test_image_instruction_uses_persona_and_domain():
    img = build_image_instruction("경제", "2", "consultant")
    assert "You are a senior 경제 consultant reviewing this figure for a client." in img
    assert "for an upcoming advisory session." in img
    assert "# Persona method (실무 컨설턴트) - apply this angle within the rules below:" in img
    for token in _IMAGE_PLACEHOLDERS:
        assert token not in img


def test_unknown_persona_raises():
    with pytest.raises(ValueError, match="Unknown persona"):
        build_text_prompt("c", "d", "1", "does-not-exist")


def test_get_persona_none_is_default():
    assert get_persona(None).key == "professor"
    assert get_persona("ANALYST").key == "analyst"  # case-insensitive


def test_all_personas_render_without_leftover_placeholders():
    for key in PERSONAS:
        t = build_text_prompt("ctx", "dom", "1", key)
        i = build_image_instruction("dom", "1", key)
        for token in _TEXT_PLACEHOLDERS:
            assert token not in t
        for token in _IMAGE_PLACEHOLDERS:
            assert token not in i


def test_every_persona_has_a_distinct_method():
    # The whole point of the refactor: each persona is a genuinely different
    # 방식, so both the text and image methods must be pairwise-unique.
    text_methods = [p.method for p in PERSONAS.values()]
    image_methods = [p.image_method for p in PERSONAS.values()]
    assert len(set(text_methods)) == len(text_methods)
    assert len(set(image_methods)) == len(image_methods)
    # ...and none of them is empty.
    assert all(m.strip() for m in text_methods + image_methods)


def test_ledger_loads_all_personas_from_yaml():
    personas, default = load_personas()
    assert default == "professor"
    assert set(personas) == {
        "professor",
        "socratic",
        "consultant",
        "interviewer",
        "analyst",
        "feynman",
        "memoirist",
    }


def test_reload_personas_returns_registry():
    import pdf_qa.prompts as prompts_mod

    reloaded = reload_personas()
    # reload rebinds the module-level registry; the returned object is that
    # same current registry and still contains the Feynman persona.
    assert reloaded is prompts_mod.PERSONAS
    assert "feynman" in reloaded


def test_malformed_ledger_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\ndefault: x\npersonas: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_personas(str(bad))


# --------------------------- device / GPU plan ---------------------------
def _gpu_report():
    return DeviceReport(
        torch_installed=True,
        torch_version="2.x",
        torch_cuda_available=True,
        cuda_device_name="RTX 3080",
        cuda_device_count=1,
        onnxruntime_installed=True,
        onnxruntime_gpu_package=True,
        onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        onnxruntime_cuda_provider=True,
    )


def _cpu_report():
    return DeviceReport()  # all defaults -> CPU


def test_gpu_ready_gated_on_torch_cuda():
    assert _gpu_report().gpu_ready is True
    assert _cpu_report().gpu_ready is False
    # onnxruntime-gpu present but no driver -> NOT gpu_ready (CPU image w/o --gpus)
    only_ort = DeviceReport(onnxruntime_installed=True, onnxruntime_gpu_package=True,
                            onnxruntime_cuda_provider=True)
    assert only_ort.gpu_ready is False


def test_cpu_plan_stays_light():
    plan = resolve_extraction_plan("auto", None, gpu_boost=True, device=_cpu_report())
    assert plan == {
        "strategy": "auto",
        "infer_table_structure": False,
        "hi_res_model_name": None,
        "gpu_accelerated": False,
    }


def test_gpu_plan_escalates_to_hires_and_tables():
    plan = resolve_extraction_plan("auto", None, gpu_boost=True, device=_gpu_report())
    assert plan["strategy"] == "hi_res"
    assert plan["infer_table_structure"] is True
    assert plan["gpu_accelerated"] is True


def test_gpu_boost_off_keeps_light_path():
    plan = resolve_extraction_plan("auto", None, gpu_boost=False, device=_gpu_report())
    assert plan["strategy"] == "auto"
    assert plan["infer_table_structure"] is False
    assert plan["gpu_accelerated"] is False


def test_explicit_strategy_is_respected_on_gpu():
    plan = resolve_extraction_plan("fast", None, gpu_boost=True, device=_gpu_report())
    assert plan["strategy"] == "fast"  # not overridden to hi_res
    assert plan["infer_table_structure"] is True  # tables still enabled by boost


def test_explicit_layout_model_selected_and_forces_hires():
    # An explicit layout model must actually reach unstructured: it is surfaced
    # as hi_res_model_name AND escalates auto -> hi_res so it is not ignored.
    plan = resolve_extraction_plan("auto", "detectron2_onnx", gpu_boost=False,
                                   device=_cpu_report())
    assert plan["hi_res_model_name"] == "detectron2_onnx"
    assert plan["strategy"] == "hi_res"
    assert plan["infer_table_structure"] is True


def test_empty_layout_model_behaves_like_none():
    # An empty string must be treated exactly like ``None``: no escalation, no
    # table inference -- never an inconsistent (auto + infer_tables) plan.
    plan = resolve_extraction_plan("auto", "", gpu_boost=False, device=_cpu_report())
    assert plan == {
        "strategy": "auto",
        "infer_table_structure": False,
        "hi_res_model_name": "",
        "gpu_accelerated": False,
    }
