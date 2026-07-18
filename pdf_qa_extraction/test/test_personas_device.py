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
)


# --------------------------- personas ---------------------------
def test_persona_registry_has_expected_keys():
    assert list_personas() == [
        "professor",
        "socratic",
        "consultant",
        "interviewer",
        "analyst",
    ]
    assert DEFAULT_PERSONA == "professor"


def test_default_persona_reproduces_professor_wording():
    prompt = build_text_prompt("CTX", "International Finance", "5")
    assert "You are a Teacher/Professor in International Finance." in prompt
    assert "for an upcoming quiz/examination." in prompt
    assert "test the students' understanding" in prompt
    # every placeholder is filled
    for token in ("{persona_role}", "{artifact}", "{persona_goal}", "{context}", "{num_questions}"):
        assert token not in prompt
    assert "CTX" in prompt


@pytest.mark.parametrize(
    "persona,needle,artifact",
    [
        ("socratic", "Socratic tutor guiding a learner through", "guided study dialogue"),
        ("consultant", "senior practitioner and consultant in", "advisory Q&A session"),
        ("interviewer", "technical interviewer assessing a candidate in", "job interview"),
        ("analyst", "research analyst in", "analytical review"),
    ],
)
def test_text_persona_swaps_role_and_artifact(persona, needle, artifact):
    prompt = build_text_prompt("CTX", "금융", "3", persona)
    assert needle in prompt
    assert f"for an upcoming {artifact}." in prompt


def test_image_instruction_uses_persona_and_domain():
    img = build_image_instruction("경제", "2", "consultant")
    assert "senior 경제 consultant" in img
    assert "for an upcoming advisory Q&A session." in img
    assert "{persona_image_role}" not in img and "{artifact}" not in img


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
        for token in ("{persona_role}", "{persona_goal}", "{persona_image_role}", "{artifact}"):
            assert token not in t and token not in i


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
        "table_model": None,
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


def test_explicit_table_model_passed_through():
    plan = resolve_extraction_plan("hi_res", "yolox", gpu_boost=True, device=_cpu_report())
    assert plan["table_model"] == "yolox"
    assert plan["infer_table_structure"] is True
