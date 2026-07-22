"""Tests for the environment-variable ledger (``settings.yaml`` / settings.py).

``validate_env`` / ``provider_configured`` take an explicit ``env`` dict, so
these are hermetic (no os.environ mutation) apart from the file-override test.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pdf_qa.settings import (
    grouped_settings,
    load_settings,
    provider_configured,
    render_dotenv_example,
    validate_env,
)


def test_ledger_declares_core_and_provider_vars():
    names = {s.name for s in load_settings()}
    for expected in (
        "LLM_PROVIDER",
        "PERSONA",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_KEY",
        "OLLAMA_MODEL",
        "FIGURES_DIR",
    ):
        assert expected in names


def test_setting_names_are_unique():
    names = [s.name for s in load_settings()]
    assert len(names) == len(set(names))


def test_grouped_has_all_provider_groups():
    groups = grouped_settings()
    for g in (
        "core",
        "provider.azure",
        "provider.openai",
        "provider.bedrock",
        "provider.ollama",
        "quality",
        "webapp",
    ):
        assert g in groups


def test_ledger_declares_new_language_vision_and_quality_vars():
    by_name = {s.name: s for s in load_settings()}
    # Output-language lock, with its aliases.
    assert "OUTPUT_LANGUAGE" in by_name
    assert set(by_name["OUTPUT_LANGUAGE"].aliases) >= {"QA_LANGUAGE", "LANGUAGE"}
    # Separate Ollama vision model.
    assert "OLLAMA_VISION_MODEL" in by_name
    assert by_name["OLLAMA_VISION_MODEL"].group == "provider.ollama"
    # Quality-control knobs.
    for name in ("VALIDATE_QA", "DEDUP_QA", "MIN_QUESTION_CHARS", "DEDUP_SIMILARITY"):
        assert name in by_name and by_name[name].group == "quality"


def test_validate_env_azure_requires_endpoint():
    assert validate_env("azure", env={}) == ["AZURE_OPENAI_ENDPOINT"]
    assert validate_env("azure", env={"AZURE_OPENAI_ENDPOINT": "https://x"}) == []


def test_validate_env_openai_requires_key():
    assert validate_env("openai", env={}) == ["OPENAI_API_KEY"]
    assert provider_configured("openai", env={"OPENAI_API_KEY": "sk-x"}) is True


def test_ollama_needs_no_credentials():
    assert validate_env("ollama", env={}) == []
    assert provider_configured("ollama", env={}) is True


def test_is_set_honours_aliases():
    by_name = {s.name: s for s in load_settings()}
    version = by_name["AZURE_OPENAI_API_VERSION"]
    assert "OPENAI_API_VERSION" in version.aliases
    assert version.is_set(env={}) is False
    # Setting only the alias still counts as set.
    assert version.is_set(env={"OPENAI_API_VERSION": "2024-10-21"}) is True


def test_render_dotenv_example_grouped_and_secret_safe():
    text = render_dotenv_example()
    assert "LLM_PROVIDER=" in text
    assert "Core" in text  # group header rendered
    # Secrets appear only as commented placeholders, never with a value.
    assert "# AZURE_OPENAI_API_KEY=" in text
    assert "# OPENAI_API_KEY=" in text
    for line in text.splitlines():
        stripped = line.strip()
        for secret in ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY"):
            if stripped.startswith(secret + "="):  # uncommented assignment
                assert stripped == secret + "=", f"secret leaked a value: {line!r}"


def test_settings_file_override(tmp_path, monkeypatch):
    ledger = tmp_path / "custom.yaml"
    ledger.write_text(
        "settings:\n"
        "  - name: MY_ONLY_VAR\n"
        "    group: core\n"
        "    default: hello\n"
        "    description: a custom ledger entry\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PDF2LLM_SETTINGS_FILE", str(ledger))
    assert {s.name for s in load_settings()} == {"MY_ONLY_VAR"}
