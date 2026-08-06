"""Tests for ``pdf2llm publish-hf`` (workflows.pdf_native_post_training.publish_hf).

Fully offline: model-card generation from the committed ``summary.json`` and the
``--dry-run`` / error paths. No network, no HF token, no upload.
"""
from workflows.pdf_native_post_training import publish_hf as PH


def _summary() -> dict:
    return PH._load_summary(PH._DEFAULT_SUMMARY)


def test_summary_loads_and_has_scores():
    s = _summary()
    assert s.get("base_model") == "Qwen/Qwen3-8B"
    assert PH._metric(s, "sft_bf16_retrieval", "f1") is not None


def test_model_card_is_honest_and_actionable():
    card = PH.build_model_card(_summary(), repo_id="me/pdf2llm-sft-qwen3-8b",
                               base_model="Qwen/Qwen3-8B", arm="sft_bf16_retrieval")
    # front-matter + provenance
    assert card.startswith("---")
    assert "base_model: Qwen/Qwen3-8B" in card
    # the exact command a cloner runs to load the REAL weights
    assert "pdf2llm ask --hf me/pdf2llm-sft-qwen3-8b" in card
    # honest closed-book vs retrieval framing (the benchmark lesson)
    assert "closed-book" in card and "검색" in card
    assert "합성" in card  # synthetic-data caveat


def test_reference_scores_card_relabels_for_small_model():
    # Uploading a non-8B model (e.g. CPU demo) must NOT present 8B scores as its own.
    card = PH.build_model_card(_summary(), repo_id="me/pdf2llm-cook-demo",
                               base_model="Qwen/Qwen2.5-0.5B-Instruct",
                               arm="sft_bf16_retrieval", reference_scores=True)
    assert "base_model: Qwen/Qwen2.5-0.5B-Instruct" in card
    # scores explicitly disowned + labelled as the 8B reference
    assert "이 업로드 모델 자체의 점수가 아닙니다" in card
    assert "이 모델의 점수가 아님" in card
    assert "소형 데모" in card
    # default (8B) card must NOT carry the disclaimer
    d = PH.build_model_card(_summary(), repo_id="me/pdf2llm-sft-qwen3-8b",
                            base_model="Qwen/Qwen3-8B", arm="sft_bf16_retrieval")
    assert "이 모델의 점수가 아님" not in d


def test_reference_scores_auto_enabled_on_base_mismatch(tmp_path, capsys):
    # publish() must auto-detect a non-8B base and relabel the card (dry-run, no token).
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    rc = PH.publish(str(d), "me/cook-demo", base_model="Qwen/Qwen2.5-0.5B-Instruct",
                    arm="sft_bf16_retrieval", summary_path=PH._DEFAULT_SUMMARY, dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "이 모델의 점수가 아님" in out


def test_dry_run_prints_card_without_upload(tmp_path, capsys):
    d = tmp_path / "sft"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    (d / "model.safetensors").write_text("weights", encoding="utf-8")
    rc = PH.publish(str(d), "me/repo", base_model="", arm="sft_bf16_retrieval",
                    summary_path=PH._DEFAULT_SUMMARY, dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in out
    assert "config.json" in out and "model.safetensors" in out
    # dry-run must not write a card into the dir
    assert not (d / "README.md").exists()


def test_missing_model_dir_returns_1(tmp_path, capsys):
    rc = PH.publish(str(tmp_path / "nope"), "me/repo", base_model="Qwen/Qwen3-8B",
                    arm="sft_bf16_retrieval", summary_path=PH._DEFAULT_SUMMARY)
    assert rc == 1


def test_no_token_returns_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    d = tmp_path / "sft"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    rc = PH.publish(str(d), "me/repo", base_model="Qwen/Qwen3-8B", arm="sft_bf16_retrieval",
                    summary_path=PH._DEFAULT_SUMMARY, dry_run=False)
    assert rc == 2  # valid dir, but no credentials → refuse before any network call


def test_non_model_dir_dry_run_warns(tmp_path, capsys):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "notes.txt").write_text("x", encoding="utf-8")
    rc = PH.publish(str(d), "me/repo", base_model="Qwen/Qwen3-8B", arm="sft_bf16_retrieval",
                    summary_path=PH._DEFAULT_SUMMARY, dry_run=True)
    err = capsys.readouterr().err
    assert rc == 0
    assert "config.json" in err  # warns the dir doesn't look like a merged model


def test_launcher_exposes_publish_hf_subcommand():
    from pdf_qa.cli import build_parser

    args = build_parser().parse_args(
        ["publish-hf", "--model-dir", "d", "--repo-id", "me/r", "--dry-run"])
    assert args.command == "publish-hf"
    assert args.model_dir == "d"
    assert args.repo_id == "me/r"
    assert args.dry_run is True
