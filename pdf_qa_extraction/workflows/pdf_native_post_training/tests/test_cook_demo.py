"""Tests for ``pdf2llm cook-demo`` (workflows.pdf_native_post_training.cook_demo).

Fully offline & torch-free: exercises the pure ``train.jsonl`` → chat-``messages``
conversion, the ``--dry-run`` path, and the launcher subcommand wiring. The actual
CPU fine-tune (which downloads a base model) is intentionally NOT run here.
"""
import json

from workflows.pdf_native_post_training import cook_demo as CD


def _train_file(tmp_path):
    rows = [
        {"question": "매출은?", "context": "2024년 매출 100억.", "answer": "100억입니다.",
         "answerable": True},
        {"question": "없는 값?", "context": "무관한 문맥.", "answer": "문서에서 확인할 수 없습니다.",
         "answerable": False},
        {"question": "빈답?", "context": "문맥", "answer": "", "answerable": True},  # dropped
    ]
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(p)


def test_build_messages_rows_shape_and_framing(tmp_path):
    rows = CD.build_messages_rows(_train_file(tmp_path))
    # only the answerable, fully-populated row survives (answerable_only default)
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == CD.EVAL_SYSTEM
    assert "[문맥]" in msgs[1]["content"] and "[질문]" in msgs[1]["content"]
    assert msgs[2]["content"] == "100억입니다."


def test_include_unanswerable_and_limit(tmp_path):
    rows = CD.build_messages_rows(_train_file(tmp_path), answerable_only=False, limit=1)
    assert len(rows) == 1  # limit respected even though 2 rows qualify


def test_write_messages_roundtrip(tmp_path):
    rows = CD.build_messages_rows(_train_file(tmp_path))
    out = CD.write_messages(rows, str(tmp_path / "m.jsonl"))
    back = [json.loads(line) for line in open(out, encoding="utf-8") if line.strip()]
    assert back == rows


def test_dry_run_main_reports_rows_without_torch(tmp_path, capsys):
    rc = CD.main(["--train", _train_file(tmp_path), "--dry-run", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "messages 행 1개" in out
    # dry-run must not create the output model dir
    assert not (tmp_path / "o").exists()


def test_launcher_exposes_cook_demo_subcommand():
    from pdf_qa.cli import build_parser

    args = build_parser().parse_args(["cook-demo", "--out", "runs/x", "--dry-run"])
    assert args.command == "cook-demo"
    assert args.out == "runs/x"
    assert args.dry_run is True
