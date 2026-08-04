"""Concrete stages for the pdf_native_post_training workflow.

Order: ingest -> generate -> verify_evidence -> policy_gate -> curate -> export
-> [train_smoke] -> eval -> report. Every stage produces a JSON-able output and a
JSON-able signature (used by the harness for resume-by-hash).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from pdf_qa.evidence import groundable_tokens
from pdf_qa.policy import DocumentPolicy, guard_provider_call, inspect_pdf
from pdf_qa.pii import has_real_pii
from pdf_qa.provenance import Document, parse_pdf
from pdf_qa.run_bundle import sha256_canonical, sha256_file, atomic_write_json, atomic_write_text

from evaluation.evidence_verifier import verify_dataset

from .prompts import build_generation_prompt
from .providers import RecordedReplayProvider, LiveOllamaProvider
from .scoring import score_pairs
from .stages.harness import Stage, StageContext


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _doc_text(documents: List[dict]) -> str:
    parts = []
    for d in documents:
        for el in d.get("elements", []):
            parts.append(el.get("text", ""))
    return " ".join(parts)


def _make_provider(ctx: StageContext):
    mode = ctx.config.get("mode", "recorded_replay")
    if mode == "recorded_replay":
        return RecordedReplayProvider.from_jsonl(ctx.config["_recorded_path"])
    if mode == "live_ollama":
        return LiveOllamaProvider(model=ctx.config.get("model", "qwen2.5:0.5b"))
    raise ValueError(f"unknown mode {mode!r}")


def _policy(ctx: StageContext) -> DocumentPolicy:
    return DocumentPolicy.from_dict(ctx.config.get("_policy", {}))


# --------------------------------------------------------------------------- #
# stages                                                                       #
# --------------------------------------------------------------------------- #
def _ingest_sig(ctx: StageContext):
    docs = [{"path": os.path.basename(p), "sha256": sha256_file(p)} for p in ctx.config["_doc_paths"]]
    return {"stage": "ingest", "docs": docs, "policy": ctx.config.get("_policy", {}), "parser": "provenance/v1"}


def _ingest_run(ctx: StageContext):
    documents = []
    for p in ctx.config["_doc_paths"]:
        inspect_pdf(p)  # PDF threat gate: raises PDFQuarantined on bad input
        doc = parse_pdf(p, version=ctx.config.get("doc_version"))
        documents.append(doc.to_dict())
    return {"documents": documents, "classification": _policy(ctx).classification}


def _generate_sig(ctx: StageContext):
    src = ctx.outputs["ingest"]
    extra = {}
    if ctx.config.get("mode") == "recorded_replay":
        extra["recorded"] = sha256_file(ctx.config["_recorded_path"])
    return {"stage": "generate", "ingest": sha256_canonical(src),
            "provider": ctx.config.get("mode"), "prompts": "prompts/v1", **extra}


def _generate_run(ctx: StageContext):
    documents = ctx.outputs["ingest"]["documents"]
    doctext = _doc_text(documents)
    provider = _make_provider(ctx)

    # egress gate BEFORE any provider use (fail-closed under restricted policy)
    guard_provider_call(_policy(ctx), provider.name)

    candidates: List[dict] = []
    if isinstance(provider, RecordedReplayProvider):
        for rec in provider.recorded_questions():
            g = provider.generate(rec["question"], doctext)  # verifies prompt-hash replay
            if g.answer != rec["answer"]:
                raise RuntimeError(f"replay mismatch for {rec['qa_id']}")
            candidates.append(rec)
    else:  # pragma: no cover - live path needs a daemon
        for rec in _load_jsonl(ctx.config["_gold_path"]):
            g = provider.generate(rec["question"], doctext)
            out = dict(rec)
            out["answer"] = g.answer
            out["generation"] = {"provider": provider.name, "model": getattr(provider, "model", provider.name),
                                 "generation_mode": provider.generation_mode, "prompt_sha256": None}
            out["evidence"] = []
            candidates.append(out)
    return {"candidates": candidates, "provider": provider.name,
            "generation_mode": provider.generation_mode, "n": len(candidates)}


def _verify_sig(ctx: StageContext):
    return {"stage": "verify_evidence", "generate": sha256_canonical(ctx.outputs["generate"]),
            "verifier": "evidence_verifier/v1"}


def _verify_run(ctx: StageContext):
    documents = {d["sha256"]: Document.from_dict(d) for d in ctx.outputs["ingest"]["documents"]}
    candidates = ctx.outputs["generate"]["candidates"]
    report = verify_dataset(candidates, documents)
    failed = {f["qa_id"] for f in report["failures"]}
    verified = [c for c in candidates if c["qa_id"] not in failed]
    return {"report": report, "verified": verified,
            "evidence_address_integrity": report["evidence_address_integrity"]}


def _policy_sig(ctx: StageContext):
    return {"stage": "policy_gate", "verify": sha256_canonical(ctx.outputs["verify_evidence"]["verified"]),
            "policy": ctx.config.get("_policy", {}), "pii": "pii/v1"}


def _policy_run(ctx: StageContext):
    policy = _policy(ctx)
    passed, quarantined = [], []
    for rec in ctx.outputs["verify_evidence"]["verified"]:
        blob = rec.get("answer", "") + " " + " ".join(e.get("quote", "") for e in rec.get("evidence", []))
        if has_real_pii(blob):
            quarantined.append({"qa_id": rec["qa_id"], "reason": "real_pii_detected"})
        else:
            passed.append(rec)
    provider = ctx.outputs["generate"]["provider"]
    return {"passed": passed, "quarantined": quarantined,
            "egress": {"provider": provider, "classification": policy.classification,
                       "raw_content_egress": policy.raw_content_egress},
            "n_passed": len(passed), "n_quarantined": len(quarantined)}


def _curate_sig(ctx: StageContext):
    return {"stage": "curate", "policy": sha256_canonical(ctx.outputs["policy_gate"]["passed"])}


def _curate_run(ctx: StageContext):
    approved = []
    for rec in ctx.outputs["policy_gate"]["passed"]:
        r = dict(rec)
        r["review_status"] = "approved"  # demo auto-approves verified+gated records
        approved.append(r)
    return {"approved": approved, "n": len(approved),
            "note": "demo auto-approves; production requires human review"}


def _export_sig(ctx: StageContext):
    return {"stage": "export", "curate": sha256_canonical(ctx.outputs["curate"]["approved"]),
            "format": "sft_chat/v1"}


def _export_run(ctx: StageContext):
    from .prompts import SYSTEM

    rows = []
    for rec in ctx.outputs["curate"]["approved"]:
        if not rec.get("answerable", True):
            user = rec["question"]
        else:
            ctx_txt = " ".join(e.get("quote", "") for e in rec.get("evidence", []))
            user = f"{rec['question']}\n\n[근거]\n{ctx_txt}"
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": rec["answer"]},
            ],
            "qa_id": rec["qa_id"],
            "category": rec.get("category"),
        })
    art_dir = os.path.join(ctx.run_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    path = os.path.join(art_dir, "train_sft.jsonl")
    atomic_write_text(path, "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return {"path": os.path.relpath(path, ctx.run_dir), "n_rows": len(rows),
            "sha256": sha256_file(path)}


def _eval_sig(ctx: StageContext):
    return {"stage": "eval", "generate": sha256_canonical(ctx.outputs["generate"]),
            "gold": sha256_file(ctx.config["_gold_path"]), "scorer": "korquad_f1/v1"}


def _eval_run(ctx: StageContext):
    gold = {r["qa_id"]: r for r in _load_jsonl(ctx.config["_gold_path"])}
    cands = {c["qa_id"]: c for c in ctx.outputs["generate"]["candidates"]}
    pairs, by_cat = [], {}
    for qid, g in gold.items():
        c = cands.get(qid)
        if c is None:
            continue
        pair = {"pred": c.get("answer", ""), "gold": g.get("answer", "")}
        pairs.append(pair)
        by_cat.setdefault(g.get("category", "?"), []).append(pair)
    overall = score_pairs(pairs)
    per_category = {k: score_pairs(v) for k, v in sorted(by_cat.items())}
    return {"overall": overall, "per_category": per_category,
            "evidence_address_integrity": ctx.outputs["verify_evidence"]["evidence_address_integrity"]}


def _report_sig(ctx: StageContext):
    keys = ["ingest", "generate", "verify_evidence", "policy_gate", "curate", "export", "eval"]
    return {"stage": "report", "inputs": {k: sha256_canonical(ctx.outputs.get(k)) for k in keys if k in ctx.outputs}}


def _report_run(ctx: StageContext):
    ev = ctx.outputs["eval"]
    vg = ctx.outputs["verify_evidence"]["report"]
    pg = ctx.outputs["policy_gate"]
    ex = ctx.outputs["export"]
    ing = ctx.outputs["ingest"]
    summary = {
        "run_id": ctx.bundle.run_id,
        "reproducibility_fingerprint": ctx.bundle.reproducibility_fingerprint(),
        "mode": ctx.config.get("mode"),
        "documents": len(ing["documents"]),
        "classification": ing["classification"],
        "candidates": ctx.outputs["generate"]["n"],
        "evidence_address_integrity": vg["evidence_address_integrity"],
        "evidence_passed": vg["passed"],
        "evidence_failed": vg["failed"],
        "policy_passed": pg["n_passed"],
        "policy_quarantined": pg["n_quarantined"],
        "train_rows_exported": ex["n_rows"],
        "eval": ev,
    }
    md = _render_md(summary)
    atomic_write_json(os.path.join(ctx.run_dir, "report.json"), summary)
    atomic_write_text(os.path.join(ctx.run_dir, "report.md"), md)
    return summary


def _render_md(s: Dict[str, Any]) -> str:
    ev = s["eval"]
    lines = [
        f"# PDF-native post-training run — {s['run_id']}",
        "",
        f"- mode: **{s['mode']}**  ·  classification: **{s['classification']}**",
        f"- reproducibility_fingerprint: `{s['reproducibility_fingerprint'][:16]}…`",
        f"- documents ingested: {s['documents']}  ·  candidates: {s['candidates']}",
        "",
        "## Evidence-address integrity (mechanical)",
        f"- passed **{s['evidence_passed']}/{s['evidence_passed'] + s['evidence_failed']}**"
        f"  (integrity = {s['evidence_address_integrity']:.3f})",
        "",
        "## Policy gate",
        f"- passed: {s['policy_passed']}  ·  quarantined (real PII): {s['policy_quarantined']}",
        "",
        "## Export",
        f"- SFT training rows: {s['train_rows_exported']}",
        "",
        "## Answer quality vs gold (KorQuAD-style)",
        f"- overall: EM **{ev['overall']['em']}** · F1 **{ev['overall']['f1']}** (n={ev['overall']['n']})",
        "",
        "| category | EM | F1 | n |",
        "|---|---|---|---|",
    ]
    for cat, m in ev["per_category"].items():
        lines.append(f"| {cat} | {m['em']} | {m['f1']} | {m['n']} |")
    lines += ["", "_Evidence-address integrity is a mechanical address/quote/hash check, "
              "not a claim of semantic correctness._", ""]
    return "\n".join(lines)


def _load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# pipeline builder                                                             #
# --------------------------------------------------------------------------- #
def build_pipeline(config: Dict[str, Any]) -> List[Stage]:
    stages = [
        Stage("ingest", _ingest_sig, _ingest_run),
        Stage("generate", _generate_sig, _generate_run),
        Stage("verify_evidence", _verify_sig, _verify_run),
        Stage("policy_gate", _policy_sig, _policy_run),
        Stage("curate", _curate_sig, _curate_run),
        Stage("export", _export_sig, _export_run),
    ]
    if config.get("train", {}).get("enabled"):
        from .train_stage import make_train_stage
        stages.append(make_train_stage())
    stages += [
        Stage("eval", _eval_sig, _eval_run),
        Stage("report", _report_sig, _report_run),
    ]
    return stages
