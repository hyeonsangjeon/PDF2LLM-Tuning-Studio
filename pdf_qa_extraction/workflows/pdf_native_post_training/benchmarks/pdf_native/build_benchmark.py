"""P1-5: build the PDF-native **public frozen regression** benchmark from the
committed synthetic corpus.

Projects ``public_finance_demo/gold_qa.jsonl`` + ``versioned_facts.jsonl`` into the
benchmark schema, assigns a ``document_family_id`` per context group, partitions by
family (so a family — incl. its v1/v2 versions — never crosses a split), runs the
leakage audit, and writes ``public_regression.jsonl`` + ``splits/`` +
``final_manifest.json`` + ``checksums.sha256``.

This is a **public frozen regression** set (input + label + evidence all committed),
so it is deliberately NOT called ``sealed`` or ``unseen``. A protected current-final
store is **not** operated here; the manifest records that as ``planned`` per the spec.
Regenerate deterministically:

    python -m workflows.pdf_native_post_training.benchmarks.pdf_native.build_benchmark
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

from evaluation.pdf_native import assert_no_split_leakage

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORPUS = os.path.normpath(os.path.join(_HERE, "..", "..", "public_finance_demo"))
BENCHMARK_SET_ID = "pdf_native_public_regression/v1"

# family per source file — a context group; v1/v2 of a family stay together.
_SOURCES = [
    {"file": "gold_qa.jsonl", "family": "finance_report", "split": "dev"},
    {"file": "versioned_facts.jsonl", "family": "finance_facts", "split": "regression"},
]
_BENCH_FIELDS = ("qa_id", "question", "answer", "answerable", "category",
                 "fact_volatility", "document_version", "source_status",
                 "effective_from", "effective_until", "supersedes", "evidence",
                 "review_status", "document_family_id", "split", "benchmark_set_id")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for src in _SOURCES:
        for row in _load_jsonl(os.path.join(_CORPUS, src["file"])):
            rec = {k: row.get(k) for k in _BENCH_FIELDS}
            rec["document_family_id"] = src["family"]
            rec["split"] = src["split"]
            rec["benchmark_set_id"] = BENCHMARK_SET_ID
            rec["review_status"] = row.get("review_status", "owner_review_pending")
            records.append(rec)
    records.sort(key=lambda r: r["qa_id"])
    return records


def _category_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    return dict(sorted(counts.items()))


def build_manifest(records: List[Dict[str, Any]], audit: Dict[str, Any]) -> Dict[str, Any]:
    pdfs = sorted(f for f in os.listdir(os.path.join(_CORPUS, "docs")) if f.endswith(".pdf"))
    input_hashes = {f: _sha256_file(os.path.join(_CORPUS, "docs", f)) for f in pdfs}
    n_reviewed = sum(1 for r in records if r["review_status"] == "human_reviewed")
    return {
        "benchmark_set_id": BENCHMARK_SET_ID,
        "role": "public_frozen_regression",
        "status": "owner_review_pending" if n_reviewed < len(records) else "human_reviewed",
        "note": ("Public frozen regression: input+label+evidence are all committed, so this "
                 "set is NOT 'sealed' or 'unseen'. A protected current-final store is not "
                 "operated here."),
        "license": "CC-BY-4.0",
        "generation_policy": "fully synthetic (public_finance_demo/build_fixture.py); no private data or credentials",
        "schema_fields": list(_BENCH_FIELDS),
        "n_examples": len(records),
        "n_human_reviewed": n_reviewed,
        "category_counts": _category_counts(records),
        "source_corpus": {
            "path": "workflows/pdf_native_post_training/public_finance_demo",
            "input_pdf_sha256": input_hashes,
            "gold_qa_sha256": _sha256_file(os.path.join(_CORPUS, "gold_qa.jsonl")),
            "versioned_facts_sha256": _sha256_file(os.path.join(_CORPUS, "versioned_facts.jsonl")),
        },
        "splits": audit["splits"],
        "families": audit["families"],
        "leakage_audit": {"intersection_size": audit["intersection_size"],
                          "disjoint": audit["disjoint"], "n_families": audit["n_families"]},
        "protected_final": {
            "status": "planned",
            "note": ("Not operated. Per spec, without a real protected input/label store the "
                     "'sealed final'/'unseen final' framing is removed and this stays a public "
                     "frozen regression set. A future protected v2 would record only set ID, "
                     "schema, category counts, policy/license, input hash and protected asset "
                     "hash — never labels — in the repo."),
        },
        "experiment_arms_status": "planned (Base/SFT/PTQ/QAT ± retrieval require the GPU workflow)",
    }


def write_all(out_dir: str = _HERE) -> Dict[str, Any]:
    records = build_records()
    splits = {name: [r for r in records if r["split"] == name]
              for name in sorted({r["split"] for r in records})}
    audit = assert_no_split_leakage(splits)

    # public_regression.jsonl
    reg_path = os.path.join(out_dir, "public_regression.jsonl")
    with open(reg_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    # splits/<name>.json
    splits_dir = os.path.join(out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    for name, rows in splits.items():
        with open(os.path.join(splits_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump({"split": name,
                       "document_family_ids": sorted({r["document_family_id"] for r in rows}),
                       "qa_ids": [r["qa_id"] for r in rows]},
                      fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")

    # final_manifest.json
    manifest = build_manifest(records, audit)
    with open(os.path.join(out_dir, "final_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    # checksums.sha256 of generated files
    gen = ["public_regression.jsonl", "final_manifest.json"] + \
          [f"splits/{n}.json" for n in splits]
    lines = [f"{_sha256_file(os.path.join(out_dir, g))}  {g}" for g in sorted(gen)]
    with open(os.path.join(out_dir, "checksums.sha256"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return {"records": len(records), "splits": audit["splits"], "manifest": manifest}


if __name__ == "__main__":  # pragma: no cover
    info = write_all()
    print(f"[build-benchmark] {info['records']} records; splits={info['splits']}; "
          f"disjoint={info['manifest']['leakage_audit']['disjoint']}")
