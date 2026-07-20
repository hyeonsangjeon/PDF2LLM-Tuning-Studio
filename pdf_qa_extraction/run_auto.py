#!/usr/bin/env python3
"""Zero-config batch runner: a folder of PDFs in, fine-tuning JSONL out.

Point it at an input folder and it runs the full chart-aware pipeline over every
``*.pdf``, writing one ``<name>.qa.jsonl`` per document, a combined
``all.qa.jsonl``, and a ``manifest.json`` (per-file counts, persona/provider,
device path, and the chart<->context linkage). Everything is configured from the
environment ledger (see ``settings.yaml`` / ``.env``); GPU/CPU is auto-detected.

    # locally
    INPUT_DIR=./in OUTPUT_DIR=./out LLM_PROVIDER=ollama python run_auto.py

    # from the public image (GPU), mounting host folders
    docker run --rm --gpus all --env-file .env \
        -v "$PWD/in:/data/input" -v "$PWD/out:/data/output" \
        ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest-gpu \
        python run_auto.py

Idempotent: a document whose ``<name>.qa.jsonl`` already exists is skipped
unless ``OVERWRITE=1``.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys

# Load a local .env when present (optional dependency).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Make ``pdf_qa`` importable whether pip-installed (container) or next to this
# script (repo checkout).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    from pdf_qa import extract_qa, get_provider, probe_device
    from pdf_qa.config import QAConfig
    from pdf_qa.manifest import build_manifest

    input_dir = os.environ.get("INPUT_DIR", "/data/input")
    output_dir = os.environ.get("OUTPUT_DIR", "/data/output")
    overwrite = _as_bool(os.environ.get("OVERWRITE"), default=False)

    if not os.path.isdir(input_dir):
        sys.exit(
            f"INPUT_DIR '{input_dir}' does not exist. Set INPUT_DIR (and mount it "
            "in the container) to a folder of PDFs."
        )
    os.makedirs(output_dir, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdfs:
        sys.exit(f"No *.pdf files found in INPUT_DIR '{input_dir}'.")

    # One config + one provider client reused across every document.
    config = QAConfig.from_env()
    provider_name = os.environ.get("LLM_PROVIDER", "azure")
    device = probe_device()
    print(f"[auto] device: {device.summary()}")
    print(
        f"[auto] provider={provider_name} persona={config.persona} "
        f"domain={config.domain} — {len(pdfs)} document(s)"
    )
    try:
        llm = get_provider(provider_name, config=config)
    except Exception as exc:  # pragma: no cover - depends on live creds
        sys.exit(
            f"Failed to initialise provider '{provider_name}': {exc}\n"
            "Check your .env (run `python -m pdf_qa.settings --check "
            f"{provider_name}`)."
        )

    combined_path = os.path.join(output_dir, "all.qa.jsonl")
    manifest = {
        "provider": provider_name,
        "persona": config.persona,
        "domain": config.domain,
        "device": dataclasses.asdict(device),
        "documents": [],
        "totals": {"documents": 0, "pairs": 0, "text": 0, "image": 0, "figures": 0},
    }
    combined: list = []
    processed = 0

    for pdf_path in pdfs:
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(output_dir, f"{name}.qa.jsonl")
        if os.path.exists(out_path) and not overwrite:
            print(f"[auto] skip (exists): {name}.qa.jsonl  (set OVERWRITE=1 to redo)")
            manifest["documents"].append({"document": name, "skipped": True})
            continue

        print(f"[auto] processing: {os.path.basename(pdf_path)}")
        try:
            pairs = extract_qa(pdf_path, out=out_path, provider_obj=llm)
        except Exception as exc:
            print(f"[auto]   ERROR on {name}: {exc}", file=sys.stderr)
            manifest["documents"].append({"document": name, "error": str(exc)})
            continue

        doc_manifest = build_manifest(
            pairs,
            {
                "document": f"{name}.pdf",
                "persona": config.persona,
                "provider": provider_name,
                "domain": config.domain,
            },
        )
        doc_manifest["output"] = os.path.basename(out_path)
        doc_manifest.pop("device", None)
        manifest["documents"].append(doc_manifest)

        counts = doc_manifest["counts"]
        manifest["totals"]["documents"] += 1
        manifest["totals"]["pairs"] += counts["total"]
        manifest["totals"]["text"] += counts["text"]
        manifest["totals"]["image"] += counts["image"]
        manifest["totals"]["figures"] += counts["figures"]
        combined.extend(pairs)
        processed += 1
        print(
            f"[auto]   -> {out_path}  ({counts['total']} Q&A, "
            f"{counts['image']} image, {counts['figures']} figures)"
        )

    if combined:
        with open(combined_path, "w", encoding="utf-8") as handle:
            for item in combined:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[auto] combined -> {combined_path} ({len(combined)} Q&A)")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(
        f"[auto] manifest -> {manifest_path}  "
        f"(processed {processed}/{len(pdfs)} documents)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
