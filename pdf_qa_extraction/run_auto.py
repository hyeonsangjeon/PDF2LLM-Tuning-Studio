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

Incremental & correct by construction:

* A document is **skipped only when its execution contract is unchanged** — the
  PDF's content hash, the resolved :class:`~pdf_qa.config.QAConfig`, the provider
  and the persona/prompt ledger. Renaming, editing the PDF, or changing persona /
  model / prompt / config all invalidate the cache. Corrupt, truncated or
  schema-mismatched output is an invalidation reason, never a cache hit.
* The combined ``all.qa.jsonl`` and manifest always include **both** freshly
  processed and validly cached documents, so an incremental run never drops
  previously extracted data.
* Every artifact is written to a temp file and ``os.replace``-d into place after
  validation (atomic; no half-written outputs).
* Failures are recorded in a structured ``failures.json`` and the process exits
  **non-zero** by default. Pass ``--allow-partial`` (or ``ALLOW_PARTIAL=1``) to
  accept a partial run; the manifest then reports the failure rate and the IDs of
  the missing documents.

Set ``OVERWRITE=1`` (or ``--overwrite``) to force reprocessing of every document.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import glob
import hashlib
import json
import os
import sys
import time

# Load a local .env when present (optional dependency).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Make ``pdf_qa`` importable whether pip-installed (container) or next to this
# script (repo checkout).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LOCK_NAME = ".run_auto.lock"
_DEFAULT_LOCK_STALE_SEC = 6 * 3600


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: str, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def _atomic_write_jsonl(path: str, items) -> None:
    lines = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    _atomic_write_text(path, lines)


def _load_jsonl_pairs(path: str):
    """Return the parsed pairs, or ``None`` if the file is missing/corrupt.

    A truncated or non-JSON line means the cached output cannot be trusted and
    must be regenerated (invalidation, not a cache hit).
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError:
        return None
    pairs = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            pairs.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    return pairs


def build_contract(pdf_path: str, config, provider_name: str) -> dict:
    """The execution contract whose hash decides cache validity.

    Any change to the PDF content, resolved config, provider or persona/prompt
    ledger changes the hash and forces reprocessing.
    """
    from pdf_qa.prompts import get_persona

    persona = get_persona(config.persona)
    return {
        "input_sha256": _sha256_file(pdf_path),
        "config": dataclasses.asdict(config),
        "provider": provider_name,
        "persona": dataclasses.asdict(persona),
    }


def _run_key(contract: dict) -> str:
    return _sha256_text(_canonical(contract))


def _sidecar_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, f"{name}.qa.meta.json")


def _cache_lookup(out_path: str, sidecar_path: str, run_key: str):
    """Return cached pairs when the contract matches and output is intact, else None."""
    meta = _read_json(sidecar_path)
    if not meta or meta.get("run_key") != run_key:
        return None
    pairs = _load_jsonl_pairs(out_path)
    if pairs is None:
        return None  # corrupt / truncated / missing -> invalidate
    if meta.get("n_pairs") is not None and meta["n_pairs"] != len(pairs):
        return None
    if meta.get("jsonl_sha256") and _sha256_file(out_path) != meta["jsonl_sha256"]:
        return None
    return pairs


def _acquire_lock(out_dir: str, stale_sec: float):
    """Create an exclusive run lock; refuse to run if a fresh one exists."""
    lock_path = os.path.join(out_dir, _LOCK_NAME)
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
            os.close(fd)
            return lock_path
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                age = 0.0
            if age > stale_sec:
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
                continue  # retry once
            return None
    return None


def _release_lock(lock_path) -> None:
    if lock_path:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Zero-config PDF -> Q&A batch runner.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess every document, ignoring the cache.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Exit 0 even if some documents fail (manifest reports the failure rate).")
    return parser.parse_args([] if argv is None else argv)


def main(argv=None) -> int:
    from pdf_qa import extract_qa, get_provider, probe_device
    from pdf_qa.config import QAConfig
    from pdf_qa.manifest import build_manifest

    args = _parse_args(argv)
    overwrite = _as_bool(os.environ.get("OVERWRITE"), default=False) or args.overwrite
    allow_partial = _as_bool(os.environ.get("ALLOW_PARTIAL"), default=False) or args.allow_partial
    stale_sec = float(os.environ.get("RUN_AUTO_LOCK_STALE_SEC", _DEFAULT_LOCK_STALE_SEC))

    input_dir = os.environ.get("INPUT_DIR", "/data/input")
    output_dir = os.environ.get("OUTPUT_DIR", "/data/output")

    if not os.path.isdir(input_dir):
        sys.exit(
            f"INPUT_DIR '{input_dir}' does not exist. Set INPUT_DIR (and mount it "
            "in the container) to a folder of PDFs."
        )
    os.makedirs(output_dir, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdfs:
        sys.exit(f"No *.pdf files found in INPUT_DIR '{input_dir}'.")

    lock_path = _acquire_lock(output_dir, stale_sec)
    if lock_path is None:
        print(
            f"[auto] another run is already writing to '{output_dir}' "
            f"(remove {_LOCK_NAME} if that is stale).",
            file=sys.stderr,
        )
        return 2
    try:
        return _run_batch(
            pdfs, output_dir, overwrite, allow_partial,
            extract_qa, get_provider, probe_device, QAConfig, build_manifest,
        )
    finally:
        _release_lock(lock_path)


def _run_batch(pdfs, output_dir, overwrite, allow_partial,
               extract_qa, get_provider, probe_device, QAConfig, build_manifest) -> int:
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

    manifest = {
        "provider": provider_name,
        "persona": config.persona,
        "domain": config.domain,
        "device": dataclasses.asdict(device),
        "documents": [],
        "totals": {"documents": 0, "pairs": 0, "text": 0, "image": 0, "figures": 0},
    }
    combined: list = []
    failures: list = []
    processed = 0
    skipped = 0

    def _account(pairs, name, out_path, *, cached):
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
        doc_manifest["cached"] = cached
        doc_manifest.pop("device", None)
        manifest["documents"].append(doc_manifest)
        counts = doc_manifest["counts"]
        manifest["totals"]["documents"] += 1
        manifest["totals"]["pairs"] += counts["total"]
        manifest["totals"]["text"] += counts["text"]
        manifest["totals"]["image"] += counts["image"]
        manifest["totals"]["figures"] += counts["figures"]
        combined.extend(pairs)

    for pdf_path in pdfs:
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(output_dir, f"{name}.qa.jsonl")
        sidecar_path = _sidecar_path(output_dir, name)

        contract = build_contract(pdf_path, config, provider_name)
        run_key = _run_key(contract)

        if not overwrite:
            cached = _cache_lookup(out_path, sidecar_path, run_key)
            if cached is not None:
                print(f"[auto] skip (unchanged contract): {name}.qa.jsonl")
                _account(cached, name, out_path, cached=True)
                skipped += 1
                continue

        print(f"[auto] processing: {os.path.basename(pdf_path)}")
        tmp_out = f"{out_path}.inprogress.{os.getpid()}"
        try:
            pairs = extract_qa(pdf_path, out=tmp_out, provider_obj=llm)
            # Post-write validation: the file we are about to publish must parse
            # and match the returned pair count, or we treat the run as failed.
            written = _load_jsonl_pairs(tmp_out)
            if written is None or len(written) != len(pairs):
                raise RuntimeError("output failed post-write validation (truncated/corrupt)")
            jsonl_sha = _sha256_file(tmp_out)
            os.replace(tmp_out, out_path)
            _atomic_write_json(sidecar_path, {
                "run_key": run_key,
                "input_sha256": contract["input_sha256"],
                "n_pairs": len(pairs),
                "jsonl_sha256": jsonl_sha,
                "output": os.path.basename(out_path),
                "contract": contract,
            })
        except Exception as exc:
            if os.path.exists(tmp_out):
                try:
                    os.unlink(tmp_out)
                except OSError:
                    pass
            print(f"[auto]   ERROR on {name}: {exc}", file=sys.stderr)
            failures.append({"document": f"{name}.pdf", "error": str(exc)})
            manifest["documents"].append({"document": name, "error": str(exc)})
            continue

        _account(pairs, name, out_path, cached=False)
        processed += 1
        counts = manifest["documents"][-1]["counts"]
        print(
            f"[auto]   -> {out_path}  ({counts['total']} Q&A, "
            f"{counts['image']} image, {counts['figures']} figures)"
        )

    # Combined dataset + manifest reflect every successful (fresh or cached) doc.
    combined_path = os.path.join(output_dir, "all.qa.jsonl")
    _atomic_write_jsonl(combined_path, combined)
    print(f"[auto] combined -> {combined_path} ({len(combined)} Q&A)")

    manifest["status"] = "ok" if not failures else ("partial" if allow_partial else "failed")
    manifest["failure_rate"] = (len(failures) / len(pdfs)) if pdfs else 0.0
    manifest["missing_documents"] = [f["document"] for f in failures]
    manifest["counts_run"] = {"processed": processed, "cached": skipped, "failed": len(failures)}

    manifest_path = os.path.join(output_dir, "manifest.json")
    _atomic_write_json(manifest_path, manifest)

    failures_path = os.path.join(output_dir, "failures.json")
    if failures:
        _atomic_write_json(failures_path, {"failures": failures, "count": len(failures)})
    elif os.path.exists(failures_path):
        try:
            os.unlink(failures_path)  # clear a stale failure ledger on a clean run
        except OSError:
            pass

    print(
        f"[auto] manifest -> {manifest_path}  "
        f"(processed {processed}, cached {skipped}, failed {len(failures)} / {len(pdfs)})"
    )

    if failures and not allow_partial:
        print(
            f"[auto] FAILED: {len(failures)}/{len(pdfs)} document(s) failed. "
            f"See {failures_path}. Re-run, or pass --allow-partial to accept a partial dataset.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
