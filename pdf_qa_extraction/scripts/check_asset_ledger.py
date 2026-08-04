#!/usr/bin/env python3
"""Asset license-ledger gate (P2-3 supply-chain evidence).

Enumerates every committed *bundled binary asset* (PDF, font, image) and
enforces that each is covered by an entry in ``docs/asset_ledger.yaml``. This
makes "a missing license-ledger entry fails the release job" real, and — with
``--release`` — blocks publishing any asset whose provenance/license is still
``unresolved`` (rather than fabricating a license for it).

Usage::

    python scripts/check_asset_ledger.py --check     # CI gate: coverage + schema
    python scripts/check_asset_ledger.py --release    # stricter: also block unresolved
    python scripts/check_asset_ledger.py --list       # print the resolved inventory

Exit code is non-zero on any violation. Scanner version + scan time are printed
so results are auditable (P2-3 principle 7). No SBOM/signature/attestation is
produced or implied here — that remains future supply-chain work, so no
"attested release" claim is made.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a hard dependency
    print("[asset-ledger] ERROR: pyyaml is required (pip install pyyaml).")
    raise

SCANNER_VERSION = "asset-ledger/1"

_ROOT = Path(__file__).resolve().parents[2]  # repo root (scripts/ -> pkg -> root)
_LEDGER = _ROOT / "pdf_qa_extraction" / "docs" / "asset_ledger.yaml"

# Bundled binary asset classes the ledger governs (spec: "bundled PDF, font,
# image"). Model *weights* are gitignored (never committed) and datasets are
# runtime downloads, so both are documented as ``committed: false`` entries.
_ASSET_EXTS = {
    ".pdf",
    ".ttf", ".otf", ".woff", ".woff2",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
}
_VALID_REDIST = {"ok", "demo_only", "unresolved"}
_REQUIRED_FIELDS = ("id", "path", "kind", "license", "source", "redistribution")

# Directories to skip in the filesystem fallback (when git is unavailable).
_FALLBACK_IGNORE = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "runs", "dist",
    "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}


def _load_ledger(path: Path = _LEDGER) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data.get("assets"), list):
        raise SystemExit(f"[asset-ledger] ERROR: {path} has no 'assets' list.")
    return data


def _git_tracked_assets() -> list[str] | None:
    """Committed asset paths (repo-relative, POSIX), or None if git unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_ROOT), "ls-files", "-z"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    paths = [p for p in out.split("\0") if p]
    return [p for p in paths if os.path.splitext(p)[1].lower() in _ASSET_EXTS]


def _walk_assets() -> list[str]:
    """Filesystem fallback: bundled assets under the repo, skipping ignored dirs."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _FALLBACK_IGNORE]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in _ASSET_EXTS:
                rel = os.path.relpath(os.path.join(dirpath, name), _ROOT)
                found.append(Path(rel).as_posix())
    return found


def _committed_assets() -> list[str]:
    tracked = _git_tracked_assets()
    return sorted(tracked if tracked is not None else _walk_assets())


def _committed_entries(ledger: dict) -> list[dict]:
    return [a for a in ledger["assets"] if a.get("committed", True)]


def _validate_schema(ledger: dict) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for a in ledger["assets"]:
        aid = a.get("id", "<no-id>")
        for field in _REQUIRED_FIELDS:
            if not a.get(field):
                errors.append(f"entry {aid!r} missing required field {field!r}")
        if a.get("redistribution") not in _VALID_REDIST:
            errors.append(
                f"entry {aid!r} has invalid redistribution "
                f"{a.get('redistribution')!r} (expected {sorted(_VALID_REDIST)})"
            )
        if aid in seen_ids:
            errors.append(f"duplicate entry id {aid!r}")
        seen_ids.add(aid)
    return errors


def _matches(asset_path: str, entry: dict) -> bool:
    pattern = entry["path"]
    return asset_path == pattern or fnmatch.fnmatch(asset_path, pattern)


def audit(ledger: dict) -> dict:
    """Return {errors, warnings, coverage, unresolved} for the current tree."""
    errors = _validate_schema(ledger)
    committed = _committed_assets()
    entries = _committed_entries(ledger)

    # 1) Every committed asset must be covered by some ledger entry.
    coverage: list[tuple[str, str]] = []
    for asset in committed:
        hit = next((e for e in entries if _matches(asset, e)), None)
        if hit is None:
            errors.append(
                f"committed asset not in ledger: {asset} "
                f"(add an entry to docs/asset_ledger.yaml)"
            )
        else:
            coverage.append((asset, hit["id"]))

    # 2) Every committed-flagged ledger entry must match at least one file.
    for e in entries:
        if not any(_matches(a, e) for a in committed):
            errors.append(
                f"stale ledger entry {e['id']!r}: path {e['path']!r} matches no "
                f"committed asset"
            )

    # 3) Assets still unresolved (blocking a release bundle).
    unresolved = [
        (a, eid) for (a, eid) in coverage
        if next(e for e in entries if e["id"] == eid)["redistribution"] == "unresolved"
    ]
    return {
        "errors": errors,
        "coverage": coverage,
        "unresolved": unresolved,
        "n_committed": len(committed),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Asset license-ledger gate (P2-3).")
    ap.add_argument("--check", action="store_true",
                    help="CI gate: ledger covers every committed asset + schema ok")
    ap.add_argument("--release", action="store_true",
                    help="stricter: also fail if any committed asset is 'unresolved'")
    ap.add_argument("--list", action="store_true",
                    help="print the resolved asset->entry inventory")
    args = ap.parse_args(argv)

    ledger = _load_ledger()
    result = audit(ledger)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[asset-ledger] scanner={SCANNER_VERSION} scanned_at={now} "
          f"committed_assets={result['n_committed']} "
          f"entries={len(ledger['assets'])}")

    if args.list:
        for asset, eid in result["coverage"]:
            print(f"  {asset}  ->  {eid}")

    errors = list(result["errors"])
    if args.release and result["unresolved"]:
        for asset, eid in result["unresolved"]:
            errors.append(
                f"[release] unresolved asset blocks publish: {asset} "
                f"(entry {eid!r}) — resolve provenance/license or exclude it"
            )

    if errors:
        print(f"[asset-ledger] FAIL — {len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    scope = "release" if args.release else "check"
    print(f"[asset-ledger] OK ({scope}) — all {result['n_committed']} committed "
          f"assets covered; {len(result['unresolved'])} unresolved (not blocking "
          f"{scope}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
