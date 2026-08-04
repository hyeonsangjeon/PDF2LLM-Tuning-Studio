"""Shared run bundle: run_id, reproducibility fingerprint, artifact-set hash.

Every stage-based run (the ``pdf_native_post_training`` workflow, ``run_auto`` and
the quantization stages) registers its inputs, outputs and stage status here and
emits a single ``run_manifest.json`` that validates against
``pdf_qa/schemas/run_manifest.schema.json``.

Three identifiers are kept distinct (P0-3):

* ``run_id``                     - unique per run instance (has timestamp entropy)
* ``reproducibility_fingerprint``- stable over timestamp / output path / run_id /
  absolute local paths; equal inputs+code+config+model => equal value
* ``artifact_set_hash``          - canonical hash of the produced artifact set

The manifest never embeds itself in its own hash (no circular hash) and never
stores secrets, cloud account IDs, endpoints or absolute local paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "run_manifest/v1"
_NOT_RECORDED = "not_recorded"


# --------------------------------------------------------------------------- #
# hashing helpers                                                             #
# --------------------------------------------------------------------------- #
def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding (sorted keys, compact, UTF-8)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_canonical(obj: Any) -> str:
    return sha256_bytes(canonical_bytes(obj))


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# environment / git                                                            #
# --------------------------------------------------------------------------- #
def git_info(cwd: Optional[str] = None) -> Dict[str, Any]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(["git"] + args, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    sha = _run(["rev-parse", "HEAD"]) or _NOT_RECORDED
    status = _run(["status", "--porcelain"])
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else False}


def _pkg_versions(names: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n in names:
        try:
            mod = __import__(n)
            out[n] = getattr(mod, "__version__", _NOT_RECORDED)
        except Exception:  # noqa: BLE001 - best effort, never fatal
            out[n] = _NOT_RECORDED
    return out


def environment_info(packages: Optional[List[str]] = None) -> Dict[str, Any]:
    packages = packages or []
    gpu = _NOT_RECORDED
    cuda = _NOT_RECORDED
    try:  # torch is optional; keep import lazy and non-fatal
        import torch  # type: ignore

        cuda = getattr(torch.version, "cuda", None) or _NOT_RECORDED
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return {
        "python": platform.python_version(),
        "packages": _pkg_versions(packages),
        "cuda": cuda,
        "driver": _NOT_RECORDED,
        "gpu": gpu,
    }


# --------------------------------------------------------------------------- #
# run bundle                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class RunBundle:
    """Accumulates run state and renders the shared run manifest."""

    run_id: str = field(default_factory=new_run_id)
    command: str = ""
    generation_mode: str = _NOT_RECORDED
    created_at_utc: str = field(default_factory=utc_now)
    base_dir: str = "."  # input paths are stored relative to this
    out_base_dir: Optional[str] = None  # output paths relative to this (run dir); defaults to base_dir
    code: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    dataset: Dict[str, Any] = field(default_factory=dict)
    container: Dict[str, Any] = field(default_factory=lambda: {"image_digest": None})
    environment: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=list)
    provider_usage: List[Dict[str, Any]] = field(default_factory=list)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- registration -------------------------------------------------------- #
    def _rel(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.base_dir)
        except ValueError:
            return os.path.basename(path)

    def _rel_out(self, path: str) -> str:
        base = self.out_base_dir or self.base_dir
        try:
            return os.path.relpath(path, base)
        except ValueError:
            return os.path.basename(path)

    def add_input(self, path: str, role: Optional[str] = None, sha256: Optional[str] = None) -> None:
        self.inputs.append({"path": self._rel(path), "sha256": sha256 or sha256_file(path), "role": role})

    def add_output(self, path: str, sha256: Optional[str] = None) -> None:
        self.outputs.append({"path": self._rel_out(path), "sha256": sha256 or sha256_file(path)})

    def add_stage(
        self,
        name: str,
        status: str,
        started_at_utc: Optional[str] = None,
        ended_at_utc: Optional[str] = None,
        input_sha256: Optional[str] = None,
        output_sha256: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.stages.append(
            {
                "name": name,
                "status": status,
                "started_at_utc": started_at_utc,
                "ended_at_utc": ended_at_utc,
                "input_sha256": input_sha256,
                "output_sha256": output_sha256,
                "error": error,
            }
        )

    def set_code(self, config_sha256=None, prompt_sha256=None, rubric_sha256=None, cwd=None) -> None:
        info = git_info(cwd)
        info.update({"config_sha256": config_sha256, "prompt_sha256": prompt_sha256, "rubric_sha256": rubric_sha256})
        self.code = info

    # -- derived hashes ------------------------------------------------------ #
    def reproducibility_fingerprint(self) -> str:
        """Stable identity of *what was run* (not *when* or *where written*)."""
        projection = {
            "inputs": sorted(({"role": i.get("role"), "sha256": i["sha256"]} for i in self.inputs), key=canonical_bytes),
            "code": {
                "git_sha": self.code.get("git_sha"),
                "config_sha256": self.code.get("config_sha256"),
                "prompt_sha256": self.code.get("prompt_sha256"),
                "rubric_sha256": self.code.get("rubric_sha256"),
            },
            "model": self.model,
            "dataset": self.dataset,
            "seeds": sorted(self.seeds),
            "command": self.command,
            "generation_mode": self.generation_mode,
        }
        return sha256_canonical(projection)

    def artifact_set_hash(self) -> str:
        items = sorted(({"path": o["path"], "sha256": o["sha256"]} for o in self.outputs), key=canonical_bytes)
        return sha256_canonical(items)

    # -- render / persist ---------------------------------------------------- #
    def to_manifest(self) -> Dict[str, Any]:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "reproducibility_fingerprint": self.reproducibility_fingerprint(),
            "artifact_set_hash": self.artifact_set_hash(),
            "created_at_utc": self.created_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "command": self.command,
            "generation_mode": self.generation_mode,
            "code": self.code or {"git_sha": _NOT_RECORDED, "git_dirty": False},
            "model": self.model,
            "dataset": self.dataset,
            "container": self.container,
            "environment": self.environment or {"python": platform.python_version(), "packages": {}},
            "seeds": self.seeds,
            "provider_usage": self.provider_usage,
            "stages": self.stages,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        manifest.update(self.extra)
        return manifest

    def write(self, run_dir: str, name: str = "run_manifest.json") -> str:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, name)
        atomic_write_json(path, self.to_manifest())
        return path


# --------------------------------------------------------------------------- #
# atomic write + schema validation                                             #
# --------------------------------------------------------------------------- #
def atomic_write_json(path: str, obj: Any) -> None:
    """Write to a temp file in the same dir then atomically rename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(__file__), "schemas", "run_manifest.schema.json")


def load_run_manifest_schema() -> Dict[str, Any]:
    with open(_schema_path(), encoding="utf-8") as fh:
        return json.load(fh)


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable validation errors ([] == valid)."""
    try:
        import jsonschema  # type: ignore
    except Exception:  # noqa: BLE001 - jsonschema is a light dep but keep optional
        return _minimal_validate(manifest)
    schema = load_run_manifest_schema()
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{list(e.path)}: {e.message}" for e in validator.iter_errors(manifest)]


def _minimal_validate(manifest: Dict[str, Any]) -> List[str]:
    required = [
        "schema_version", "run_id", "reproducibility_fingerprint", "artifact_set_hash",
        "created_at_utc", "command", "code", "environment", "stages", "inputs", "outputs",
    ]
    return [f"missing required field: {k}" for k in required if k not in manifest]


if __name__ == "__main__":  # tiny smoke
    rb = RunBundle(command="python -m pdf_qa.run_bundle", generation_mode="recorded_replay")
    rb.set_code(config_sha256="deadbeef")
    rb.environment = environment_info()
    rb.add_stage("demo", "completed")
    m = rb.to_manifest()
    errs = validate_manifest(m)
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("valid" if not errs else f"errors: {errs}", file=sys.stderr)
