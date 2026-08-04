"""Minimal stage harness: resume-by-content-hash, atomic writes, partial=fail.

A pipeline is an ordered list of :class:`Stage`. Each stage exposes a
``signature`` (a JSON-able description of everything that determines its output:
input content hashes + config) and a ``run`` that produces a JSON-able output.

The harness:

* hashes each stage's signature and, if a prior output for the same stage with a
  matching signature hash exists in the run dir, **skips** recomputation
  (resume-by-hash, not by filename);
* writes each stage output atomically (temp file + ``os.replace``);
* on any stage error, records the failure in the manifest and re-raises so the
  process exits non-zero (**partial completion is a failure**);
* keeps both skipped and freshly-computed stages in the run manifest.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from pdf_qa.run_bundle import (
    RunBundle,
    atomic_write_json,
    sha256_canonical,
    utc_now,
)


@dataclass
class StageContext:
    config: Dict[str, Any]
    run_dir: str
    base_dir: str
    outputs: Dict[str, Any]
    bundle: RunBundle


@dataclass
class Stage:
    name: str
    signature: Callable[[StageContext], Any]
    run: Callable[[StageContext], Any]


class StageError(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"stage {stage!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


def _stage_path(run_dir: str, name: str) -> str:
    return os.path.join(run_dir, "stages", f"{name}.json")


def run_pipeline(stages: List[Stage], ctx: StageContext, *, resume: bool = True) -> Dict[str, Any]:
    os.makedirs(os.path.join(ctx.run_dir, "stages"), exist_ok=True)
    summary: Dict[str, Any] = {"stages": []}

    for stage in stages:
        sig_hash = sha256_canonical(stage.signature(ctx))
        out_path = _stage_path(ctx.run_dir, stage.name)

        # resume-by-hash: reuse a prior output only if its signature matches
        if resume and os.path.isfile(out_path):
            try:
                import json

                with open(out_path, encoding="utf-8") as fh:
                    prior = json.load(fh)
            except Exception:  # noqa: BLE001
                prior = None
            if prior and prior.get("_signature") == sig_hash:
                ctx.outputs[stage.name] = prior["output"]
                ctx.bundle.add_stage(
                    stage.name, "skipped", started_at_utc=utc_now(), ended_at_utc=utc_now(),
                    input_sha256=sig_hash, output_sha256=sha256_canonical(prior["output"]),
                )
                summary["stages"].append({"name": stage.name, "status": "skipped"})
                continue

        started = utc_now()
        try:
            output = stage.run(ctx)
        except Exception as exc:  # noqa: BLE001 - convert to fail + non-zero exit
            ctx.bundle.add_stage(
                stage.name, "failed", started_at_utc=started, ended_at_utc=utc_now(),
                input_sha256=sig_hash, error=f"{type(exc).__name__}: {exc}",
            )
            summary["stages"].append({"name": stage.name, "status": "failed"})
            _finalize(ctx, summary, ok=False)
            raise StageError(stage.name, exc) from exc

        atomic_write_json(out_path, {"_signature": sig_hash, "output": output})
        ctx.outputs[stage.name] = output
        ctx.bundle.add_stage(
            stage.name, "completed", started_at_utc=started, ended_at_utc=utc_now(),
            input_sha256=sig_hash, output_sha256=sha256_canonical(output),
        )
        summary["stages"].append({"name": stage.name, "status": "completed"})

    _finalize(ctx, summary, ok=True)
    return summary


def _finalize(ctx: StageContext, summary: Dict[str, Any], ok: bool) -> None:
    # register stage outputs as run outputs (relative paths only)
    for stage in summary["stages"]:
        p = _stage_path(ctx.run_dir, stage["name"])
        if os.path.isfile(p):
            ctx.bundle.add_output(p)
    ctx.bundle.extra["status"] = "completed" if ok else "failed"
    ctx.bundle.write(ctx.run_dir)
