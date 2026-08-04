#!/usr/bin/env python3
"""Build a wheel, install it into a throwaway venv, and prove it is usable.

Verifies P0-7 packaging: the wheel installs, the ``pdf2llm`` console script is on
PATH, the JSON schemas and the synthetic demo fixture are packaged (importable
from site-packages, not the source tree), and ``pdf2llm verify-demo`` passes from
an unrelated working directory.

Offline-friendly: the wheel is built with the *current* interpreter's setuptools
(``--no-build-isolation``) and the venv inherits system site-packages so runtime
dependencies resolve without a network round-trip.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import venv

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pdf_qa_extraction


def _run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kw)


def _clean_build_droppings() -> None:
    for name in ("build", "dist", "UNKNOWN.egg-info", "pdf_qa.egg-info", "pdf_qa.egg-info".replace("_", "-")):
        p = os.path.join(PKG_ROOT, name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    _clean_build_droppings()
    # A neutral env so child imports resolve to site-packages, never the source tree.
    child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    with tempfile.TemporaryDirectory() as tmp:
        # 1) build the wheel with the current interpreter's setuptools (offline).
        dist = os.path.join(tmp, "dist")
        _run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
              "-w", dist, "."], cwd=PKG_ROOT)
        wheels = [f for f in os.listdir(dist) if f.endswith(".whl")]
        if not wheels or wheels[0].startswith("UNKNOWN"):
            print(f"FAIL: expected a named wheel, got {wheels}", file=sys.stderr)
            return 1
        wheel = os.path.join(dist, wheels[0])
        print("built", wheels[0])

        # 2) fresh venv that inherits system deps; install just the wheel.
        env_dir = os.path.join(tmp, "venv")
        venv.create(env_dir, with_pip=True, system_site_packages=True)
        vpy = os.path.join(env_dir, "bin", "python")
        vbin = os.path.join(env_dir, "bin")
        _run([vpy, "-m", "pip", "install", "-q", "--no-deps", wheel])

        # 3) schemas + fixture are packaged (loaded from site-packages, not source).
        workdir = os.path.join(tmp, "work")
        os.makedirs(workdir)
        check = (
            "import os, workflows.pdf_native_post_training as w, pdf_qa;"
            "d=os.path.dirname(w.__file__);"
            "fx=os.path.join(d,'public_finance_demo','gold_qa.jsonl');"
            "sc=os.path.join(os.path.dirname(pdf_qa.__file__),'schemas','run_manifest.schema.json');"
            "assert os.path.isfile(fx), fx; assert os.path.isfile(sc), sc;"
            "assert 'site-packages' in d, ('not installed from site-packages: '+d);"
            "print('packaged fixture + schema OK from site-packages')"
        )
        _run([vpy, "-c", check], cwd=workdir, env=child_env)

        # 4) console script works and the demo passes from an unrelated cwd.
        _run([os.path.join(vbin, "pdf2llm"), "verify-demo"], cwd=workdir, env=child_env)

    print("\n[verify_wheel_install] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
