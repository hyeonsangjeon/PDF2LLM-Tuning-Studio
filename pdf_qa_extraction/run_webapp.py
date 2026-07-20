#!/usr/bin/env python3
"""Launch the single-node local demo web app.

The web app runs the PDF -> Q&A pipeline **in-process** (it imports ``pdf_qa``
and calls it directly), so nothing spawns a container per request and the GPU
auto-detection works exactly as in a normal run. Start it with::

    # locally (install the extra once): pip install .[webapp]
    python run_webapp.py

    # or from the public image (GPU demo):
    docker run --rm --gpus all -p 8000:8000 --env-file .env \
        ghcr.io/hyeonsangjeon/pdf2llm-tuning-studio/pdf-qa-extractor:latest-gpu \
        python run_webapp.py

Then open http://localhost:8000. Configure ``HOST``/``PORT`` via env if needed.

Two processing modes -- pick one with the ``WORKERS`` env var:

* **single-node / in-process** (default, ``WORKERS=1``): one process owns the
  GPU, so torch/onnxruntime GPU auto-detection is unambiguous. Best for the GPU
  demo and the simplest thing to reason about.
* **multi-process** (``WORKERS=N``): N independent worker processes behind one
  port for higher request concurrency (mainly CPU-bound preview / throughput).
  On a single GPU, keep ``WORKERS=1`` so the workers don't contend for VRAM::

      WORKERS=4 python run_webapp.py            # 4-worker CPU demo
      docker run --rm -e WORKERS=4 -p 8000:8000 $IMG python run_webapp.py
"""

from __future__ import annotations

import os
import sys

# Load a local .env when present (optional dependency).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Make ``pdf_qa`` and ``webapp`` importable whether pip-installed (container) or
# sitting next to this script (repo checkout).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    try:
        import uvicorn  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "uvicorn/fastapi is not installed. Install the web extra with "
            "`pip install .[webapp]` (the container image already ships it)."
        )

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # WORKERS selects the processing mode: 1 = single-node in-process (default,
    # GPU-friendly); N>1 = multi-process (N worker processes, higher CPU-bound
    # concurrency). See the module docstring for guidance.
    workers = max(1, int(os.getenv("WORKERS", "1") or "1"))
    mode = (
        "single-node (in-process)"
        if workers == 1
        else f"multi-process ({workers} workers)"
    )
    print(
        f"[webapp] Serving PDF2LLM local demo on http://{host}:{port} "
        f"— mode: {mode}"
    )
    import uvicorn

    if workers > 1:
        uvicorn.run(
            "webapp.app:app", host=host, port=port, log_level="info", workers=workers
        )
    else:
        uvicorn.run("webapp.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
