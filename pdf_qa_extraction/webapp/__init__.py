"""Single-node local demo web app for the PDF -> Q&A pipeline.

This package is an *optional* thin UI layer on top of the :mod:`pdf_qa` core. It
runs the pipeline **in-process** (no docker-in-docker, no per-request container
spawning): the same Python process that serves the page also imports ``pdf_qa``
and calls it directly, so the GPU auto-detection (:func:`pdf_qa.probe_device`)
"just works" -- when the container is started with ``--gpus all`` the extraction
escalates to the GPU path automatically.

Install the extra deps with ``pip install .[webapp]`` (or use the container
image, which ships them) and launch with ``python run_webapp.py``.
"""

from .app import app, create_app

__all__ = ["app", "create_app"]
