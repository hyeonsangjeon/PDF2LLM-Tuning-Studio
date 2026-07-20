"""PDF partitioning helpers built on `unstructured`.

The default container image (``:latest``) is CPU-only, so extraction runs on
CPU. Engines differ per stage: the ``hi_res`` **layout** model (default
``yolox``) runs on ``onnxruntime``; scanned pages are OCR'd with ``tesseract``
(CPU only); the **table-structure** model (Table Transformer,
``infer_table_structure=True``) is PyTorch-based. The ``:latest-gpu`` image
ships CUDA torch **and** ``onnxruntime-gpu``, so both the layout model
(onnxruntime-gpu) and the table model (CUDA torch) run on the GPU when launched
with ``--gpus all``; only Tesseract OCR and pdfminer stay on CPU. See the module
README "PDF parsing models -- GPU/CPU" for the full breakdown.

:func:`resolve_extraction_plan` makes this concrete and *automatic*: it probes
the device (see :mod:`pdf_qa.device`) and, when a GPU is actually reachable,
escalates ``strategy="auto"`` to ``"hi_res"`` and enables table-structure
inference so both heavy models run on the GPU. On CPU it keeps the light path.

`unstructured` is imported lazily inside :func:`extract_elements_from_pdf` so
the rest of the package (providers, parsing, tests) can be imported without the
heavy PDF/OCR dependency stack installed.
"""

from __future__ import annotations

import base64
import glob
import os
from typing import List, Optional

from .device import DeviceReport, probe_device

_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")


def resolve_extraction_plan(
    strategy: str = "auto",
    hi_res_model_name: Optional[str] = None,
    gpu_boost: bool = True,
    device: Optional[DeviceReport] = None,
) -> dict:
    """Decide the partitioning strategy, escalating to the GPU path when able.

    This is where the GPU advantage becomes concrete: when a GPU is actually
    reachable (``device.gpu_ready``) and ``gpu_boost`` is on, we force the
    ``hi_res`` strategy (so the ONNX **layout** model runs -- on the
    ``onnxruntime-gpu`` CUDA provider in the GPU image) and turn on
    ``infer_table_structure`` (so the PyTorch **Table Transformer** runs on CUDA
    torch). Both are far too slow to enable by default on CPU, so on a CPU host
    the pipeline stays on the light path.

    ``hi_res_model_name`` selects the layout-detection model (``yolox`` default,
    ``detectron2_onnx``, ``yolox_quantized`` ...). Because that model is only
    consulted under the ``hi_res`` strategy, an explicit choice **forces**
    ``hi_res`` (otherwise unstructured would silently ignore it). An explicit
    ``strategy`` other than ``"auto"`` is always respected -- the boost and the
    model selection only fill in / escalate defaults.

    Returns a dict with keys ``strategy``, ``infer_table_structure``,
    ``hi_res_model_name`` and ``gpu_accelerated``.
    """
    report = device if device is not None else probe_device()

    effective_strategy = strategy
    # Selecting a layout model only makes sense on the hi_res path, and it
    # implies we want the richer (layout + table) extraction. Use truthiness
    # (not ``is not None``) so an empty string behaves exactly like ``None`` --
    # otherwise it would enable table inference without escalating to hi_res,
    # yielding an inconsistent plan.
    infer_tables = bool(hi_res_model_name)
    gpu_accelerated = False

    # An explicit model must actually take effect: escalate auto -> hi_res so
    # unstructured consults ``hi_res_model_name`` instead of ignoring it.
    if hi_res_model_name and effective_strategy == "auto":
        effective_strategy = "hi_res"

    if gpu_boost and report.gpu_ready:
        gpu_accelerated = True
        if effective_strategy == "auto":
            # Force the layout model even for digital PDFs -> exercises the GPU.
            effective_strategy = "hi_res"
        # Turn table-structure inference on if the caller did not opt out.
        infer_tables = True
        print(
            "[extract] GPU 감지 → hi_res 레이아웃(onnxruntime-gpu) + "
            "표 구조 추론(CUDA torch)을 GPU로 실행합니다."
            + (f" 레이아웃 모델={hi_res_model_name}." if hi_res_model_name else "")
        )
    else:
        print(
            f"[extract] CPU 경로 → strategy={effective_strategy}, "
            f"표 추론={infer_tables}, 레이아웃 모델={hi_res_model_name or '기본(yolox)'}."
        )

    return {
        "strategy": effective_strategy,
        "infer_table_structure": infer_tables,
        "hi_res_model_name": hi_res_model_name,
        "gpu_accelerated": gpu_accelerated,
    }


def extract_elements_from_pdf(
    filepath: str,
    hi_res_model_name: Optional[str] = None,
    figures_dir: str = "figures",
    strategy: str = "auto",
    gpu_boost: bool = True,
    device: Optional[DeviceReport] = None,
) -> list:
    """Extract text/image/table elements from a PDF.

    Args:
        filepath: Path to the PDF file.
        hi_res_model_name: Layout-detection model for the ``hi_res`` path
            (``yolox`` default, ``yolox_quantized``, ``detectron2_onnx`` ...).
            Passed to ``unstructured`` as ``hi_res_model_name`` and, when set,
            escalates ``strategy="auto"`` to ``hi_res`` so the choice actually
            takes effect. ``None`` uses unstructured's default model.
        figures_dir: Directory where extracted image blocks are written.
        strategy: ``unstructured`` strategy (``auto`` | ``fast`` | ``hi_res`` |
            ``ocr_only``). ``auto`` is escalated to ``hi_res`` when a GPU is
            reachable and ``gpu_boost`` is on, or when a model is selected.
        gpu_boost: When True (default), route the heavy layout + table models to
            the GPU whenever one is detected. Set False to keep the light path.
        device: Pre-computed :class:`~pdf_qa.device.DeviceReport` (probed if
            omitted) so the pipeline can log it once and reuse it here.

    Returns:
        The list of elements produced by ``unstructured.partition.pdf``.
    """
    # Imported here so importing this module never requires unstructured.
    from unstructured.partition.pdf import partition_pdf

    plan = resolve_extraction_plan(
        strategy=strategy,
        hi_res_model_name=hi_res_model_name,
        gpu_boost=gpu_boost,
        device=device,
    )

    partition_kwargs = {
        "filename": filepath,
        "extract_images_in_pdf": True,
        "chunking_strategy": "by_title",
        "max_characters": 4000,
        "new_after_n_chars": 3800,
        "combine_text_under_n_chars": 2000,
        "extract_image_block_output_dir": figures_dir,
        # "auto" -> fast for digital PDFs, hi_res (layout + OCR) for scans;
        # escalated to hi_res by the GPU boost so the layout model runs on GPU.
        "strategy": plan["strategy"],
        "infer_table_structure": plan["infer_table_structure"],
    }

    # Select the layout-detection model only when the caller asked for one.
    # ``hi_res_model_name`` is the correct unstructured knob (the older
    # ``model_name`` is deprecated); it is consulted under the hi_res strategy.
    if plan["hi_res_model_name"]:
        partition_kwargs["hi_res_model_name"] = plan["hi_res_model_name"]

    return partition_pdf(**partition_kwargs)


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Return the base64 string for an image file, or ``None`` on failure."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"이미지 인코딩 에러 {image_path}: {exc}")
        return None


def get_extracted_images(figures_dir: str = "figures") -> List[str]:
    """List image files written by the partitioner, sorted by name."""
    if not os.path.exists(figures_dir):
        return []

    image_files: List[str] = []
    for extension in _IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(figures_dir, extension)))
    return sorted(image_files)
