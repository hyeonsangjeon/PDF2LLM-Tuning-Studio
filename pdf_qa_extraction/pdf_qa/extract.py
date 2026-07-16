"""PDF partitioning helpers built on `unstructured` (GPU accelerated).

`unstructured` is imported lazily inside :func:`extract_elements_from_pdf` so
the rest of the package (providers, parsing, tests) can be imported without the
heavy PDF/OCR dependency stack installed.
"""

from __future__ import annotations

import base64
import glob
import os
from typing import List, Optional

_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")


def extract_elements_from_pdf(
    filepath: str,
    table_model: Optional[str] = None,
    figures_dir: str = "figures",
) -> list:
    """Extract text/image/table elements from a PDF.

    Args:
        filepath: Path to the PDF file.
        table_model: Table detection model (``yolox``, ``tatr``,
            ``table-transformer``, ``detectron2``, ...). ``None`` disables
            table-structure inference.
        figures_dir: Directory where extracted image blocks are written.

    Returns:
        The list of elements produced by ``unstructured.partition.pdf``.
    """
    # Imported here so importing this module never requires unstructured.
    from unstructured.partition.pdf import partition_pdf

    partition_kwargs = {
        "filename": filepath,
        "extract_images_in_pdf": True,
        "chunking_strategy": "by_title",
        "max_characters": 4000,
        "new_after_n_chars": 3800,
        "combine_text_under_n_chars": 2000,
        "extract_image_block_output_dir": figures_dir,
        # "auto" -> fast for digital PDFs, high_res (layout detection + OCR) for scans.
        "strategy": "auto",
    }

    if table_model:
        partition_kwargs["infer_table_structure"] = True
        partition_kwargs["table_model"] = table_model
    else:
        partition_kwargs["infer_table_structure"] = False

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
