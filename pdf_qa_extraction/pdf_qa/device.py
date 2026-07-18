"""Runtime device / accelerator probe.

Reports whether the process can actually reach an NVIDIA GPU and surfaces it in
the logs, so the GPU advantage of the ``:latest-gpu`` image is *visible* and the
extraction pipeline can escalate to the heavier, GPU-accelerated path (hi_res
layout on ``onnxruntime-gpu`` + Table Transformer on CUDA torch).

Design notes
------------
* The authoritative "is a GPU usable right now?" signal is
  ``torch.cuda.is_available()`` -- torch actually probes the NVIDIA driver
  (``libcuda``), which is only injected into the container by ``--gpus all`` /
  nvidia-container-toolkit. We gate GPU behaviour on this.
* ``onnxruntime.get_available_providers()`` lists **compile-time** providers, so
  ``CUDAExecutionProvider`` shows up even on a CPU-only host when the
  ``onnxruntime-gpu`` wheel is installed. It therefore tells us *which image*
  we're in (``onnxruntime-gpu`` present) but **not** whether a GPU is reachable.
  We report both, and only trust the driver signal for routing.
* Every import is defensive: this module must import on the CPU image (and in a
  bare test venv) where torch / onnxruntime may be absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DeviceReport:
    """Snapshot of the accelerators visible to this process."""

    torch_installed: bool = False
    torch_version: Optional[str] = None
    torch_cuda_available: bool = False
    cuda_device_name: Optional[str] = None
    cuda_device_count: int = 0

    onnxruntime_installed: bool = False
    onnxruntime_version: Optional[str] = None
    onnxruntime_gpu_package: bool = False
    onnxruntime_providers: List[str] = field(default_factory=list)
    onnxruntime_cuda_provider: bool = False

    @property
    def gpu_ready(self) -> bool:
        """True when a GPU is actually reachable (NVIDIA driver present).

        Gated on torch's driver probe. Because ``onnxruntime-gpu`` and CUDA
        torch share the same NVIDIA driver, this also implies onnxruntime-gpu's
        ``CUDAExecutionProvider`` will work at session time.
        """
        return self.torch_cuda_available

    def summary(self) -> str:
        """A one-block, human-readable (Korean) summary for the run log."""
        lines = ["===== 디바이스 점검 (GPU/CPU) ====="]
        if self.torch_installed:
            if self.torch_cuda_available:
                lines.append(
                    f"- torch {self.torch_version}: CUDA 사용 가능 "
                    f"({self.cuda_device_name}, {self.cuda_device_count}개)"
                )
            else:
                lines.append(
                    f"- torch {self.torch_version}: CUDA 미사용 (CPU) "
                    "- 드라이버 미탐지 / `--gpus all` 없음 / CPU 이미지"
                )
        else:
            lines.append("- torch: 미설치")

        if self.onnxruntime_installed:
            pkg = "onnxruntime-gpu" if self.onnxruntime_gpu_package else "onnxruntime(CPU)"
            cuda = "CUDAExecutionProvider 노출" if self.onnxruntime_cuda_provider else "CPU providers"
            lines.append(
                f"- onnxruntime {self.onnxruntime_version} [{pkg}]: {cuda} "
                f"({', '.join(self.onnxruntime_providers) or 'none'})"
            )
        else:
            lines.append("- onnxruntime: 미설치")

        if self.gpu_ready:
            lines.append(
                ">> GPU 감지: hi_res 레이아웃(onnxruntime-gpu) + 표 구조(CUDA torch)를 "
                "GPU로 가속합니다."
            )
        elif self.onnxruntime_gpu_package and not self.torch_cuda_available:
            lines.append(
                ">> GPU 이미지지만 드라이버 미탐지 → CPU로 실행합니다 "
                "(`docker run --gpus all ...`로 실행하세요)."
            )
        else:
            lines.append(">> CPU 모드로 실행합니다 (경량 경로).")
        lines.append("=" * 34)
        return "\n".join(lines)


def _dist_names() -> set:
    """Installed distribution names, lower-cased (empty set on failure)."""
    try:
        import importlib.metadata as md

        return {d.metadata["Name"].lower() for d in md.distributions() if d.metadata["Name"]}
    except Exception:  # pragma: no cover - defensive
        return set()


def probe_device() -> DeviceReport:
    """Inspect torch / onnxruntime and return a :class:`DeviceReport`.

    Never raises: missing packages simply leave the corresponding fields at
    their defaults.
    """
    report = DeviceReport()
    dists = _dist_names()
    report.onnxruntime_gpu_package = "onnxruntime-gpu" in dists

    # --- torch (authoritative GPU signal) ---
    try:
        import torch

        report.torch_installed = True
        report.torch_version = getattr(torch, "__version__", None)
        try:
            if torch.cuda.is_available():
                report.torch_cuda_available = True
                report.cuda_device_count = torch.cuda.device_count()
                report.cuda_device_name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - driver/query hiccup -> treat as CPU
            report.torch_cuda_available = False
    except Exception:
        report.torch_installed = False

    # --- onnxruntime (informational: which image / compiled providers) ---
    try:
        import onnxruntime as ort

        report.onnxruntime_installed = True
        report.onnxruntime_version = getattr(ort, "__version__", None)
        try:
            providers = list(ort.get_available_providers())
        except Exception:  # pragma: no cover - defensive
            providers = []
        report.onnxruntime_providers = providers
        report.onnxruntime_cuda_provider = "CUDAExecutionProvider" in providers
    except Exception:
        report.onnxruntime_installed = False

    return report
