from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeDevice:
    kind: str  # cuda | mps | cpu
    name: str
    torch_device: str
    supports_fp16: bool


def detect_device() -> RuntimeDevice:
    override = os.environ.get("DEPTH_VIDEO_DEVICE", "").strip().lower()
    try:
        import torch
    except ImportError:
        return RuntimeDevice("cpu", "CPU (PyTorch not installed)", "cpu", False)

    if override in {"cpu", "cuda", "mps"}:
        if override == "cuda" and not torch.cuda.is_available():
            override = "cpu"
        if override == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            override = "cpu"
        if override == "cuda":
            name = torch.cuda.get_device_name(0)
            return RuntimeDevice("cuda", name, "cuda", True)
        if override == "mps":
            return RuntimeDevice("mps", "Apple Silicon", "mps", False)
        return RuntimeDevice("cpu", "CPU", "cpu", False)

    if torch.cuda.is_available():
        return RuntimeDevice("cuda", torch.cuda.get_device_name(0), "cuda", True)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return RuntimeDevice("mps", "Apple Silicon", "mps", False)
    return RuntimeDevice("cpu", "CPU", "cpu", False)
