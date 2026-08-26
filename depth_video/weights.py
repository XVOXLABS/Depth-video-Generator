from __future__ import annotations

from pathlib import Path
from typing import Callable

from .paths import CHECKPOINTS_DIR

ProgressCallback = Callable[[str, float | None], None]

ENCODERS = ("vits", "vitb", "vitl")

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

WEIGHT_SPECS = {
    ("vits", False): (
        "depth-anything/Video-Depth-Anything-Small",
        "video_depth_anything_vits.pth",
        "Apache-2.0",
    ),
    ("vitb", False): (
        "depth-anything/Video-Depth-Anything-Base",
        "video_depth_anything_vitb.pth",
        "CC-BY-NC-4.0",
    ),
    ("vitl", False): (
        "depth-anything/Video-Depth-Anything-Large",
        "video_depth_anything_vitl.pth",
        "CC-BY-NC-4.0",
    ),
    ("vits", True): (
        "depth-anything/Metric-Video-Depth-Anything-Small",
        "metric_video_depth_anything_vits.pth",
        "Apache-2.0",
    ),
    ("vitb", True): (
        "depth-anything/Metric-Video-Depth-Anything-Base",
        "metric_video_depth_anything_vitb.pth",
        "CC-BY-NC-4.0",
    ),
    ("vitl", True): (
        "depth-anything/Metric-Video-Depth-Anything-Large",
        "metric_video_depth_anything_vitl.pth",
        "CC-BY-NC-4.0",
    ),
}


def local_checkpoint_path(encoder: str, metric: bool) -> Path:
    _, filename, _ = WEIGHT_SPECS[(encoder, metric)]
    return CHECKPOINTS_DIR / filename


def ensure_checkpoint(
    encoder: str,
    metric: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from {ENCODERS}.")

    dest = local_checkpoint_path(encoder, metric)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        if progress:
            progress(f"Using cached weights: {dest.name}", None)
        return dest

    repo_id, filename, license_id = WEIGHT_SPECS[(encoder, metric)]
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"Downloading {filename} ({license_id})…", None)

    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(CHECKPOINTS_DIR),
        )
    )
    if downloaded.resolve() != dest.resolve() and downloaded.exists() and not dest.exists():
        dest.write_bytes(downloaded.read_bytes())
    if progress:
        progress(f"Weights ready: {dest.name}", None)
    return dest if dest.exists() else downloaded
