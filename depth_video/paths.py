from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = Path(
    __import__("os").environ.get("DEPTH_VIDEO_CHECKPOINTS", str(ROOT / "checkpoints"))
)
UPLOADS_DIR = Path(__import__("os").environ.get("DEPTH_VIDEO_UPLOADS", str(ROOT / "uploads")))
OUTPUTS_DIR = Path(__import__("os").environ.get("DEPTH_VIDEO_OUTPUTS", str(ROOT / "outputs")))
JOBS_DIR = Path(__import__("os").environ.get("DEPTH_VIDEO_JOBS", str(ROOT / "jobs")))
STATIC_DIR = ROOT / "static"


def ensure_runtime_dirs() -> None:
    for path in (CHECKPOINTS_DIR, UPLOADS_DIR, OUTPUTS_DIR, JOBS_DIR):
        path.mkdir(parents=True, exist_ok=True)
