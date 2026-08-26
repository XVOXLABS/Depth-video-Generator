"""Configure a fake backend and isolated runtime dirs before app modules import."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="depth-video-test-"))
os.environ["DEPTH_VIDEO_BACKEND"] = "fake"
os.environ["DEPTH_VIDEO_CHECKPOINTS"] = str(ROOT / "checkpoints")
os.environ["DEPTH_VIDEO_UPLOADS"] = str(ROOT / "uploads")
os.environ["DEPTH_VIDEO_OUTPUTS"] = str(ROOT / "outputs")
os.environ["DEPTH_VIDEO_JOBS"] = str(ROOT / "jobs")
os.environ["DEPTH_VIDEO_DEVICE"] = "cpu"
