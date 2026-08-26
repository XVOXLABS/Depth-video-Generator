from __future__ import annotations

import subprocess
from pathlib import Path


def make_video(
    path: Path,
    seconds: float = 1.0,
    fps: int = 8,
    size: str = "320x240",
    audio: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={seconds}:size={size}:rate={fps}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path
