from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_HOMEBREW_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/homebrew/sbin",
    "/opt/local/bin",
)
_EXTRA_DIRS = (
    "/usr/bin",
    "/bin",
    str(Path.home() / ".local" / "bin"),
)


def _candidate_dirs() -> list[str]:
    dirs: list[str] = []
    for item in (*_HOMEBREW_DIRS, *_EXTRA_DIRS):
        if item not in dirs and Path(item).is_dir():
            dirs.append(item)
    return dirs


def ensure_ffmpeg_on_path() -> None:
    """GUI-launched apps on macOS often miss Homebrew's PATH. Put it first."""
    extras = [d for d in _candidate_dirs() if d not in os.environ.get("PATH", "").split(os.pathsep)]
    if extras:
        os.environ["PATH"] = os.pathsep.join(extras) + os.pathsep + os.environ.get("PATH", "")


def _looks_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def find_tool(name: str) -> str | None:
    ensure_ffmpeg_on_path()
    found = shutil.which(name)
    if found:
        return found
    for directory in _candidate_dirs():
        candidate = Path(directory) / name
        if _looks_executable(candidate):
            return str(candidate)
    return None


def install_hint() -> str:
    if sys.platform == "darwin":
        return (
            "FFmpeg is required to write the depth MP4. On a Mac run: brew install ffmpeg "
            "then restart the app (python3 app.py --replace). "
            "If it is already installed, the app will look in /opt/homebrew/bin."
        )
    if sys.platform.startswith("linux"):
        return (
            "FFmpeg is required to write the depth MP4. Install it with: "
            "sudo apt install ffmpeg   then restart the app."
        )
    if sys.platform.startswith("win"):
        return (
            "FFmpeg is required to write the depth MP4. Install it (winget install Gyan.FFmpeg "
            "or from https://ffmpeg.org) and restart the app."
        )
    return "FFmpeg is required to write the depth MP4. Install ffmpeg and ffprobe, then restart the app."


def require_ffmpeg() -> str:
    path = find_tool("ffmpeg")
    if path:
        return path
    raise RuntimeError(install_hint())


def require_ffprobe() -> str | None:
    return find_tool("ffprobe")


@dataclass(frozen=True)
class FFmpegStatus:
    ok: bool
    path: str | None
    ffprobe: str | None
    hint: str


def ffmpeg_status() -> FFmpegStatus:
    path = find_tool("ffmpeg")
    probe = find_tool("ffprobe")
    return FFmpegStatus(ok=bool(path), path=path, ffprobe=probe, hint="" if path else install_hint())
