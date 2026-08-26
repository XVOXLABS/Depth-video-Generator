from __future__ import annotations

import os
from pathlib import Path

import pytest

from depth_video.ffmpeg_bin import find_tool, install_hint, require_ffmpeg


def test_finds_system_ffmpeg():
    path = find_tool("ffmpeg")
    assert path is not None
    assert Path(path).name.startswith("ffmpeg")


def test_require_ffmpeg_returns_path():
    path = require_ffmpeg()
    assert os.path.isfile(path)


def test_missing_ffmpeg_message(monkeypatch):
    monkeypatch.setattr("depth_video.ffmpeg_bin.shutil.which", lambda name: None)
    monkeypatch.setattr("depth_video.ffmpeg_bin._candidate_dirs", lambda: [])
    with pytest.raises(RuntimeError, match="FFmpeg is required"):
        require_ffmpeg()
    hint = install_hint()
    assert "ffmpeg" in hint.lower()
