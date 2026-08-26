from __future__ import annotations

from pathlib import Path

from depth_video.cli import main
from depth_video.video_io import probe_video, write_demo_clip


def test_write_demo_clip(tmp_path: Path):
    clip = write_demo_clip(tmp_path / "demo.mp4", seconds=0.5, fps=8, size="160x120")
    info = probe_video(clip)
    assert info.width == 160
    assert info.height == 120
    assert info.has_audio


def test_convert_missing_input_hints_demo(capsys):
    code = main(["convert", "input.mp4"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Input not found: input.mp4" in captured.err
    assert "python3 -m depth_video demo" in captured.err


def test_demo_command_fake(tmp_path: Path):
    code = main(["demo", "--fake", "--output-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "demo.mp4").exists()
    depth = tmp_path / "demo_depth.mp4"
    assert depth.exists() and depth.stat().st_size > 0
