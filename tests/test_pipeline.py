from __future__ import annotations

from pathlib import Path

from depth_video.model import FakeDepthBackend
from depth_video.pipeline import ConversionOptions, convert_video
from depth_video.video_io import probe_video
from tests.helpers import make_video


def _convert(tmp_path: Path, frames_seconds: float, fps: int, **opts) -> tuple[Path, int]:
    src = make_video(tmp_path / "src.mp4", seconds=frames_seconds, fps=fps, size="160x120")
    out = tmp_path / "depth.mp4"
    options = ConversionOptions(max_res=160, target_fps=fps, keep_audio=False, **opts)
    result = convert_video(src, out, options=options, backend=FakeDepthBackend())
    return result.output_path, result.frames


def test_short_video_windowed(tmp_path: Path):
    out, frames = _convert(tmp_path, 1.0, 8)
    assert out.exists() and out.stat().st_size > 0
    assert frames == 8
    info = probe_video(out)
    assert info.width == 160
    assert info.height == 120


def test_longer_than_window_video(tmp_path: Path):
    # 5 seconds at 10 fps = 50 frames, longer than the 32-frame model window.
    out, frames = _convert(tmp_path, 5.0, 10)
    assert frames == 50
    assert probe_video(out).frame_count >= 45


def test_streaming_mode(tmp_path: Path):
    out, frames = _convert(tmp_path, 1.2, 10, mode="streaming")
    assert frames == 12
    assert out.exists()


def test_side_by_side_doubles_width(tmp_path: Path):
    src = make_video(tmp_path / "src.mp4", seconds=0.5, fps=8, size="160x120")
    out = tmp_path / "sbs.mp4"
    result = convert_video(
        src,
        out,
        options=ConversionOptions(max_res=160, keep_audio=False, layout="side_by_side"),
        backend=FakeDepthBackend(),
    )
    assert result.width == 320
    assert probe_video(out).width == 320


def test_max_len_truncates(tmp_path: Path):
    src = make_video(tmp_path / "src.mp4", seconds=3.0, fps=10, size="160x120")
    out = tmp_path / "trim.mp4"
    result = convert_video(
        src,
        out,
        options=ConversionOptions(max_res=160, keep_audio=False, max_len=7),
        backend=FakeDepthBackend(),
    )
    assert result.frames == 7
