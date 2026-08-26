from __future__ import annotations

from pathlib import Path

import numpy as np

from depth_video.colormap import StreamingColorMapper
from depth_video.video_io import FrameReader, FFmpegWriter, mux_audio, probe_video
from tests.helpers import make_video


def test_probe_and_read(tmp_path: Path):
    video = make_video(tmp_path / "in.mp4", seconds=1.0, fps=10)
    info = probe_video(video)
    assert info.width == 320
    assert info.height == 240
    reader = FrameReader(video, max_res=160, target_fps=5)
    frames = list(reader)
    reader.close()
    assert frames, "expected frames"
    assert frames[0].shape[2] == 3
    assert frames[0].shape[1] == 160
    assert reader.output_fps == 5


def test_writer_roundtrip(tmp_path: Path):
    out = tmp_path / "out.mp4"
    with FFmpegWriter(out, 64, 48, 8) as writer:
        for i in range(12):
            frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            writer.write(frame)
    info = probe_video(out)
    assert info.width == 64
    assert info.height == 48
    assert info.frame_count >= 10


def test_mux_audio(tmp_path: Path):
    src = make_video(tmp_path / "src.mp4", seconds=0.6, fps=8, audio=True)
    silent = tmp_path / "silent.mp4"
    with FFmpegWriter(silent, 320, 240, 8) as writer:
        for _ in range(5):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    muxed = tmp_path / "muxed.mp4"
    assert mux_audio(silent, src, muxed)
    assert probe_video(muxed).has_audio


def test_colormap_adapts():
    mapper = StreamingColorMapper(colormap="inferno")
    near = np.ones((8, 8), dtype=np.float32) * 0.2
    far = np.ones((8, 8), dtype=np.float32) * 4.0
    mapper.observe(near)
    a = mapper.colorize(near)
    mapper.observe(far)
    b = mapper.colorize(far)
    assert a.shape == (8, 8, 3)
    assert b.dtype == np.uint8
    assert a.mean() != b.mean()
