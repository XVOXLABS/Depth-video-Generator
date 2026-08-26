from __future__ import annotations

import numpy as np

from depth_video.align import FRAME_STEP, INFER_LEN, INTERP_LEN, WindowAligner


def _window(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random((12, 16)).astype(np.float32) + 0.2 for _ in range(INFER_LEN)]


def test_first_window_holds_interpolation_tail():
    aligner = WindowAligner(metric=False)
    emitted = aligner.add_window(_window(0))
    assert len(emitted) == INFER_LEN - INTERP_LEN
    assert len(aligner.flush()) == INTERP_LEN


def test_second_window_emits_step_frames():
    aligner = WindowAligner(metric=False)
    first = aligner.add_window(_window(1))
    second = aligner.add_window(_window(2))
    assert len(first) == INFER_LEN - INTERP_LEN
    assert len(second) == FRAME_STEP
    flushed = aligner.flush()
    assert len(flushed) == INTERP_LEN
    total = len(first) + len(second) + len(flushed)
    assert total == INFER_LEN + FRAME_STEP


def test_metric_skips_scale_shift():
    aligner = WindowAligner(metric=True)
    depths = _window(3)
    aligner.add_window(depths)
    emitted = aligner.add_window(depths)
    assert all(np.isfinite(frame).all() for frame in emitted)
