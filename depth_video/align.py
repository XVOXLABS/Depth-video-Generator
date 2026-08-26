from __future__ import annotations

import numpy as np

from depth_video.vendor import THIRD_PARTY  # noqa: F401 — puts third_party on sys.path
from utils.util import compute_scale_and_shift, get_interpolate_frames

# Official Video-Depth-Anything inference window. Do not change.
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8
FRAME_STEP = INFER_LEN - OVERLAP
ALIGN_LEN = OVERLAP - INTERP_LEN
KF_ALIGN_LIST = KEYFRAMES[:ALIGN_LEN]


class WindowAligner:
    """Streaming version of the official overlap alignment in video_depth.py."""

    def __init__(self, metric: bool = False) -> None:
        self.metric = metric
        self._first = True
        self._tail: list[np.ndarray] = []
        self._ref_align: list[np.ndarray] = []

    def add_window(self, depths: list[np.ndarray]) -> list[np.ndarray]:
        if len(depths) != INFER_LEN:
            raise ValueError(f"Expected {INFER_LEN} depth maps, got {len(depths)}")

        if self._first:
            self._first = False
            self._ref_align = [depths[k] for k in KF_ALIGN_LIST]
            emit = depths[: INFER_LEN - INTERP_LEN]
            self._tail = [d.copy() for d in depths[INFER_LEN - INTERP_LEN :]]
            return [d.copy() for d in emit]

        curr_align = [depths[i] for i in range(len(KF_ALIGN_LIST))]
        if self.metric:
            scale, shift = 1.0, 0.0
        else:
            scale, shift = compute_scale_and_shift(
                np.concatenate(curr_align),
                np.concatenate(self._ref_align),
                np.concatenate(np.ones_like(self._ref_align) == 1),
            )

        post = []
        for depth in depths[ALIGN_LEN:OVERLAP]:
            aligned = depth * scale + shift
            aligned[aligned < 0] = 0
            post.append(aligned)

        interpolated = get_interpolate_frames(self._tail, post)
        rest = []
        for i in range(OVERLAP, INFER_LEN):
            new_depth = depths[i] * scale + shift
            new_depth[new_depth < 0] = 0
            rest.append(new_depth)

        combined = list(interpolated) + rest
        emit = combined[:-INTERP_LEN]
        self._tail = combined[-INTERP_LEN:]

        self._ref_align = self._ref_align[:1]
        for kf_id in KF_ALIGN_LIST[1:]:
            new_depth = depths[kf_id] * scale + shift
            new_depth[new_depth < 0] = 0
            self._ref_align.append(new_depth)

        return emit

    def flush(self) -> list[np.ndarray]:
        tail = self._tail
        self._tail = []
        return tail
