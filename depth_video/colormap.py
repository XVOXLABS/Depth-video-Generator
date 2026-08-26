from __future__ import annotations

import cv2
import numpy as np

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "turbo": cv2.COLORMAP_TURBO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "gray": None,
}


class StreamingColorMapper:
    """Normalize depth with a slowly adapting range so long videos stay consistent."""

    def __init__(
        self,
        colormap: str = "inferno",
        invert: bool = False,
        ema: float = 0.08,
        lo_percentile: float = 1.0,
        hi_percentile: float = 99.0,
    ) -> None:
        if colormap not in COLORMAPS:
            raise ValueError(f"Unknown colormap '{colormap}'. Choose from {tuple(COLORMAPS)}")
        self.colormap = colormap
        self.invert = invert
        self.ema = ema
        self.lo_percentile = lo_percentile
        self.hi_percentile = hi_percentile
        self.lo: float | None = None
        self.hi: float | None = None

    def observe(self, depths: np.ndarray) -> None:
        sample = depths if depths.size < 250_000 else depths.reshape(-1)[:: max(1, depths.size // 200_000)]
        lo = float(np.percentile(sample, self.lo_percentile))
        hi = float(np.percentile(sample, self.hi_percentile))
        if hi <= lo:
            hi = lo + 1e-6
        if self.lo is None or self.hi is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo = (1.0 - self.ema) * self.lo + self.ema * lo
            self.hi = (1.0 - self.ema) * self.hi + self.ema * hi

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        lo = 0.0 if self.lo is None else self.lo
        hi = (lo + 1.0) if self.hi is None else max(self.hi, lo + 1e-6)
        norm = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        if self.invert:
            norm = 1.0 - norm
        depth_u8 = (norm * 255.0).astype(np.uint8)
        cmap = COLORMAPS[self.colormap]
        if cmap is None:
            return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)
        bgr = cv2.applyColorMap(depth_u8, cmap)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
