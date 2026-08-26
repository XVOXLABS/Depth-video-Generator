from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from depth_video.align import FRAME_STEP, INFER_LEN, WindowAligner
from depth_video.colormap import StreamingColorMapper
from depth_video.model import DepthBackend, create_backend
from depth_video.video_io import FrameReader, FFmpegWriter, mux_audio, probe_video

ProgressCallback = Callable[[dict], None]


@dataclass
class ConversionOptions:
    encoder: str = "vits"
    metric: bool = False
    input_size: int = 518
    max_res: int = 1280
    target_fps: float = -1
    max_len: int = -1
    colormap: str = "inferno"
    invert: bool = False
    grayscale: bool = False
    keep_audio: bool = True
    layout: str = "depth"  # depth | side_by_side
    mode: str = "windowed"  # windowed | streaming
    use_fp16: bool = True


@dataclass
class ConversionResult:
    output_path: Path
    frames: int
    fps: float
    width: int
    height: int
    duration_s: float
    device: str
    elapsed_s: float
    options: ConversionOptions


class CancelledError(RuntimeError):
    pass


def _emit(callback: ProgressCallback | None, **payload) -> None:
    if callback:
        callback(payload)


def _compose_frame(src: np.ndarray, depth_rgb: np.ndarray, layout: str) -> np.ndarray:
    if layout == "side_by_side":
        if src.shape[0] != depth_rgb.shape[0] or src.shape[1] != depth_rgb.shape[1]:
            src = cv2.resize(src, (depth_rgb.shape[1], depth_rgb.shape[0]))
        return np.concatenate([src, depth_rgb], axis=1)
    return depth_rgb


def convert_video(
    input_path: str | Path,
    output_path: str | Path,
    options: ConversionOptions | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
    backend: DepthBackend | None = None,
) -> ConversionResult:
    options = options or ConversionOptions()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if options.grayscale:
        options.colormap = "gray"

    info = probe_video(input_path)
    _emit(progress, stage="probe", message=f"Opened {input_path.name}", progress=0.01)

    def report(message: str, fraction: float | None = None) -> None:
        _emit(progress, stage="weights", message=message, progress=fraction)

    if backend is None:
        _emit(progress, stage="load", message="Loading Video Depth Anything…", progress=0.04)
        backend = create_backend(
            encoder=options.encoder,
            metric=options.metric,
            input_size=options.input_size,
            use_fp16=options.use_fp16,
            progress=report,
        )

    reader = FrameReader(
        input_path,
        max_res=options.max_res,
        target_fps=options.target_fps,
        max_len=options.max_len,
    )
    colormap = StreamingColorMapper(colormap=options.colormap, invert=options.invert)

    out_w = reader.width * (2 if options.layout == "side_by_side" else 1)
    tmp_video = output_path.with_suffix(".silent.mp4")
    writer = FFmpegWriter(tmp_video, out_w, reader.height, reader.output_fps)

    started = time.time()
    total_hint = reader.estimated_frames

    try:
        if options.mode == "streaming":
            emitted = _run_streaming(
                backend, reader, writer, colormap, options, progress, should_cancel, total_hint
            )
        else:
            emitted = _run_windowed(
                backend, reader, writer, colormap, options, progress, should_cancel, total_hint
            )
    finally:
        reader.close()
        writer.close()

    if options.keep_audio and info.has_audio:
        _emit(progress, stage="audio", message="Muxing original audio…", progress=0.97)
        muxed = output_path.with_suffix(".audio.mp4")
        if mux_audio(tmp_video, input_path, muxed):
            tmp_video.unlink(missing_ok=True)
            muxed.replace(output_path)
        else:
            tmp_video.replace(output_path)
    else:
        tmp_video.replace(output_path)

    elapsed = time.time() - started
    result = ConversionResult(
        output_path=output_path,
        frames=emitted,
        fps=reader.output_fps,
        width=out_w,
        height=reader.height,
        duration_s=emitted / reader.output_fps if reader.output_fps else 0,
        device=getattr(backend, "device_name", "unknown"),
        elapsed_s=elapsed,
        options=options,
    )
    _emit(
        progress,
        stage="done",
        message=f"Wrote {emitted} frames in {elapsed:.1f}s",
        progress=1.0,
        result={
            "output_path": str(output_path),
            "frames": emitted,
            "fps": reader.output_fps,
            "width": out_w,
            "height": reader.height,
            "elapsed_s": elapsed,
            "device": result.device,
        },
    )
    return result


def _maybe_cancel(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel and should_cancel():
        raise CancelledError("Conversion cancelled")


def _progress_fraction(emitted: int, total_hint: int) -> float:
    if total_hint <= 0:
        return min(0.95, 0.08 + emitted / 400.0)
    return min(0.95, 0.08 + 0.87 * (emitted / total_hint))


class _Emitter:
    def __init__(
        self,
        writer: FFmpegWriter,
        colormap: StreamingColorMapper,
        layout: str,
        progress: ProgressCallback | None,
        total_hint: int,
        source_limit: int | None = None,
    ) -> None:
        self.writer = writer
        self.colormap = colormap
        self.layout = layout
        self.progress = progress
        self.total_hint = total_hint
        self.source_limit = source_limit
        self.emitted = 0
        self.pending_sources: deque[np.ndarray] = deque()

    def push_source(self, frame: np.ndarray) -> None:
        self.pending_sources.append(frame)

    def write_depths(self, depths: list[np.ndarray], fallback: np.ndarray) -> None:
        if not depths:
            return
        self.colormap.observe(np.stack(depths, axis=0))
        for depth in depths:
            if self.source_limit is not None and self.emitted >= self.source_limit:
                return
            src = self.pending_sources.popleft() if self.pending_sources else fallback
            vis = self.colormap.colorize(depth)
            self.writer.write(_compose_frame(src, vis, self.layout))
            self.emitted += 1
            if self.emitted % 8 == 0:
                self._report()

    def _report(self) -> None:
        limit = self.source_limit or self.total_hint
        _emit(
            self.progress,
            stage="infer",
            message=f"Rendered frame {self.emitted}" + (f" / {limit}" if limit else ""),
            progress=_progress_fraction(self.emitted, limit),
            frames_done=self.emitted,
            frames_total=limit,
        )


def _run_windowed(
    backend: DepthBackend,
    reader: FrameReader,
    writer: FFmpegWriter,
    colormap: StreamingColorMapper,
    options: ConversionOptions,
    progress: ProgressCallback | None,
    should_cancel: Callable[[], bool] | None,
    total_hint: int,
) -> int:
    """Official overlapping 32-frame windows, streamed so RAM stays bounded."""
    aligner = WindowAligner(metric=backend.metric)
    pending: list[np.ndarray] = []
    reuse_input = None
    source_len = 0
    last_frame: np.ndarray | None = None
    window_index = 0
    emitter = _Emitter(writer, colormap, options.layout, progress, total_hint)

    def run_window() -> None:
        nonlocal reuse_input, window_index
        sources = pending[:INFER_LEN]
        depths, reuse_input = backend.infer_window(sources, reuse_input=reuse_input)
        aligned = aligner.add_window(depths)
        emitter.write_depths(aligned, last_frame)
        pending[:] = pending[FRAME_STEP:]
        window_index += 1
        _emit(
            progress,
            stage="infer",
            message=f"Window {window_index} · {emitter.emitted} frames written",
            progress=_progress_fraction(emitter.emitted, emitter.source_limit or total_hint),
            frames_done=emitter.emitted,
            frames_total=emitter.source_limit or total_hint,
        )

    while True:
        _maybe_cancel(should_cancel)
        frame = reader.read()
        if frame is None:
            break
        pending.append(frame)
        emitter.push_source(frame)
        last_frame = frame
        source_len += 1
        if len(pending) >= INFER_LEN:
            run_window()

    if source_len == 0 or last_frame is None:
        raise ValueError("The input file has no readable video frames")

    emitter.source_limit = source_len

    while emitter.emitted < source_len:
        _maybe_cancel(should_cancel)
        while len(pending) < INFER_LEN:
            pending.append(last_frame)
        run_window()

    remaining = aligner.flush()
    if remaining and emitter.emitted < source_len:
        emitter.write_depths(remaining, last_frame)

    return emitter.emitted


def _run_streaming(
    backend: DepthBackend,
    reader: FrameReader,
    writer: FFmpegWriter,
    colormap: StreamingColorMapper,
    options: ConversionOptions,
    progress: ProgressCallback | None,
    should_cancel: Callable[[], bool] | None,
    total_hint: int,
) -> int:
    """Experimental one-frame streaming path for very tight VRAM budgets."""
    backend.reset_stream()
    emitted = 0
    warmup: list[np.ndarray] = []
    while True:
        _maybe_cancel(should_cancel)
        frame = reader.read()
        if frame is None:
            break
        depth = backend.infer_one(frame)
        warmup.append(depth)
        if len(warmup) >= 8:
            colormap.observe(np.stack(warmup, axis=0))
            warmup = warmup[-2:]
        elif colormap.lo is None:
            colormap.observe(depth)
        vis = colormap.colorize(depth)
        writer.write(_compose_frame(frame, vis, options.layout))
        emitted += 1
        if emitted % 5 == 0:
            _emit(
                progress,
                stage="infer",
                message=f"Streamed frame {emitted}" + (f" / {total_hint}" if total_hint else ""),
                progress=_progress_fraction(emitted, total_hint),
                frames_done=emitted,
                frames_total=total_hint,
            )
    if emitted == 0:
        raise ValueError("The input file has no readable video frames")
    return emitted
