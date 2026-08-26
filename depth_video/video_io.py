from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def write_demo_clip(
    path: str | Path,
    seconds: float = 2.0,
    fps: int = 12,
    size: str = "640x360",
) -> Path:
    """Write a short synthetic MP4 so a first run does not need an input file."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to create a demo clip")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}:size={size}:rate={fps}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(path),
        ]
    )
    return path


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    has_audio: bool
    source_fps: float


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not shutil.which("ffprobe"):
        return _probe_with_opencv(path)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = _run(cmd)
    data = json.loads(result.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))

    fps = _parse_rate(video.get("r_frame_rate") or video.get("avg_frame_rate") or "30/1")
    width = int(video["width"])
    height = int(video["height"])
    nb_frames = video.get("nb_frames")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    if nb_frames and nb_frames != "N/A":
        frame_count = int(nb_frames)
    elif duration > 0:
        frame_count = max(1, int(round(duration * fps)))
    else:
        frame_count = 0
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        has_audio=audio,
        source_fps=fps,
    )


def _probe_with_opencv(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(path, width, height, fps, frame_count, duration, False, fps)


def _parse_rate(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else float(num)
    return float(rate)


def even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


class FrameReader:
    """Decode frames sequentially, optionally scaling and dropping to a target fps."""

    def __init__(
        self,
        path: str | Path,
        max_res: int = 1280,
        target_fps: float = -1,
        max_len: int = -1,
    ) -> None:
        self.info = probe_video(path)
        self.max_res = max_res
        self.max_len = max_len
        self.source_fps = self.info.fps if self.info.fps > 1e-3 else 30.0
        self.output_fps = self.source_fps if target_fps is None or target_fps <= 0 else float(target_fps)
        self.stride = max(int(round(self.source_fps / self.output_fps)), 1)
        self.output_fps = self.source_fps / self.stride

        src_w, src_h = self.info.width, self.info.height
        if max_res > 0 and max(src_w, src_h) > max_res:
            scale = max_res / max(src_w, src_h)
            self.width = even(round(src_w * scale))
            self.height = even(round(src_h * scale))
        else:
            self.width = even(src_w)
            self.height = even(src_h)

        estimated = self.info.frame_count // self.stride if self.info.frame_count else 0
        if max_len > 0:
            estimated = min(estimated, max_len) if estimated else max_len
        self.estimated_frames = estimated

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self._index = 0
        self._emitted = 0
        self.exhausted = False
        self.actual_frames = 0
        self.last_frame: np.ndarray | None = None

    def __iter__(self) -> "FrameReader":
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def read(self) -> np.ndarray | None:
        if self.exhausted:
            return None
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                self.exhausted = True
                self.actual_frames = self._emitted
                return None
            keep = self._index % self.stride == 0
            self._index += 1
            if not keep:
                continue
            if self.max_len > 0 and self._emitted >= self.max_len:
                self.exhausted = True
                self.actual_frames = self._emitted
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
                rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self._emitted += 1
            self.last_frame = rgb
            return rgb

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class FFmpegWriter:
    """Stream RGB frames into an H.264 MP4 without holding the video in RAM."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 18) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required to write output videos")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = even(width)
        self.height = even(height)
        self.fps = max(fps, 1.0)
        self.frames_written = 0
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(frame)
        self._proc.stdin.write(rgb.tobytes())
        self.frames_written += 1

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            code = self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)
            raise RuntimeError(f"ffmpeg timed out while writing {self.path}")
        self._proc = None
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with status {code} while writing {self.path}")

    def __enter__(self) -> "FFmpegWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def mux_audio(video_path: str | Path, source_path: str | Path, output_path: str | Path) -> bool:
    """Copy the source audio track onto a silent depth video. Returns True if muxed."""
    if not shutil.which("ffmpeg"):
        return False
    video_path = Path(video_path)
    output_path = Path(output_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
