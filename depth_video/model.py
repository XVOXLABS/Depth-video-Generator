from __future__ import annotations

import contextlib
import os
from typing import Protocol

import numpy as np

from depth_video.align import INFER_LEN, KEYFRAMES
from depth_video.device import RuntimeDevice, detect_device
from depth_video.weights import MODEL_CONFIGS, ensure_checkpoint


class DepthBackend(Protocol):
    metric: bool
    encoder: str
    device_name: str

    def infer_window(self, frames: list[np.ndarray], reuse_input=None):
        """Return (depths[32,H,W], pre_input tensor or None)."""

    def infer_one(self, frame: np.ndarray) -> np.ndarray: ...

    def reset_stream(self) -> None: ...


def _build_transform(frame_h: int, frame_w: int, input_size: int):
    import cv2
    from torchvision.transforms import Compose

    from depth_video.vendor import THIRD_PARTY  # noqa: F401
    from video_depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

    ratio = max(frame_h, frame_w) / max(min(frame_h, frame_w), 1)
    size = input_size
    if ratio > 1.78:
        size = int(input_size * 1.777 / ratio)
        size = round(size / 14) * 14
    size = max(size, 14)
    transform = Compose(
        [
            Resize(
                width=size,
                height=size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    return transform, size


class VideoDepthAnythingBackend:
    """Official 32-frame windowed Video Depth Anything model."""

    def __init__(
        self,
        encoder: str = "vits",
        metric: bool = False,
        input_size: int = 518,
        device: RuntimeDevice | None = None,
        use_fp16: bool = True,
        progress=None,
    ) -> None:
        import torch

        from depth_video.vendor import THIRD_PARTY  # noqa: F401
        from video_depth_anything.video_depth import VideoDepthAnything

        self.encoder = encoder
        self.metric = metric
        self.input_size = input_size
        self.runtime = device or detect_device()
        self.device_name = self.runtime.name
        self.torch_device = torch.device(self.runtime.torch_device)
        self.use_fp16 = bool(use_fp16 and self.runtime.supports_fp16)
        self._torch = torch
        self._transform = None
        self._frame_hw: tuple[int, int] | None = None
        self._stream_model = None

        ckpt = ensure_checkpoint(encoder, metric, progress=progress)
        model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=metric)
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.torch_device).eval()

    def _autocast(self):
        torch = self._torch
        if self.use_fp16 and self.torch_device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _ensure_transform(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if self._transform is None or self._frame_hw != (h, w):
            self._transform, _ = _build_transform(h, w, self.input_size)
            self._frame_hw = (h, w)
        return self._transform

    def _frames_to_tensor(self, frames: list[np.ndarray]):
        torch = self._torch
        transform = self._ensure_transform(frames[0])
        chunks = []
        for frame in frames:
            image = transform({"image": frame.astype(np.float32) / 255.0})["image"]
            chunks.append(torch.from_numpy(image).unsqueeze(0).unsqueeze(0))
        return torch.cat(chunks, dim=1).to(self.torch_device)

    def infer_window(self, frames: list[np.ndarray], reuse_input=None):
        torch = self._torch
        if len(frames) != INFER_LEN:
            raise ValueError(f"Expected {INFER_LEN} frames")
        cur_input = self._frames_to_tensor(frames)
        if reuse_input is not None:
            cur_input = cur_input.clone()
            cur_input[:, :10, ...] = reuse_input[:, KEYFRAMES, ...]

        frame_h, frame_w = frames[0].shape[:2]
        with torch.no_grad():
            with self._autocast():
                depth = self.model.forward(cur_input)
            depth = depth.to(cur_input.dtype)
            depth = torch.nn.functional.interpolate(
                depth.flatten(0, 1).unsqueeze(1),
                size=(frame_h, frame_w),
                mode="bilinear",
                align_corners=True,
            )
        depths = [depth[i][0].detach().cpu().numpy() for i in range(depth.shape[0])]
        return depths, cur_input

    def reset_stream(self) -> None:
        self._stream_model = None

    def infer_one(self, frame: np.ndarray) -> np.ndarray:
        """Experimental training-free streaming inference (one frame at a time)."""
        from depth_video.vendor import THIRD_PARTY  # noqa: F401
        from video_depth_anything.video_depth_stream import VideoDepthAnything as StreamModel

        if self._stream_model is None:
            stream = StreamModel(**MODEL_CONFIGS[self.encoder])
            stream.load_state_dict(self.model.state_dict(), strict=False)
            self._stream_model = stream.to(self.torch_device).eval()
        return self._stream_model.infer_video_depth_one(
            frame,
            input_size=self.input_size,
            device=str(self.torch_device),
            fp32=not self.use_fp16,
        )


class FakeDepthBackend:
    """Deterministic luminance-based stand-in used for tests and UI smoke runs."""

    def __init__(self, encoder: str = "vits", metric: bool = False, **_kwargs) -> None:
        self.encoder = encoder
        self.metric = metric
        self.device_name = "fake"
        self._prev: np.ndarray | None = None

    def infer_window(self, frames: list[np.ndarray], reuse_input=None):
        depths = [self._fake_depth(frame) for frame in frames]
        return depths, None

    def infer_one(self, frame: np.ndarray) -> np.ndarray:
        return self._fake_depth(frame)

    def reset_stream(self) -> None:
        self._prev = None

    def _fake_depth(self, frame: np.ndarray) -> np.ndarray:
        rgb = frame.astype(np.float32) / 255.0
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        depth = 1.0 - luma
        if self._prev is not None and self._prev.shape == depth.shape:
            depth = 0.65 * depth + 0.35 * self._prev
        self._prev = depth
        return depth.astype(np.float32)


def create_backend(
    encoder: str = "vits",
    metric: bool = False,
    input_size: int = 518,
    use_fp16: bool = True,
    progress=None,
) -> DepthBackend:
    backend = os.environ.get("DEPTH_VIDEO_BACKEND", "").strip().lower()
    if backend in {"fake", "dummy", "test"}:
        if progress:
            progress("Using fake depth backend (DEPTH_VIDEO_BACKEND)", None)
        return FakeDepthBackend(encoder=encoder, metric=metric)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Video Depth Anything. "
            "Install it with `pip install torch torchvision` "
            "(use the CUDA wheel from pytorch.org if you have an NVIDIA GPU)."
        ) from exc
    return VideoDepthAnythingBackend(
        encoder=encoder,
        metric=metric,
        input_size=input_size,
        use_fp16=use_fp16,
        progress=progress,
    )
