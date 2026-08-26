from __future__ import annotations

import uuid
from pathlib import Path

import asyncio
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from depth_video.device import detect_device
from depth_video.jobs import get_manager
from depth_video.paths import STATIC_DIR, UPLOADS_DIR, ensure_runtime_dirs
from depth_video.pipeline import ConversionOptions
from depth_video.weights import ENCODERS

ensure_runtime_dirs()


def as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


app = FastAPI(title="Depth Video Generator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ready_output(job_id: str):
    job = get_manager().get(job_id)
    if job is None or not job.output_path or not job.output_path.exists():
        raise HTTPException(404, "Output not ready")
    return job


@app.api_route("/api/health", methods=["GET", "HEAD"])
def health():
    device = detect_device()
    torch_ok = True
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_ok = False
    return {
        "ok": True,
        "torch": torch_ok,
        "device": device.kind,
        "device_name": device.name,
        "fp16": device.supports_fp16,
        "encoders": list(ENCODERS),
        "cpu_inference": device.kind == "cpu",
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    encoder: str = Form("vits"),
    metric: str = Form("false"),
    input_size: int = Form(518),
    max_res: int = Form(1280),
    target_fps: float = Form(-1),
    max_len: int = Form(-1),
    colormap: str = Form("inferno"),
    invert: str = Form("false"),
    grayscale: str = Form("false"),
    keep_audio: str = Form("true"),
    layout: str = Form("depth"),
    mode: str = Form("windowed"),
    use_fp16: str = Form("true"),
):
    if encoder not in ENCODERS:
        raise HTTPException(400, f"encoder must be one of {ENCODERS}")
    if layout not in {"depth", "side_by_side"}:
        raise HTTPException(400, "layout must be depth or side_by_side")
    if mode not in {"windowed", "streaming"}:
        raise HTTPException(400, "mode must be windowed or streaming")

    original = file.filename or "upload.mp4"
    suffix = Path(original).suffix or ".mp4"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    with temp_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty")

    options = ConversionOptions(
        encoder=encoder,
        metric=as_bool(metric),
        input_size=input_size,
        max_res=max_res,
        target_fps=target_fps,
        max_len=max_len,
        colormap=colormap,
        invert=as_bool(invert),
        grayscale=as_bool(grayscale),
        keep_audio=as_bool(keep_audio),
        layout=layout,
        mode=mode,
        use_fp16=as_bool(use_fp16),
    )
    job = get_manager().create(temp_path, original, options)
    return JSONResponse(job.snapshot())


@app.api_route("/api/jobs/{job_id}", methods=["GET", "HEAD"])
def get_job(job_id: str):
    job = get_manager().get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job.snapshot()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = get_manager().request_cancel(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job.snapshot()


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    if get_manager().get(job_id) is None:
        raise HTTPException(404, "Unknown job")

    async def generate():
        last = None
        while True:
            job = get_manager().get(job_id)
            if job is None:
                break
            snap = job.snapshot()
            payload = json.dumps(snap)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if snap["status"] in {"done", "error", "cancelled"}:
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.api_route("/api/jobs/{job_id}/video", methods=["GET", "HEAD"])
def job_video(job_id: str):
    job = _ready_output(job_id)
    # Inline preview. Setting filename= makes Starlette send Content-Disposition:
    # attachment, which leaves <video> blank even though conversion succeeded.
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        content_disposition_type="inline",
        headers={
            "Content-Disposition": "inline",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )


@app.api_route("/api/jobs/{job_id}/download", methods=["GET", "HEAD"])
def job_download(job_id: str):
    job = _ready_output(job_id)
    name = job.output_path.name.replace('"', "")
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=name,
        content_disposition_type="attachment",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def create_app() -> FastAPI:
    return app
