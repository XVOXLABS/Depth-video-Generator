from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from depth_video.paths import JOBS_DIR, OUTPUTS_DIR, UPLOADS_DIR, ensure_runtime_dirs
from depth_video.pipeline import ConversionOptions, convert_video


@dataclass
class Job:
    id: str
    status: str = "queued"
    message: str = "Waiting…"
    progress: float = 0.0
    stage: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    input_path: Path | None = None
    output_path: Path | None = None
    original_name: str = "video.mp4"
    options: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    cancel: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "stage": self.stage,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "original_name": self.original_name,
            "options": self.options,
            "result": self.result,
            "has_output": bool(self.output_path and self.output_path.exists()),
        }


class JobManager:
    def __init__(self) -> None:
        ensure_runtime_dirs()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._wake = threading.Event()
        self._queue: list[str] = []
        self._worker.start()

    def create(self, upload_path: Path, original_name: str, options: ConversionOptions) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        stored = job_dir / Path(original_name).name
        shutil.move(str(upload_path), stored)
        stem = Path(original_name).stem
        output = OUTPUTS_DIR / f"{stem}_{job_id}_depth.mp4"
        job = Job(
            id=job_id,
            input_path=stored,
            output_path=output,
            original_name=original_name,
            options=options.__dict__.copy(),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
        self._wake.set()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def request_cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in {"queued", "running"}:
                job.cancel = True
                job.message = "Cancelling…"
                job.updated_at = time.time()
            return job

    def _run_loop(self) -> None:
        while True:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            job_id = None
            with self._lock:
                if self._queue:
                    job_id = self._queue.pop(0)
            if job_id:
                self._execute(job_id)

    def _execute(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None or job.input_path is None or job.output_path is None:
            return
        if job.cancel:
            job.status = "cancelled"
            job.message = "Cancelled"
            return

        job.status = "running"
        job.stage = "load"
        job.message = "Starting conversion…"
        job.updated_at = time.time()

        def on_progress(payload: dict) -> None:
            job.message = str(payload.get("message") or job.message)
            if payload.get("progress") is not None:
                job.progress = float(payload["progress"])
            job.stage = str(payload.get("stage") or job.stage)
            job.updated_at = time.time()
            if payload.get("result"):
                job.result = payload["result"]

        try:
            options = ConversionOptions(**job.options)
            result = convert_video(
                job.input_path,
                job.output_path,
                options=options,
                progress=on_progress,
                should_cancel=lambda: job.cancel,
            )
            job.status = "done"
            job.progress = 1.0
            job.stage = "done"
            job.result = {
                "output_path": str(result.output_path),
                "frames": result.frames,
                "fps": result.fps,
                "width": result.width,
                "height": result.height,
                "elapsed_s": result.elapsed_s,
                "device": result.device,
            }
            job.message = f"Done · {result.frames} frames in {result.elapsed_s:.1f}s"
            (JOBS_DIR / job.id / "result.json").write_text(json.dumps(job.snapshot(), indent=2))
        except Exception as exc:
            if job.cancel:
                job.status = "cancelled"
                job.message = "Cancelled"
            else:
                job.status = "error"
                job.error = str(exc)
                job.message = f"Failed: {exc}"
        finally:
            job.updated_at = time.time()


MANAGER = JobManager()
