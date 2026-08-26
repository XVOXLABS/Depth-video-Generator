from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from depth_video.paths import JOBS_DIR, OUTPUTS_DIR, ensure_runtime_dirs
from depth_video.pipeline import ConversionOptions


@dataclass
class Job:
    id: str
    status: str = "queued"
    message: str = "Waiting…"
    progress: float = 0.0
    stage: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    input_path: Path | None = None
    output_path: Path | None = None
    original_name: str = "video.mp4"
    options: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    cancel: bool = False
    process: Any = field(default=None, repr=False)

    def elapsed_s(self) -> float:
        start = self.started_at or self.created_at
        end = time.time() if self.status in {"queued", "running"} else self.updated_at
        return max(0.0, end - start)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "stage": self.stage,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "elapsed_s": self.elapsed_s(),
            "original_name": self.original_name,
            "options": self.options,
            "result": self.result,
            "has_output": bool(self.output_path and self.output_path.exists()),
        }


def _job_worker(
    input_path: str,
    output_path: str,
    options_dict: dict[str, Any],
    progress_path: str,
    cancel_path: str,
    result_path: str,
) -> None:
    """Child process: keeps torch CPU work off the web-server GIL."""
    os.environ.setdefault("DEPTH_VIDEO_JOB_WORKER", "1")
    from depth_video.pipeline import CancelledError, ConversionOptions, convert_video

    progress_file = Path(progress_path)
    cancel_file = Path(cancel_path)
    result_file = Path(result_path)

    def write_state(**payload: Any) -> None:
        payload.setdefault("updated_at", time.time())
        tmp = progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(progress_file)

    def on_progress(payload: dict) -> None:
        body = {k: v for k, v in payload.items() if k != "result"}
        write_state(status="running", result=payload.get("result"), **body)

    try:
        write_state(status="running", stage="load", message="Starting conversion…", progress=0.02)
        result = convert_video(
            input_path,
            output_path,
            options=ConversionOptions(**options_dict),
            progress=on_progress,
            should_cancel=lambda: cancel_file.exists(),
        )
        snapshot = {
            "status": "done",
            "stage": "done",
            "progress": 1.0,
            "message": f"Done · {result.frames} frames in {result.elapsed_s:.1f}s",
            "result": {
                "output_path": str(result.output_path),
                "frames": result.frames,
                "fps": result.fps,
                "width": result.width,
                "height": result.height,
                "elapsed_s": result.elapsed_s,
                "device": result.device,
            },
        }
        result_file.write_text(json.dumps(snapshot, indent=2))
        write_state(**snapshot)
    except CancelledError:
        write_state(status="cancelled", stage="cancelled", message="Cancelled", progress=0.0)
    except Exception as exc:
        if cancel_file.exists():
            write_state(status="cancelled", stage="cancelled", message="Cancelled", progress=0.0)
        else:
            write_state(status="error", stage="error", message=f"Failed: {exc}", error=str(exc), progress=0.0)


class JobManager:
    def __init__(self) -> None:
        ensure_runtime_dirs()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._wake = threading.Event()
        self._queue: list[str] = []
        self._ctx = mp.get_context("spawn")
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
            cached = self._jobs.get(job_id)
        if cached:
            return cached
        restored = self._restore(job_id)
        if restored is None:
            return None
        with self._lock:
            return self._jobs.setdefault(job_id, restored)

    def _restore(self, job_id: str) -> Job | None:
        job_dir = JOBS_DIR / job_id
        result_path = job_dir / "result.json"
        progress_path = job_dir / "progress.json"
        source = result_path if result_path.exists() else progress_path
        if not source.exists():
            return None
        try:
            payload = json.loads(source.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        output = None
        result = payload.get("result")
        if isinstance(result, dict) and result.get("output_path"):
            output = Path(result["output_path"])
        if output is None or not output.exists():
            matches = sorted(OUTPUTS_DIR.glob(f"*_{job_id}_depth.mp4"))
            if matches:
                output = matches[0]
        uploads = [p for p in job_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}]
        original = uploads[0].name if uploads else f"{job_id}.mp4"
        job = Job(
            id=job_id,
            status=str(payload.get("status") or "done"),
            message=str(payload.get("message") or "Done"),
            progress=float(payload.get("progress") or (1.0 if payload.get("status") == "done" else 0.0)),
            stage=str(payload.get("stage") or payload.get("status") or "done"),
            error=payload.get("error"),
            input_path=uploads[0] if uploads else None,
            output_path=output,
            original_name=original,
            result=result if isinstance(result, dict) else None,
        )
        return job

    def request_cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in {"queued", "running"}:
                job.cancel = True
                job.message = "Cancelling…"
                job.updated_at = time.time()
                cancel_path = JOBS_DIR / job.id / "cancel"
                cancel_path.parent.mkdir(parents=True, exist_ok=True)
                cancel_path.touch()
                if job.process is not None and job.process.is_alive():
                    job.process.terminate()
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

    def _apply_progress_file(self, job: Job, progress_path: Path) -> None:
        if not progress_path.exists():
            return
        try:
            payload = json.loads(progress_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        job.message = str(payload.get("message") or job.message)
        if payload.get("progress") is not None:
            job.progress = float(payload["progress"])
        job.stage = str(payload.get("stage") or job.stage)
        if payload.get("status") in {"running", "done", "error", "cancelled"}:
            if job.status == "running" or payload["status"] != "running":
                job.status = payload["status"]
        job.error = payload.get("error") or job.error
        if payload.get("result"):
            job.result = payload["result"]
        job.updated_at = time.time()

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
        job.started_at = time.time()
        job.updated_at = time.time()

        job_dir = JOBS_DIR / job.id
        progress_path = job_dir / "progress.json"
        cancel_path = job_dir / "cancel"
        result_path = job_dir / "result.json"

        proc = self._ctx.Process(
            target=_job_worker,
            args=(
                str(job.input_path),
                str(job.output_path),
                job.options,
                str(progress_path),
                str(cancel_path),
                str(result_path),
            ),
        )
        job.process = proc
        proc.start()
        while proc.is_alive():
            if job.cancel and proc.is_alive():
                proc.terminate()
                break
            self._apply_progress_file(job, progress_path)
            time.sleep(0.25)
        proc.join(timeout=15)
        self._apply_progress_file(job, progress_path)
        if job.cancel and job.status == "running":
            job.status = "cancelled"
            job.message = "Cancelled"
        elif proc.exitcode not in (0, None) and job.status == "running":
            job.status = "error"
            job.error = f"Converter exited with code {proc.exitcode}"
            job.message = f"Failed: {job.error}"
        elif result_path.exists() and job.status == "running":
            self._apply_progress_file(job, result_path)
        job.updated_at = time.time()
        job.process = None


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
