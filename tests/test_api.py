from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from depth_video.server import app
from tests.helpers import make_video

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "vits" in body["encoders"]
    assert "torch" in body
    assert "ffmpeg" in body
    assert "device" in body
    assert "cpu_inference" in body


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Depth Video Generator" in res.text
    assert "ffmpegBanner" in res.text


def test_job_converts_upload(tmp_path: Path):
    video = make_video(tmp_path / "clip.mp4", seconds=0.8, fps=8, size="160x120")
    with video.open("rb") as handle:
        res = client.post(
            "/api/jobs",
            files={"file": ("clip.mp4", handle, "video/mp4")},
            data={
                "encoder": "vits",
                "max_res": "160",
                "keep_audio": "false",
                "colormap": "inferno",
            },
        )
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]

    snapshot = None
    for _ in range(200):
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] in {"done", "error"}:
            break
        time.sleep(0.1)
    assert snapshot is not None
    assert snapshot["status"] == "done", snapshot
    video_res = client.get(f"/api/jobs/{job_id}/video")
    assert video_res.status_code == 200
    assert video_res.headers["content-type"].startswith("video/")
    assert len(video_res.content) > 1000
    disposition = video_res.headers.get("content-disposition", "").lower()
    assert "attachment" not in disposition
    assert disposition.startswith("inline") or disposition == ""

    head_res = client.head(f"/api/jobs/{job_id}/video")
    assert head_res.status_code == 200
    assert "attachment" not in head_res.headers.get("content-disposition", "").lower()

    download_res = client.get(f"/api/jobs/{job_id}/download")
    assert download_res.status_code == 200
    assert "attachment" in download_res.headers.get("content-disposition", "").lower()


def test_finished_job_restores_after_manager_reset(tmp_path: Path):
    video = make_video(tmp_path / "restore.mp4", seconds=0.5, fps=8, size="160x120")
    with video.open("rb") as handle:
        res = client.post(
            "/api/jobs",
            files={"file": ("restore.mp4", handle, "video/mp4")},
            data={"encoder": "vits", "max_res": "160", "keep_audio": "false"},
        )
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]
    snapshot = None
    for _ in range(200):
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] in {"done", "error"}:
            break
        time.sleep(0.1)
    assert snapshot is not None
    assert snapshot["status"] == "done", snapshot

    from depth_video import jobs as jobs_mod

    jobs_mod._manager = None
    restored = client.get(f"/api/jobs/{job_id}")
    assert restored.status_code == 200
    body = restored.json()
    assert body["status"] == "done"
    assert body["has_output"] is True
    video_res = client.get(f"/api/jobs/{job_id}/video")
    assert video_res.status_code == 200
    assert "attachment" not in video_res.headers.get("content-disposition", "").lower()
