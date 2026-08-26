from __future__ import annotations

import os
import socket
from pathlib import Path

from depth_video.cli import can_bind, choose_port, main, pids_listening
from depth_video.video_io import probe_video, write_demo_clip


def test_write_demo_clip(tmp_path: Path):
    clip = write_demo_clip(tmp_path / "demo.mp4", seconds=0.5, fps=8, size="160x120")
    info = probe_video(clip)
    assert info.width == 160
    assert info.height == 120
    assert info.has_audio


def test_convert_missing_input_hints_demo(capsys):
    code = main(["convert", "input.mp4"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Input not found: input.mp4" in captured.err
    assert "python3 -m depth_video demo" in captured.err


def test_demo_command_fake(tmp_path: Path):
    code = main(["demo", "--fake", "--output-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "demo.mp4").exists()
    depth = tmp_path / "demo_depth.mp4"
    assert depth.exists() and depth.stat().st_size > 0


def test_pids_listening_finds_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert os.getpid() in pids_listening(port)
    finally:
        sock.close()


def test_can_bind_detects_occupied_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert can_bind("127.0.0.1", port) is False
        assert choose_port("127.0.0.1", port) != port
    finally:
        sock.close()


def test_serve_reports_existing_app(monkeypatch, capsys):
    monkeypatch.setattr("depth_video.cli.existing_app_url", lambda port: "http://127.0.0.1:7860")
    code = main(["serve", "--port", "7860"])
    captured = capsys.readouterr()
    assert code == 0
    assert "already running at http://127.0.0.1:7860" in captured.out
    assert "python3 app.py --replace" in captured.out
    assert "http://127.0.0.1:7860" in captured.out


def test_serve_replace_skips_reuse(monkeypatch, capsys):
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append("run")

    monkeypatch.setattr("depth_video.cli.existing_app_url", lambda port: "http://127.0.0.1:7860")
    monkeypatch.setattr("depth_video.cli.terminate_port", lambda port: calls.append(f"kill:{port}"))
    monkeypatch.setattr("depth_video.cli.choose_port", lambda host, port: port)
    monkeypatch.setattr("uvicorn.run", fake_run)
    code = main(["serve", "--port", "7860", "--replace"])
    captured = capsys.readouterr()
    assert code == 0
    assert "kill:7860" in calls
    assert "run" in calls
    assert "already running" not in captured.out
