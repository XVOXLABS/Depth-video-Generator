from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from depth_video.pipeline import ConversionOptions, convert_video
from depth_video.video_io import write_demo_clip
from depth_video.weights import ENCODERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depth-video",
        description="Turn any video into a temporally consistent depth video with Video Depth Anything.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start the local web app")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=7860)

    convert = sub.add_parser("convert", help="Convert a video from the command line")
    convert.add_argument("input", type=Path, help="Input video path")
    convert.add_argument("-o", "--output", type=Path, help="Output MP4 path")
    convert.add_argument("--encoder", choices=ENCODERS, default="vits")
    convert.add_argument("--metric", action="store_true")
    convert.add_argument("--input-size", type=int, default=518)
    convert.add_argument("--max-res", type=int, default=1280)
    convert.add_argument("--target-fps", type=float, default=-1)
    convert.add_argument("--max-len", type=int, default=-1, help="Max frames, -1 for the full video")
    convert.add_argument("--colormap", default="inferno", choices=["inferno", "magma", "turbo", "plasma", "viridis", "gray"])
    convert.add_argument("--invert", action="store_true")
    convert.add_argument("--grayscale", action="store_true")
    convert.add_argument("--no-audio", action="store_true")
    convert.add_argument("--side-by-side", action="store_true")
    convert.add_argument("--mode", choices=["windowed", "streaming"], default="windowed")
    convert.add_argument("--fp32", action="store_true")
    convert.add_argument("--fake", action="store_true", help="Use a luminance stand-in instead of the neural model")

    demo = sub.add_parser("demo", help="Create a short sample clip and convert it")
    demo.add_argument("-o", "--output-dir", type=Path, default=Path("examples"))
    demo.add_argument("--encoder", choices=ENCODERS, default="vits")
    demo.add_argument("--fake", action="store_true")
    return parser


def existing_app_url(port: int) -> str | None:
    """Return the local UI URL if this app is already serving on `port`."""
    for host in ("127.0.0.1", "localhost"):
        url = f"http://{host}:{port}/api/health"
        try:
            with urlopen(url, timeout=1.0) as response:
                payload = json.loads(response.read().decode())
        except (URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
            continue
        if payload.get("ok"):
            return f"http://{host}:{port}"
    return None


def can_bind(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bind_host = "0.0.0.0" if host in {"0.0.0.0", "::", ""} else host
    try:
        sock.bind((bind_host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def choose_port(host: str, port: int) -> int:
    if can_bind(host, port):
        return port
    for candidate in range(port + 1, port + 20):
        if can_bind(host, candidate):
            return candidate
    raise OSError(f"No free port in {port}-{port + 19}")


def cmd_serve(args: argparse.Namespace) -> int:
    already = existing_app_url(args.port)
    if already:
        print(f"Depth Video Generator is already running at {already}")
        print("Open that URL in your browser. To start another copy: python3 app.py --port 7861")
        return 0

    try:
        port = choose_port(args.host, args.port)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if port != args.port:
        print(f"Port {args.port} is in use; serving on {port} instead.")

    import uvicorn

    print(f"Open http://127.0.0.1:{port}")
    uvicorn.run("depth_video.server:app", host=args.host, port=port, reload=False)
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    if args.fake:
        os.environ["DEPTH_VIDEO_BACKEND"] = "fake"
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        print("Pass a real video path, or run: python3 -m depth_video demo", file=sys.stderr)
        return 2
    output = args.output or args.input.with_name(args.input.stem + "_depth.mp4")
    options = ConversionOptions(
        encoder=args.encoder,
        metric=args.metric,
        input_size=args.input_size,
        max_res=args.max_res,
        target_fps=args.target_fps,
        max_len=args.max_len,
        colormap=args.colormap,
        invert=args.invert,
        grayscale=args.grayscale,
        keep_audio=not args.no_audio,
        layout="side_by_side" if args.side_by_side else "depth",
        mode=args.mode,
        use_fp16=not args.fp32,
    )

    def on_progress(payload: dict) -> None:
        pct = payload.get("progress")
        msg = payload.get("message", "")
        if pct is None:
            print(msg)
        else:
            print(f"[{float(pct)*100:5.1f}%] {msg}")

    result = convert_video(args.input, output, options=options, progress=on_progress)
    print(
        f"Saved {result.output_path} ({result.frames} frames, {result.width}x{result.height} @ {result.fps:.2f} fps, {result.elapsed_s:.1f}s on {result.device})"
    )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = write_demo_clip(args.output_dir / "demo.mp4")
    print(f"Wrote sample clip {source}")
    demo_args = argparse.Namespace(
        input=source,
        output=args.output_dir / "demo_depth.mp4",
        encoder=args.encoder,
        metric=False,
        input_size=518,
        max_res=640,
        target_fps=-1,
        max_len=-1,
        colormap="inferno",
        invert=False,
        grayscale=False,
        no_audio=False,
        side_by_side=False,
        mode="windowed",
        fp32=False,
        fake=args.fake,
    )
    return cmd_convert(demo_args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "convert":
        return cmd_convert(args)
    if args.command == "demo":
        return cmd_demo(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
