from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from depth_video.pipeline import ConversionOptions, convert_video
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
    return parser


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("depth_video.server:app", host=args.host, port=args.port, reload=False)
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    if args.fake:
        os.environ["DEPTH_VIDEO_BACKEND"] = "fake"
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "convert":
        return cmd_convert(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
