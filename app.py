#!/usr/bin/env python3
"""Launch the Depth Video Generator web app."""

from depth_video.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve", *(__import__("sys").argv[1:])]))
