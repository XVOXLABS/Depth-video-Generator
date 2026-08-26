# Depth Video Generator

Turn **any video of any length** into a temporally consistent depth video, using [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) (CVPR 2025 Highlight).

The original research demo loads the whole clip into RAM. This app streams overlapping 32-frame windows, writes H.264 incrementally, and keeps memory bounded, so a 10-second clip and a 2-hour film use the same working set.

## Features

- Local web app: drag-and-drop a video, watch progress, preview, download
- CLI for batch jobs
- Official windowed inference (quality) plus experimental streaming mode (lower VRAM)
- Relative or metric depth, Small / Base / Large encoders
- Colormaps, grayscale, side-by-side layout, original audio muxed back in
- Auto-download of Hugging Face checkpoints
- CUDA, Apple Silicon, or CPU

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) on `PATH` (`ffmpeg` and `ffprobe`)
- A NVIDIA GPU is strongly recommended for the neural model. CPU works but is slow.

## Install

```bash
git clone https://github.com/XVOXLABS/Depth-video-Generator.git
cd Depth-video-Generator
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

This project’s commands use `python3`. Some Linux images (including this one) do not ship a `python` binary.

For NVIDIA GPUs, install a CUDA build of PyTorch instead of the CPU wheel:

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Weights download automatically on first run. To prefetch the Small model:

```bash
python3 -m depth_video convert --help
```

The first conversion of a given encoder/metric pair stores checkpoints in `./checkpoints/` (override with `DEPTH_VIDEO_CHECKPOINTS`).

## Web app

```bash
python3 app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860). Drop a video of any length, pick a model, and generate. The Small encoder is the default (Apache-2.0, fastest). Use Base or Large when you have more VRAM and want higher quality.

Useful environment variables:

| Variable | Meaning |
|---|---|
| `DEPTH_VIDEO_DEVICE` | Force `cuda`, `mps`, or `cpu` |
| `DEPTH_VIDEO_BACKEND` | Set to `fake` for a luminance stand-in (no weights, for UI/dev) |
| `DEPTH_VIDEO_CHECKPOINTS` | Where `.pth` files are stored |
| `DEPTH_VIDEO_OUTPUTS` | Where finished MP4s are written |

## Command line

```bash
# Create a 2-second sample and convert it (no input file required)
python3 -m depth_video demo

# Convert your own video
python3 -m depth_video convert your_video.mp4 -o depth.mp4 --encoder vits

python3 -m depth_video convert your_video.mp4 --side-by-side --colormap magma --encoder vitl

python3 -m depth_video convert long.mkv --mode streaming --max-res 960
```

`./depth-video` is a wrapper that calls `python3` (then `python` if needed), so `./depth-video demo` also works.

`--max-len -1` (default) processes the entire file. `--target-fps -1` keeps the source frame rate.

## How long videos work

Video Depth Anything infers 32-frame windows with a 10-frame overlap, then scale-shifts consecutive windows so depth stays consistent. This app:

1. Decodes frames sequentially (never the whole movie)
2. Runs the official 32-frame window
3. Aligns the overlap the same way as the paper code
4. Color-maps with a slowly adapting range
5. Pipes RGB frames into FFmpeg immediately

Peak RAM therefore depends on resolution and the 32-frame batch, **not** on duration.

Use `--mode streaming` only if VRAM is tight. That path is the authors' experimental one-frame cache and is slightly less accurate.

## Model licenses

| Encoder | Relative / metric weights | License |
|---|---|---|
| Small (`vits`) | 28.4M | Apache-2.0 |
| Base (`vitb`) | 113.1M | CC-BY-NC-4.0 |
| Large (`vitl`) | 381.8M | CC-BY-NC-4.0 |

Base and Large checkpoints are **non-commercial**. For commercial use, stick to Small or contact the Video Depth Anything authors.

## Tests

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

Tests use a fake luminance backend and synthetic FFmpeg clips, so they do not download multi-hundred-megabyte weights.

## Docker

```bash
docker build -t depth-video-generator .
docker run --gpus all -p 7860:7860 depth-video-generator
```

## Credits

Inference architecture, window schedule, and checkpoints come from [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything):

```
Chen, Sili et al. "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos." arXiv:2501.12375, 2025.
```
