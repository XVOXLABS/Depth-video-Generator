# Third-party notices

This application vendors inference code from:

## Video Depth Anything
- Source: https://github.com/DepthAnything/Video-Depth-Anything
- Paper: Chen et al., "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos", arXiv:2501.12375, CVPR 2025
- License: Apache-2.0 (see `third_party/LICENSE-Video-Depth-Anything`)
- Small model weights: Apache-2.0
- Base and Large model weights: CC-BY-NC-4.0 (non-commercial)

The `third_party/video_depth_anything` and `third_party/utils` trees are copied from that repository for local inference. DINOv2 backbone components included there are from Meta, and the temporal motion module traces to AnimateDiff (Apache-2.0).
