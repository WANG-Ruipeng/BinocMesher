#!/usr/bin/env python3
"""Validate this fork's Blender flow against static-scene camera/depth reprojection."""
import argparse
import json
from pathlib import Path
import numpy as np
from warped_ssim import prepare_flow

parser = argparse.ArgumentParser()
parser.add_argument("scene", type=Path)
parser.add_argument("--frame", type=int, default=25)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.scene / "frames"
frame = args.frame
depth = np.load(root / f"Depth/camera_0/Depth_0_0_{frame:04d}_0.npy")
current = np.load(root / f"camview/camera_0/camview_0_0_{frame:04d}_0.npz")
following = np.load(root / f"camview/camera_0/camview_0_0_{frame+1:04d}_0.npz")
height, width = depth.shape
flow, resized = prepare_flow(root / f"Flow/camera_0/Flow_0_0_{frame:04d}_0.npy", height, width)
assert not resized
x, y = np.meshgrid(np.arange(width), np.arange(height))
pixels = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float64)
rays = pixels @ np.linalg.inv(current["K"]).T
points = rays * depth[..., None]
relative = np.linalg.inv(following["T"]) @ current["T"]
points_next = points @ relative[:3, :3].T + relative[:3, 3]
projected = points_next @ following["K"].T
reference = projected[..., :2] / projected[..., 2:] - pixels[..., :2]
valid = ((depth > 1) & (depth < 1000) & (np.linalg.norm(flow, axis=2) > 0.05)
         & np.isfinite(reference).all(axis=2))
assert np.count_nonzero(valid) > 100
error = np.linalg.norm(reference - flow, axis=2)[valid]
result = {
    "frame": frame, "next_frame": frame+1,
    "flow_format": "infinigen-blender",
    "reference": "Static scene; z-depth and saved K/T camera reprojection",
    "pixels_compared": int(np.count_nonzero(valid)),
    "median_error_px": float(np.median(error)),
    "p90_error_px": float(np.percentile(error, 90)),
}
assert result["median_error_px"] < 0.05, result
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
