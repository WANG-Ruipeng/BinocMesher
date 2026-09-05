#!/usr/bin/env python3
"""Compute BinocMesher's frame-to-frame warped SSIM from Infinigen outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


IMAGE_RE = re.compile(r"Image_0_0_(\d+)_0\.png$")
FLOW_RE = re.compile(r"Flow_0_0_(\d+)_0\.npy$")
SSIM_RADIUS = 3  # structural_similarity's default win_size is 7.


def indexed_files(folder: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(folder.iterdir()):
        match = pattern.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] < 3:
        raise RuntimeError(f"Expected at least three channels: {path} {image.shape}")
    return image[..., :3]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p1": None, "min": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p1": float(np.percentile(array, 1)),
        "min": float(np.min(array)),
    }


def jump_summary(values: list[float]) -> dict[str, float | int | None]:
    if len(values) < 3:
        return {"count": 0, "max": None, "p99": None}
    array = np.asarray(values, dtype=np.float64)
    jumps = array[:-2] + array[2:] - 2.0 * array[1:-1]
    return {
        "count": int(jumps.size),
        "max": float(np.max(jumps)),
        "p99": float(np.percentile(jumps, 99)),
    }


def prepare_flow(
    flow_path: Path, height: int, width: int,
    flow_format: str = "infinigen-blender",
) -> tuple[np.ndarray, bool]:
    flow = np.load(flow_path)
    if flow.ndim != 3 or flow.shape[2] < 2:
        raise RuntimeError(f"Expected HxWx>=2 flow: {flow_path} {flow.shape}")
    flow = flow[..., :2].astype(np.float32, copy=True)
    if flow_format == "infinigen-blender":
        # This fork exports Cycles Vector.ZW through an RGB compositor.
        # In array coordinates its Flow.npy contains (-dx, +dy).
        # Verified independently against camera/depth reprojection.
        flow[..., 0] *= -1
    old_height, old_width = flow.shape[:2]
    resized = (old_height, old_width) != (height, width)
    if resized:
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= width / old_width
        flow[..., 1] *= height / old_height
    return flow, resized


def valid_ssim_mean(ssim_map: np.ndarray, valid: np.ndarray) -> float:
    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)
    kernel = np.ones((2 * SSIM_RADIUS + 1, 2 * SSIM_RADIUS + 1), np.uint8)
    valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    valid[:SSIM_RADIUS, :] = False
    valid[-SSIM_RADIUS:, :] = False
    valid[:, :SSIM_RADIUS] = False
    valid[:, -SSIM_RADIUS:] = False
    if not np.any(valid):
        return float("nan")
    return float(np.mean(ssim_map[valid]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path, help="Scene folder containing frames/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument(
        "--flow-format", choices=["infinigen-blender", "forward"],
        default="infinigen-blender")
    args = parser.parse_args()

    image_dir = args.scene / "frames" / "Image" / "camera_0"
    flow_dir = args.scene / "frames" / "Flow" / "camera_0"
    if not image_dir.is_dir() or not flow_dir.is_dir():
        raise RuntimeError(f"Missing Image/Flow directory below {args.scene / 'frames'}")

    images = indexed_files(image_dir, IMAGE_RE)
    flows = indexed_files(flow_dir, FLOW_RE)
    if args.expected_frames is not None:
        if args.expected_frames < 2:
            raise ValueError("--expected-frames must be at least 2")
        stop = args.start_frame + args.expected_frames
        expected = set(range(args.start_frame, stop))
        missing_images = sorted(expected - images.keys())
        if missing_images:
            raise RuntimeError(f"Missing expected image frames: {missing_images[:20]}")
        missing_flows = sorted(set(range(args.start_frame, stop - 1)) - flows.keys())
        if missing_flows:
            raise RuntimeError(f"Missing required forward-flow frames: {missing_flows[:20]}")
        # Only score the requested interval.
        images = {frame: path for frame, path in images.items() if frame in expected}
        flows = {frame: path for frame, path in flows.items() if frame in expected}

    pair_frames = sorted(frame for frame in images if frame + 1 in images and frame in flows)
    if not pair_frames:
        raise RuntimeError("No consecutive Image_i, Image_i+1, Flow_i triples found")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | bool]] = []
    worst: tuple[float, int, np.ndarray, np.ndarray, np.ndarray] | None = None

    for frame in pair_frames:
        current = load_rgb(images[frame])
        following = load_rgb(images[frame + 1])
        if current.shape != following.shape:
            raise RuntimeError(
                f"Image shape mismatch at {frame}->{frame + 1}: "
                f"{current.shape} != {following.shape}"
            )
        height, width = current.shape[:2]
        flow, resized = prepare_flow(flows[frame], height, width, args.flow_format)
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing="xy",
        )
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        finite = np.isfinite(map_x) & np.isfinite(map_y)
        valid = (
            finite
            & (map_x >= 0.0)
            & (map_x <= width - 1.0)
            & (map_y >= 0.0)
            & (map_y <= height - 1.0)
        )
        map_x = np.nan_to_num(map_x, nan=-1.0, posinf=-1.0, neginf=-1.0)
        map_y = np.nan_to_num(map_y, nan=-1.0, posinf=-1.0, neginf=-1.0)
        warped = cv2.remap(
            following,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        full_score, score_map = structural_similarity(
            current,
            warped,
            channel_axis=2,
            data_range=255,
            full=True,
        )
        valid_score = valid_ssim_mean(score_map, valid)
        if not np.isfinite(valid_score):
            raise RuntimeError(f"No valid SSIM support at {frame}->{frame + 1}")
        unwarped_score = structural_similarity(
            current,
            following,
            channel_axis=2,
            data_range=255,
        )
        row = {
            "frame": frame,
            "next_frame": frame + 1,
            "ssim_warped_full": float(full_score),
            "ssim_warped_valid": valid_score,
            "ssim_unwarped": float(unwarped_score),
            "valid_fraction": float(np.mean(valid)),
            "flow_resized": resized,
        }
        rows.append(row)
        if np.isfinite(valid_score) and (worst is None or valid_score < worst[0]):
            worst = (valid_score, frame, current.copy(), warped.copy(), following.copy())

    with (args.output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    full_values = [float(row["ssim_warped_full"]) for row in rows]
    valid_values = [float(row["ssim_warped_valid"]) for row in rows]
    unwarped_values = [float(row["ssim_unwarped"]) for row in rows]
    summary = {
        "schema": "binoc-warped-ssim-v2",
        "scene": str(args.scene.resolve()),
        "flow_format": args.flow_format,
        "flow_convention": (
            "Sample next-frame image at (x - raw_flow_x, y + raw_flow_y)"
            if args.flow_format == "infinigen-blender" else
            "Sample next-frame image at grid + forward_flow"
        ),
        "frame_start": pair_frames[0],
        "frame_end": pair_frames[-1] + 1,
        "images_found": len(images),
        "flows_found": len(flows),
        "pairs_scored": len(rows),
        "flow_resized_pairs": sum(bool(row["flow_resized"]) for row in rows),
        "warped_full": summarize(full_values),
        "warped_valid": summarize(valid_values),
        "unwarped": summarize(unwarped_values),
        "jump_warped_full": jump_summary(full_values),
        "jump_warped_valid": jump_summary(valid_values),
        "mean_valid_fraction": float(
            np.mean([float(row["valid_fraction"]) for row in rows])
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if worst is not None:
        _, frame, current, warped, following = worst
        cv2.imwrite(str(args.output / f"worst_{frame:04d}_current.png"), current)
        cv2.imwrite(str(args.output / f"worst_{frame:04d}_warped_next.png"), warped)
        cv2.imwrite(str(args.output / f"worst_{frame:04d}_next.png"), following)

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
