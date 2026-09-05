#!/usr/bin/env python3
"""Stage reusable geometry in a fresh run; never reuse old rendered frames."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def stage(source_root: Path, destination: Path, method: str, frames: int, ppc: float):
    manifest = dict(
        line.split("=", 1) for line in (source_root / "manifest.txt").read_text().splitlines()
        if "=" in line
    )
    if int(manifest["frames"]) != frames or manifest["resolution"] != "960x540":
        raise ValueError("Source frame range/resolution does not match the requested run")
    if method == "binoc" and float(manifest["pixels_per_cube"]) != ppc:
        raise ValueError("Binoc cache resolution does not match the requested ppc")
    source = source_root / method / "0"
    required = ["FINISH_coarse", "FINISH_populate"]
    if method == "binoc":
        required.append("FINISH_fineterrain")
        if not (source / "fine" / "HyperMesh").is_dir():
            raise FileNotFoundError("Missing Binoc 4D cache")
        if (source / "fine" / "mesher_backend.txt").read_text().splitlines() != [
            "BinocMesher", "1", str(frames)]:
            raise ValueError("Binoc geometry metadata does not match the source manifest")
    for marker in required:
        if not (source / "logs" / marker).is_file():
            raise FileNotFoundError(source / "logs" / marker)
    if not (source / "coarse" / "scene.blend").is_file():
        raise FileNotFoundError(source / "coarse" / "scene.blend")

    # mkdir without exist_ok prevents overwriting an existing recovery attempt.
    destination.mkdir()
    scene = destination / "0"
    logs = scene / "logs"
    logs.mkdir(parents=True)
    shutil.copytree(source / "coarse", scene / "coarse", symlinks=True)
    if method == "binoc":
        shutil.copytree(source / "fine", scene / "fine", symlinks=True)
        assets = scene / "fine" / "assets"
        if assets.is_symlink():
            assets.unlink()
            assets.symlink_to(scene / "coarse" / "assets", target_is_directory=True)
    for marker in required:
        shutil.copy2(source / "logs" / marker, logs / marker)
    with (destination / "scenes_db.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "configs"])
        writer.writeheader()
        writer.writerow({"seed": "0", "configs": repr(
            ["mountain", "monocular", "simple", "no_assets"])})
    record = {
        "source": str(source.resolve()),
        "reused_markers": required,
        "reused_rendered_frames": False,
        "binoc_cache_copied": method == "binoc",
    }
    (destination / "reuse.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--method", choices=["binoc", "ocmesher96", "ocmesher24"], required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--ppc", type=float, required=True)
    args = parser.parse_args()
    stage(args.source_root, args.destination, args.method, args.frames, args.ppc)
