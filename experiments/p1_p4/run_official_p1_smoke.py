#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_manifest(path: Path) -> dict[str, str]:
    included_roots = {"hypervertices", "hyperpolys", "processed_hyperpolys"}
    manifest: dict[str, str] = {}
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = file.relative_to(path)
        if relative.parts[0] not in included_roots:
            continue
        if file.name.startswith("event_registry_p1"):
            continue
        manifest[str(relative)] = sha256(file)
    return manifest


def source_cache_manifest(path: Path) -> dict[str, str]:
    """Hash only observer input caches, excluding generated slicing output."""
    included_roots = {"hypervertices", "hyperpolys"}
    manifest: dict[str, str] = {}
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = file.relative_to(path)
        if relative.parts[0] not in included_roots:
            continue
        manifest[str(relative)] = sha256(file)
    return manifest


def mesh_hash(meshes, in_view_tags) -> str:
    digest = hashlib.sha256()
    for mesh, tags in zip(meshes, in_view_tags):
        vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64))
        faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64))
        tags_array = np.ascontiguousarray(np.asarray(tags, dtype=np.int8))
        digest.update(vertices.shape.__repr__().encode())
        digest.update(vertices.tobytes())
        digest.update(faces.shape.__repr__().encode())
        digest.update(faces.tobytes())
        digest.update(tags_array.tobytes())
    return digest.hexdigest()


def make_inputs():
    camera_poses, intrinsics, heights, widths, times = [], [], [], [], []
    width, height = 320, 180
    fx = fy = 420.0
    for index in range(8):
        camera_poses.append(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.35 + 0.1 * index],
            [0.0, -1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64))
        intrinsics.append(np.array([
            [fx, 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64))
        heights.append(height)
        widths.append(width)
        times.append((0.5 + index) / 24.0)
    return (camera_poses, intrinsics, heights, widths, times)


def terrain_sdf(points: np.ndarray) -> np.ndarray:
    height = 0.16 * np.sin(1.3 * points[:, 0]) + 0.11 * np.cos(1.1 * points[:, 1])
    return points[:, 2] - height


def worker(repo: Path, output: Path, mode: int) -> dict:
    os.environ["BINOC_EVENT_MODE"] = str(mode)
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(repo))
    from binocmesher import BinocMesher  # type: ignore

    output.mkdir(parents=True, exist_ok=True)
    cameras = make_inputs()
    bounds = [-4.0, 4.0, -4.0, 4.0, -1.0, 1.0]
    slice_times = [cameras[4][index] for index in (1, 3, 5, 7)]
    mesh_hashes = []
    mesh_counts = []
    source_cache_audits = []
    for slicing_time in slice_times:
        mesher = BinocMesher(
            cameras,
            bounds=bounds,
            slicing_time=slicing_time,
            pixels_per_cube=96,
            pixels_per_cube_coarse=192,
            pixels_per_cube_outview=384,
            min_dist=0.25,
            simplify_occluded=False,
            relax_margin=0,
            boundary_margin=1,
            relax_iters=0,
            n_coarse_nodes=512,
            bisection_iters=1,
            fading_time=1.0 / 24.0,
            seed_stride=2,
            medium_group=2048,
            fine_group=512,
            bisection_group=100000,
            path=output,
        )
        # Audit the exact observer call in the same process. This avoids
        # comparing unstable object representations written by independent
        # upstream runs.
        if mode > 0 and not (output / "slicing_preprocess.finish").exists():
            original_slicing_preprocess = mesher.slicing_preprocess

            def audited_slicing_preprocess(
                _original=original_slicing_preprocess,
            ):
                before = source_cache_manifest(output)
                return_value = _original()
                after = source_cache_manifest(output)
                source_cache_audits.append({
                    "before": before,
                    "after": after,
                    "file_sets_identical": set(before) == set(after),
                    "bytes_identical": before == after,
                    "unchanged": before == after,
                })
                return return_value

            mesher.slicing_preprocess = audited_slicing_preprocess

        meshes, tags = mesher([terrain_sdf])
        mesh_hashes.append(mesh_hash(meshes, tags))
        mesh_counts.append({
            "vertices": [int(len(mesh.vertices)) for mesh in meshes],
            "faces": [int(len(mesh.faces)) for mesh in meshes],
        })
    result = {
        "mode": mode,
        "slice_times": slice_times,
        "mesh_hashes": mesh_hashes,
        "mesh_counts": mesh_counts,
        "cache_manifest": cache_manifest(output),
        "source_cache_audits": source_cache_audits,
        "sidecar_csv_exists": (output / "event_registry_p1.csv").exists(),
        "sidecar_summary_exists": (output / "event_registry_p1_summary.json").exists(),
    }
    (output / "p1_worker_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def controller(repo: Path, output: Path, force: bool) -> int:
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists; pass --force only for a disposable result directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    script = Path(__file__).resolve()
    baseline_dir = output / "baseline"
    instrumented_dir = output / "instrumented"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    for mode, directory in ((0, baseline_dir), (1, instrumented_dir)):
        subprocess.run([
            sys.executable,
            str(script),
            "--worker",
            "--repo", str(repo),
            "--output", str(directory),
            "--mode", str(mode),
        ], check=True, env=env)
    baseline = json.loads((baseline_dir / "p1_worker_result.json").read_text())
    instrumented = json.loads((instrumented_dir / "p1_worker_result.json").read_text())
    source_cache_audits = instrumented.get("source_cache_audits", [])
    source_cache_unchanged = (
        bool(source_cache_audits)
        and all(item.get("unchanged") for item in source_cache_audits)
    )

    checks = {
        "cache_file_sets_identical": set(baseline["cache_manifest"]) == set(instrumented["cache_manifest"]),
        # Compatibility key consumed by validate_smoke.py. It now means
        # exact source-cache identity immediately before/after the observer.
        "cache_bytes_identical": source_cache_unchanged,
        "source_cache_bytes_unchanged_instrumented": source_cache_unchanged,
        "mesh_hashes_identical": baseline["mesh_hashes"] == instrumented["mesh_hashes"],
        "mesh_counts_identical": baseline["mesh_counts"] == instrumented["mesh_counts"],
        "baseline_has_no_sidecar": not baseline["sidecar_csv_exists"] and not baseline["sidecar_summary_exists"],
        "instrumented_has_sidecar": instrumented["sidecar_csv_exists"] and instrumented["sidecar_summary_exists"],
    }
    result = {
        "verdict": "PASS_P1_OFFICIAL_PIPELINE_READ_ONLY" if all(checks.values()) else "STOP_P1_OFFICIAL_PIPELINE_READ_ONLY",
        "repo": str(repo.resolve()),
        "checks": checks,
        "diagnostics": {
            "cross_run_raw_cache_bytes_identical":
                baseline["cache_manifest"] == instrumented["cache_manifest"],
            "cross_run_raw_cache_note":
                "Non-gating because upstream serializes unordered/padded C++ objects.",
        },
        "baseline": baseline,
        "instrumented": instrumented,
    }
    (output / "p1_official_smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(result["verdict"])
    return 0 if result["verdict"].startswith("PASS") else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        worker(args.repo.resolve(), args.output.resolve(), args.mode)
        return 0
    return controller(args.repo.resolve(), args.output.resolve(), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
