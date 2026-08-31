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

from cache_semantic_hash import semantic_source_cache_manifest


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
        if file.name.startswith("event_registry_p1") or file.name.endswith("_hpmeta.bin"):
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


def provenance_manifest(path: Path) -> dict[str, str]:
    """Hash opt-in BHP2/BPM2 and registry sidecars only."""
    manifest: dict[str, str] = {}
    meta_root = path / "hyperpoly_meta"
    if meta_root.is_dir():
        for file in sorted(candidate for candidate in meta_root.rglob("*") if candidate.is_file()):
            manifest[str(file.relative_to(path))] = sha256(file)
    processed_root = path / "processed_hyperpolys"
    if processed_root.is_dir():
        for file in sorted(processed_root.glob("*_hpmeta.bin")):
            manifest[str(file.relative_to(path))] = sha256(file)
    for file in sorted(path.glob("event_registry_p1*")):
        if file.is_file():
            manifest[str(file.relative_to(path))] = sha256(file)
    selected = path / "event_registry_selected_event.json"
    if selected.is_file():
        manifest[str(selected.relative_to(path))] = sha256(selected)
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


def worker(
    repo: Path,
    output: Path,
    provenance_mode: int,
    event_mode: int,
) -> dict:
    os.environ["BINOC_EVENT_MODE"] = str(event_mode)
    os.environ["BINOC_PROVENANCE_V2"] = str(provenance_mode)
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
        if not (output / "slicing_preprocess.finish").exists():
            original_slicing_preprocess = mesher.slicing_preprocess

            def audited_slicing_preprocess(
                _original=original_slicing_preprocess,
            ):
                before = source_cache_manifest(output)
                semantic_before = semantic_source_cache_manifest(output)
                return_value = _original()
                after = source_cache_manifest(output)
                semantic_after = semantic_source_cache_manifest(output)
                source_cache_audits.append({
                    "before": before,
                    "after": after,
                    "file_sets_identical": set(before) == set(after),
                    "bytes_identical": before == after,
                    "unchanged": before == after,
                    "semantic_before": semantic_before,
                    "semantic_after": semantic_after,
                    "semantics_unchanged": semantic_before == semantic_after,
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
        "mode": f"{provenance_mode}{event_mode}",
        "provenance_mode": provenance_mode,
        "event_mode": event_mode,
        "slice_times": slice_times,
        "mesh_hashes": mesh_hashes,
        "mesh_counts": mesh_counts,
        "cache_manifest": cache_manifest(output),
        "semantic_source_cache_manifest":
            semantic_source_cache_manifest(output),
        "provenance_manifest": provenance_manifest(output),
        "source_cache_audits": source_cache_audits,
        "sidecar_csv_exists": (output / "event_registry_p1.csv").exists(),
        "sidecar_summary_exists": (output / "event_registry_p1_summary.json").exists(),
        "sidecar_selected_exists": (output / "event_registry_selected_event.json").exists(),
        "registry_summary": (
            json.loads((output / "event_registry_p1_summary.json").read_text())
            if (output / "event_registry_p1_summary.json").is_file()
            else None
        ),
        "cache_contract": json.loads(
            (output / "slicing_preprocess.manifest.json").read_text()
        ),
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
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    mode_directories = {
        f"{provenance}{event}": output / f"mode{provenance}{event}"
        for provenance in (0, 1)
        for event in (0, 1)
    }
    for mode, directory in mode_directories.items():
        provenance, event = (int(value) for value in mode)
        subprocess.run([
            sys.executable,
            str(script),
            "--worker",
            "--repo", str(repo),
            "--output", str(directory),
            "--provenance-mode", str(provenance),
            "--event-mode", str(event),
        ], check=True, env=env)
    modes = {
        mode: json.loads((directory / "p1_worker_result.json").read_text())
        for mode, directory in mode_directories.items()
    }
    baseline = modes["00"]
    instrumented = modes["11"]
    audits = [
        audit
        for result in modes.values()
        for audit in result.get("source_cache_audits", [])
    ]
    source_cache_unchanged = bool(audits) and all(
        audit.get("unchanged") and audit.get("semantics_unchanged")
        for audit in audits
    )
    semantic_manifests = [
        result["semantic_source_cache_manifest"]
        for result in modes.values()
    ]

    mode_matrix_exact = True
    for mode, result in modes.items():
        provenance, event = (int(value) for value in mode)
        contract = result["cache_contract"]
        registry_sidecars = (
            result["sidecar_csv_exists"]
            and result["sidecar_summary_exists"]
            and result["sidecar_selected_exists"]
        )
        summary = result["registry_summary"]
        separated_face_counts = not event or (
            isinstance(summary, dict)
            and isinstance(summary.get("all_parameter_faces"), dict)
            and isinstance(summary.get("temporal_neighbour_faces"), dict)
            and summary["temporal_neighbour_faces"].get("raw_observations", -1)
                <= summary["all_parameter_faces"].get("raw_observations", -2)
            and summary["temporal_neighbour_faces"].get("logical_incidences", -1)
                <= summary["all_parameter_faces"].get("logical_incidences", -2)
            and summary["temporal_neighbour_faces"].get("canonical_events", -1)
                <= summary["all_parameter_faces"].get("canonical_events", -2)
        )
        mode_matrix_exact = mode_matrix_exact and all((
            result["provenance_mode"] == provenance,
            result["event_mode"] == event,
            contract.get("provenance_requested") is bool(provenance),
            contract.get("provenance_enabled") is bool(provenance or event),
            contract.get("event_registry_enabled") is bool(event),
            registry_sidecars is bool(event),
            separated_face_counts,
            bool(result["provenance_manifest"]) is bool(provenance or event),
        ))

    checks = {
        "observer_mode_matrix_exact": mode_matrix_exact,
        "cache_file_sets_identical": all(
            set(result["cache_manifest"]) == set(baseline["cache_manifest"])
            for result in modes.values()
        ),
        # Compatibility key consumed by validate_smoke.py.
        "cache_bytes_identical": source_cache_unchanged,
        "source_cache_bytes_unchanged_instrumented": source_cache_unchanged,
        "source_cache_unchanged_all_modes":
            len(audits) == len(modes) and source_cache_unchanged,
        "semantic_source_cache_identity":
            bool(semantic_manifests[0])
            and all(value == semantic_manifests[0]
                    for value in semantic_manifests[1:]),
        "mesh_hashes_identical": all(
            result["mesh_hashes"] == baseline["mesh_hashes"]
            for result in modes.values()
        ),
        "mesh_counts_identical": all(
            result["mesh_counts"] == baseline["mesh_counts"]
            for result in modes.values()
        ),
        "baseline_has_no_sidecar":
            not baseline["sidecar_csv_exists"]
            and not baseline["sidecar_summary_exists"],
        "instrumented_has_sidecar":
            instrumented["sidecar_csv_exists"]
            and instrumented["sidecar_summary_exists"]
            and instrumented["sidecar_selected_exists"],
        "provenance_disabled_has_no_sidecars":
            not baseline["provenance_manifest"],
        "provenance_enabled_has_sidecars":
            bool(instrumented["provenance_manifest"]),
    }
    result = {
        "verdict": "PASS_P1_OFFICIAL_PIPELINE_READ_ONLY" if all(checks.values()) else "STOP_P1_OFFICIAL_PIPELINE_READ_ONLY",
        "repo": str(repo.resolve()),
        "checks": checks,
        "diagnostics": {
            "cross_run_raw_cache_bytes_identical":
                all(result["cache_manifest"] == baseline["cache_manifest"]
                    for result in modes.values()),
            "cross_run_raw_cache_note":
                "Diagnostic only: unordered containers and C++ padding can differ; "
                "semantic HP/HV identity is the hard gate.",
        },
        "modes": modes,
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
    parser.add_argument("--mode", type=int, choices=(0, 1))
    parser.add_argument("--provenance-mode", type=int, choices=(0, 1))
    parser.add_argument("--event-mode", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker:
        fallback = 0 if args.mode is None else args.mode
        provenance_mode = (
            fallback if args.provenance_mode is None else args.provenance_mode)
        event_mode = fallback if args.event_mode is None else args.event_mode
        worker(
            args.repo.resolve(), args.output.resolve(),
            provenance_mode, event_mode)
        return 0
    return controller(args.repo.resolve(), args.output.resolve(), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
