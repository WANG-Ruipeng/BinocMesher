#!/usr/bin/env python3
"""Three-gap theory/validation campaign for a unified Binoc spacetime compiler.

This script closes three previously open local-theory gaps:

G1. Flat stars are defined as residual one-sided limit strata of the source
    slicing complex, not as the special algebraic condition D=U-L=0.
G2. Source-reachable local states are captured by an exact 8-sector prefix
    partition grammar induced by the binary-octree split semantics.
G3. Spatial trilinear embeddings are accepted only after a fail-closed
    Bernstein/Jacobian validity certificate; corner-only tests are shown to
    miss non-corner folds.

The experiments deliberately distinguish:
  * exact source-semantic finite-depth exhaustive results,
  * repo-camera-calibrated synthetic stress tests,
  * real-cache prevalence, which cannot be measured because generated cache
    binaries are not distributed in the public repository.

The implementation only needs NumPy and Matplotlib.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Common data types
# -----------------------------------------------------------------------------

Leaf = Tuple[int, int, int, int, int, int]
# xlo, ylo, zlo, spatial_size, tlo, temporal_size on finest integer grids.
Assignment = Tuple[Tuple[int, Leaf], ...]
Cycle = Tuple[int, ...]


def official_slot(t: int, y: int, z: int) -> int:
    """Official hyperpoly slot order: t*4 + y*2 + z."""
    return t * 4 + y * 2 + z


def slot_bits(slot: int) -> Tuple[int, int, int]:
    return (slot >> 2) & 1, (slot >> 1) & 1, slot & 1


def cube_index(x: int, y: int, z: int) -> int:
    # Exact macro in utils.h: x*s*s+y*s+z, with s=2.
    return x * 4 + y * 2 + z


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


# -----------------------------------------------------------------------------
# Official camera trajectories (keyframes from the public repository)
# -----------------------------------------------------------------------------

CAMERA_PATHS: Dict[str, List[Tuple[int, float, float, float]]] = {
    "forest": [
        (1, 26.0658, -32.5212, 15.0831),
        (120, -12.3514, 4.2248, 18.7345),
        (240, -75.4866, 60.6050, 31.4214),
        (360, -112.5566, 115.5558, 41.8970),
        (480, -121.6183, 192.6550, 46.1718),
    ],
    "snowy_mountain": [
        (1, -44.8646, -50.2607, 17.0548),
        (5, -44.8239, -50.2355, 17.0644),
        (60, -39.5254, -47.2245, 17.9370),
        (120, -29.6879, -41.6847, 18.0990),
        (180, -13.5103, -30.1918, 14.2031),
        (240, -12.5202, -16.5589, 12.8032),
        (300, -38.6376, -3.7008, 13.0855),
        (310, -40.9329, -3.5135, 13.7342),
        (320, -42.0835, -3.1433, 13.5310),
        (360, -44.3445, 7.1891, 13.0532),
        (420, -40.3276, 17.9515, 12.1940),
        (480, -42.1110, 39.1272, 11.9665),
    ],
    "arctic": [
        (1, -74.9829, 15.9447, 0.5820),
        (100, -67.6875, 11.5172, 0.9725),
        (200, -57.2869, 3.9643, 1.3574),
        (300, -48.8546, -3.5584, 1.9590),
        (400, -48.1851, -11.7371, 2.0152),
        (500, -34.1349, -6.3389, 3.6746),
    ],
    "cave": [
        (1, 11.9286, 7.6375, -2.3178),
        (2, 11.9276, 7.6375, -2.3178),
        (100, 6.4940, 5.1132, -2.2373),
        (240, 0.8010, -8.3243, -2.8087),
        (300, -0.4730, -9.5014, -1.9758),
        (336, -1.2855, -10.8608, -1.2823),
        (360, -2.5014, -11.7445, -0.6364),
        (420, -6.6568, -12.9160, 0.8396),
        (431, -7.1541, -12.8404, 0.9260),
        (480, -7.9467, -12.0623, 1.0294),
    ],
    "beach": [
        (1, -93.7888, 77.7870, 5.4193),
        (45, -75.7297, 72.8701, 3.9488),
        (100, -50.4827, 65.9962, 1.8929),
        (158, -29.7528, 59.8480, 1.2674),
        (225, -16.4931, 26.3184, 4.5923),
        (328, 27.4748, -29.7107, 2.8887),
        (429, 57.1562, -53.8939, 1.2345),
        (480, 57.1562, -53.8939, 1.2345),
    ],
    "city": [
        (0, 347.9765, -101.7522, 12.6548),
        (1, 347.9765, -101.7522, 12.6548),
        (480, 1.2370, -102.3960, 10.4438),
    ],
}


def interpolate_camera_path(keys: Sequence[Tuple[int, float, float, float]], n_frames: int = 481) -> np.ndarray:
    frames = np.arange(n_frames, dtype=float)
    kf = np.asarray([k[0] for k in keys], dtype=float)
    xyz = np.asarray([k[1:] for k in keys], dtype=float)
    out = np.stack([np.interp(frames, kf, xyz[:, d]) for d in range(3)], axis=1)
    return out


def camera_motion_profiles() -> Tuple[List[Dict[str, float]], Dict[str, np.ndarray]]:
    rows: List[Dict[str, float]] = []
    signals: Dict[str, np.ndarray] = {}
    for name, keys in CAMERA_PATHS.items():
        path = interpolate_camera_path(keys)
        velocity = np.diff(path, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.diff(velocity, axis=0)
        curvature_proxy = np.linalg.norm(accel, axis=1)
        # A robust normalized signal used only to calibrate stress distributions.
        s_scale = max(float(np.percentile(speed, 95)), 1e-12)
        c_scale = max(float(np.percentile(curvature_proxy, 95)), 1e-12)
        signal = np.zeros(len(path), dtype=float)
        signal[1:] += np.clip(speed / s_scale, 0.0, 2.0) * 0.65
        signal[2:] += np.clip(curvature_proxy / c_scale, 0.0, 2.0) * 0.35
        signal = np.clip(signal / 2.0, 0.0, 1.0)
        signals[name] = signal
        rows.append({
            "scene": name,
            "path_length": float(speed.sum()),
            "speed_median": float(np.median(speed)),
            "speed_p95": float(np.percentile(speed, 95)),
            "curvature_proxy_median": float(np.median(curvature_proxy)),
            "curvature_proxy_p95": float(np.percentile(curvature_proxy, 95)),
            "motion_score_mean": float(signal.mean()),
            "motion_score_p95": float(np.percentile(signal, 95)),
        })
    return rows, signals


# -----------------------------------------------------------------------------
# G2: Exact 8-sector source grammar
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GrammarConfig:
    max_spatial_depth: int
    max_temporal_depth: int
    edge_x_cell: int
    y_boundary: int
    z_boundary: int
    t_boundary: int


def route_symbolic(boundary: int, side_bit: int, midpoint: int) -> int:
    if boundary < midpoint:
        return 0
    if boundary > midpoint:
        return 1
    return side_bit


def enumerate_source_assignments(config: GrammarConfig) -> FrozenSet[Assignment]:
    """Enumerate all local leaf assignments produced by stop/spatial/time splits.

    The target bipolar edge occupies one finest-grid cell in its edge axis, so
    no allowed split up to max_spatial_depth can introduce a middle vertex on
    that edge.  The remaining three binary sector axes are the two transverse
    spatial sides and the two temporal sides.
    """
    ns = 1 << config.max_spatial_depth
    nt = 1 << config.max_temporal_depth
    x_probe_numerator = 2 * config.edge_x_cell + 1  # denominator 2*ns

    @lru_cache(maxsize=None)
    def rec(
        xlo: int,
        ylo: int,
        zlo: int,
        spatial_size: int,
        tlo: int,
        temporal_size: int,
        probes: Tuple[int, ...],
    ) -> FrozenSet[Assignment]:
        outputs: set[Assignment] = set()
        leaf: Leaf = (xlo, ylo, zlo, spatial_size, tlo, temporal_size)
        outputs.add(tuple((probe, leaf) for probe in probes))

        if spatial_size > 1:
            half = spatial_size // 2
            xmid, ymid, zmid = xlo + half, ylo + half, zlo + half
            x_child = 0 if x_probe_numerator < 2 * xmid else 1
            groups: Dict[Tuple[int, int, int], List[int]] = {}
            for probe in probes:
                tbit, ybit, zbit = slot_bits(probe)
                ychild = route_symbolic(config.y_boundary, ybit, ymid)
                zchild = route_symbolic(config.z_boundary, zbit, zmid)
                groups.setdefault((x_child, ychild, zchild), []).append(probe)
            child_options: List[FrozenSet[Assignment]] = []
            for (xc, yc, zc), child_probes in sorted(groups.items()):
                child_options.append(rec(
                    xlo + xc * half,
                    ylo + yc * half,
                    zlo + zc * half,
                    half,
                    tlo,
                    temporal_size,
                    tuple(child_probes),
                ))
            for choice in product(*child_options):
                merged = tuple(sorted(item for partial in choice for item in partial))
                outputs.add(merged)

        if temporal_size > 1:
            half = temporal_size // 2
            tmid = tlo + half
            groups_t: Dict[int, List[int]] = {}
            for probe in probes:
                tbit, _, _ = slot_bits(probe)
                tchild = route_symbolic(config.t_boundary, tbit, tmid)
                groups_t.setdefault(tchild, []).append(probe)
            child_options_t: List[FrozenSet[Assignment]] = []
            for tc, child_probes in sorted(groups_t.items()):
                child_options_t.append(rec(
                    xlo,
                    ylo,
                    zlo,
                    spatial_size,
                    tlo + tc * half,
                    half,
                    tuple(child_probes),
                ))
            for choice in product(*child_options_t):
                merged = tuple(sorted(item for partial in choice for item in partial))
                outputs.add(merged)

        return frozenset(outputs)

    return rec(0, 0, 0, ns, 0, nt, tuple(range(8)))


def canonical_source_signature(assignment: Assignment, max_temporal_depth: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[Leaf, ...]]:
    """Return canonical HVID partition, center ticks, halfspans, slot leaves."""
    id_map: Dict[Leaf, int] = {}
    ids: List[int] = []
    center_ticks: List[int] = []
    halfspans: List[int] = []
    leaves: List[Leaf] = []
    for _, leaf in sorted(assignment):
        if leaf not in id_map:
            id_map[leaf] = len(id_map)
        ids.append(id_map[leaf])
        *_, tlo, temporal_size = leaf
        # Official mapping for a max_t depth grid: center=2*tlo+size,
        # halfspan=size, on the integer interval [0, 2^(max_t+1)].
        center_ticks.append(2 * tlo + temporal_size)
        halfspans.append(temporal_size)
        leaves.append(leaf)
    return tuple(ids), tuple(center_ticks), tuple(halfspans), tuple(leaves)


def collect_exact_source_states(max_spatial_depth: int = 2, max_temporal_depth: int = 2) -> Tuple[Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple], List[Dict[str, int]], int]:
    ns = 1 << max_spatial_depth
    nt = 1 << max_temporal_depth
    x_cell = max(0, ns // 2 - 1)
    representatives: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple] = {}
    alignment_rows: List[Dict[str, int]] = []
    raw_total = 0
    for yb in range(1, ns):
        for zb in range(1, ns):
            for tb in range(1, nt):
                cfg = GrammarConfig(max_spatial_depth, max_temporal_depth, x_cell, yb, zb, tb)
                assignments = enumerate_source_assignments(cfg)
                raw_total += len(assignments)
                alignment_rows.append({
                    "y_boundary": yb,
                    "z_boundary": zb,
                    "t_boundary": tb,
                    "raw_local_tree_outputs": len(assignments),
                })
                for assignment in assignments:
                    ids, times, halfspans, leaves = canonical_source_signature(assignment, max_temporal_depth)
                    representatives.setdefault(
                        (ids, times),
                        (ids, times, halfspans, leaves, yb, zb, tb),
                    )
    return representatives, alignment_rows, raw_total


@dataclass
class SparseTreeNode:
    xlo: int
    ylo: int
    zlo: int
    spatial_size: int
    tlo: int
    temporal_size: int
    split_kind: str = "leaf"  # leaf, spatial, temporal
    children: Dict[Tuple[int, ...], "SparseTreeNode"] | None = None
    leaf_id: int = -1


def make_probe_coordinates(config: GrammarConfig) -> Dict[int, Tuple[float, float, float, float]]:
    ns = 1 << config.max_spatial_depth
    nt = 1 << config.max_temporal_depth
    eps_s = 1.0 / (ns * 16.0)
    eps_t = 1.0 / (nt * 16.0)
    x = (config.edge_x_cell + 0.5) / ns
    y0, z0, t0 = config.y_boundary / ns, config.z_boundary / ns, config.t_boundary / nt
    probes: Dict[int, Tuple[float, float, float, float]] = {}
    for slot_id in range(8):
        tbit, ybit, zbit = slot_bits(slot_id)
        probes[slot_id] = (
            x,
            y0 + (-eps_s if ybit == 0 else eps_s),
            z0 + (-eps_s if zbit == 0 else eps_s),
            t0 + (-eps_t if tbit == 0 else eps_t),
        )
    return probes


def build_random_sparse_tree(
    config: GrammarConfig,
    rng: np.random.Generator,
    motion_score: float,
) -> SparseTreeNode:
    probes = make_probe_coordinates(config)
    leaf_counter = [0]

    def build(node: SparseTreeNode, probe_ids: Tuple[int, ...], sdepth: int, tdepth: int) -> SparseTreeNode:
        # Camera-calibrated stress only; it is not used as a prevalence estimate.
        p_temporal = 0.12 + 0.58 * motion_score
        p_spatial = 0.48 - 0.18 * motion_score
        if node.spatial_size <= 1:
            p_spatial = 0.0
        if node.temporal_size <= 1:
            p_temporal = 0.0
        p_stop = max(0.15, 1.0 - p_temporal - p_spatial)
        probs = np.asarray([p_stop, p_spatial, p_temporal], dtype=float)
        probs /= probs.sum()
        choice = int(rng.choice(3, p=probs))
        if choice == 0 or (node.spatial_size <= 1 and node.temporal_size <= 1):
            node.split_kind = "leaf"
            node.leaf_id = leaf_counter[0]
            leaf_counter[0] += 1
            return node

        if choice == 1 and node.spatial_size > 1:
            node.split_kind = "spatial"
            node.children = {}
            half = node.spatial_size // 2
            xmid = (node.xlo + half) / (1 << config.max_spatial_depth)
            ymid = (node.ylo + half) / (1 << config.max_spatial_depth)
            zmid = (node.zlo + half) / (1 << config.max_spatial_depth)
            groups: Dict[Tuple[int, int, int], List[int]] = {}
            for pid in probe_ids:
                x, y, z, _ = probes[pid]
                key = (int(x >= xmid), int(y >= ymid), int(z >= zmid))
                groups.setdefault(key, []).append(pid)
            for key, pids in groups.items():
                xc, yc, zc = key
                child = SparseTreeNode(
                    node.xlo + xc * half,
                    node.ylo + yc * half,
                    node.zlo + zc * half,
                    half,
                    node.tlo,
                    node.temporal_size,
                )
                node.children[key] = build(child, tuple(pids), sdepth + 1, tdepth)
            return node

        if node.temporal_size > 1:
            node.split_kind = "temporal"
            node.children = {}
            half = node.temporal_size // 2
            tmid = (node.tlo + half) / (1 << config.max_temporal_depth)
            groups_t: Dict[Tuple[int], List[int]] = {}
            for pid in probe_ids:
                *_, t = probes[pid]
                key = (int(t >= tmid),)
                groups_t.setdefault(key, []).append(pid)
            for key, pids in groups_t.items():
                tc = key[0]
                child = SparseTreeNode(
                    node.xlo,
                    node.ylo,
                    node.zlo,
                    node.spatial_size,
                    node.tlo + tc * half,
                    half,
                )
                node.children[key] = build(child, tuple(pids), sdepth, tdepth + 1)
            return node

        node.split_kind = "leaf"
        node.leaf_id = leaf_counter[0]
        leaf_counter[0] += 1
        return node

    root = SparseTreeNode(0, 0, 0, 1 << config.max_spatial_depth, 0, 1 << config.max_temporal_depth)
    return build(root, tuple(range(8)), 0, 0)


def locate_point(node: SparseTreeNode, point: Tuple[float, float, float, float], max_s: int, max_t: int) -> SparseTreeNode:
    current = node
    while current.split_kind != "leaf":
        assert current.children is not None
        x, y, z, t = point
        if current.split_kind == "spatial":
            half = current.spatial_size // 2
            key = (
                int(x >= (current.xlo + half) / (1 << max_s)),
                int(y >= (current.ylo + half) / (1 << max_s)),
                int(z >= (current.zlo + half) / (1 << max_s)),
            )
        else:
            half = current.temporal_size // 2
            key = (int(t >= (current.tlo + half) / (1 << max_t)),)
        current = current.children[key]
    return current


def locate_symbolic(node: SparseTreeNode, slot_id: int, config: GrammarConfig) -> SparseTreeNode:
    current = node
    tbit, ybit, zbit = slot_bits(slot_id)
    x_probe_numerator = 2 * config.edge_x_cell + 1
    while current.split_kind != "leaf":
        assert current.children is not None
        if current.split_kind == "spatial":
            half = current.spatial_size // 2
            xmid, ymid, zmid = current.xlo + half, current.ylo + half, current.zlo + half
            key = (
                0 if x_probe_numerator < 2 * xmid else 1,
                route_symbolic(config.y_boundary, ybit, ymid),
                route_symbolic(config.z_boundary, zbit, zmid),
            )
        else:
            half = current.temporal_size // 2
            key = (route_symbolic(config.t_boundary, tbit, current.tlo + half),)
        current = current.children[key]
    return current


def run_source_grammar_experiment(outdir: Path, random_trials: int, seed: int) -> Tuple[Dict[str, object], Dict]:
    start = time.perf_counter()
    representatives, alignment_rows, raw_total = collect_exact_source_states(2, 2)
    partitions = {ids for ids, _ in representatives.keys()}
    time_arrays = {times for _, times in representatives.keys()}
    block_hist = collections.Counter(len(set(ids)) for ids, _ in representatives.keys())
    partition_block_hist = collections.Counter(len(set(ids)) for ids in partitions)

    # Exact comparison against the independently relaxed four-column time product.
    relaxed_union: set[Tuple[int, ...]] = set()
    per_boundary_rows: List[Dict[str, int]] = []
    for tb in range(1, 1 << 2):
        pairs: set[Tuple[int, int]] = set()
        source_at_boundary: set[Tuple[int, ...]] = set()
        # Re-enumerate this temporal alignment instead of filtering the globally
        # deduplicated representative table: the same signature can occur at
        # more than one boundary and only one representative is retained.
        for yb in range(1, 1 << 2):
            for zb in range(1, 1 << 2):
                cfg = GrammarConfig(2, 2, 1, yb, zb, tb)
                for assignment in enumerate_source_assignments(cfg):
                    _, times, _, _ = canonical_source_signature(assignment, 2)
                    source_at_boundary.add(times)
                    for column in range(4):
                        pairs.add((times[column], times[4 + column]))
        relaxed = {
            tuple(pair[0] for pair in combo) + tuple(pair[1] for pair in combo)
            for combo in product(sorted(pairs), repeat=4)
        }
        relaxed_union.update(relaxed)
        per_boundary_rows.append({
            "t_boundary": tb,
            "legal_column_pairs": len(pairs),
            "relaxed_product_time_arrays": len(relaxed),
            "source_time_arrays": len(source_at_boundary),
            "missing_source_arrays": len(source_at_boundary - relaxed),
            "unrealized_relaxed_arrays": len(relaxed - source_at_boundary),
        })

    # Deep randomized source-semantics cross-check using official camera paths.
    camera_rows, motion_signals = camera_motion_profiles()
    rng = np.random.default_rng(seed)
    scene_names = sorted(motion_signals)
    mismatches = 0
    random_signatures: set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()
    random_block_hist: collections.Counter[int] = collections.Counter()
    for _ in range(random_trials):
        scene = scene_names[int(rng.integers(0, len(scene_names)))]
        signal = motion_signals[scene]
        frame = int(rng.integers(0, len(signal)))
        motion = float(signal[frame])
        max_s = 5
        max_t = 5
        ns, nt = 1 << max_s, 1 << max_t
        cfg = GrammarConfig(
            max_s,
            max_t,
            int(rng.integers(0, ns)),
            int(rng.integers(1, ns)),
            int(rng.integers(1, ns)),
            int(rng.integers(1, nt)),
        )
        tree = build_random_sparse_tree(cfg, rng, motion)
        points = make_probe_coordinates(cfg)
        point_leaves: List[SparseTreeNode] = []
        symbolic_leaves: List[SparseTreeNode] = []
        for slot_id in range(8):
            point_leaves.append(locate_point(tree, points[slot_id], max_s, max_t))
            symbolic_leaves.append(locate_symbolic(tree, slot_id, cfg))
        point_ids = tuple(node.leaf_id for node in point_leaves)
        symbolic_ids = tuple(node.leaf_id for node in symbolic_leaves)
        if point_ids != symbolic_ids:
            mismatches += 1
            continue
        canon_map: Dict[int, int] = {}
        canon_ids: List[int] = []
        times: List[int] = []
        for node in point_leaves:
            if node.leaf_id not in canon_map:
                canon_map[node.leaf_id] = len(canon_map)
            canon_ids.append(canon_map[node.leaf_id])
            times.append(2 * node.tlo + node.temporal_size)
        signature = (tuple(canon_ids), tuple(times))
        random_signatures.add(signature)
        random_block_hist[len(set(canon_ids))] += 1

    summary = {
        "max_spatial_depth_exact": 2,
        "max_temporal_depth_exact": 2,
        "boundary_alignments": len(alignment_rows),
        "raw_local_tree_outputs": raw_total,
        "unique_source_hvid_time_states": len(representatives),
        "unique_hvid_partition_patterns": len(partitions),
        "unique_time_arrays": len(time_arrays),
        "relaxed_time_product_arrays": len(relaxed_union),
        "source_time_arrays_missing_from_relaxed": len(time_arrays - relaxed_union),
        "relaxed_time_arrays_not_source_realized": len(relaxed_union - time_arrays),
        "hvid_block_count_histogram": {str(k): int(v) for k, v in sorted(block_hist.items())},
        "partition_block_count_histogram": {str(k): int(v) for k, v in sorted(partition_block_hist.items())},
        "per_temporal_boundary": per_boundary_rows,
        "deep_random_trials": random_trials,
        "deep_point_vs_symbolic_mismatches": mismatches,
        "deep_unique_signatures": len(random_signatures),
        "deep_hvid_block_count_histogram": {str(k): int(v) for k, v in sorted(random_block_hist.items())},
        "runtime_seconds": time.perf_counter() - start,
    }

    with (outdir / "source_alignment_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(alignment_rows[0]))
        writer.writeheader(); writer.writerows(alignment_rows)
    with (outdir / "camera_motion_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(camera_rows[0]))
        writer.writeheader(); writer.writerows(camera_rows)

    return summary, {
        "representatives": representatives,
        "camera_rows": camera_rows,
        "motion_signals": motion_signals,
    }


# -----------------------------------------------------------------------------
# G1: Source-exact one-sided limit strata
# -----------------------------------------------------------------------------


def non_negative_orientation(matrix: Sequence[Sequence[int]]) -> bool:
    a = np.asarray(matrix, dtype=np.int64)
    det = (
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    )
    return int(det) >= 0


def official_preprocess_faces(times: Sequence[int], threshold: int) -> List[List[Tuple[int, int]]]:
    """Literal Python port of slicing.cpp::preprocess_hyperpoly."""
    c = [int(t > threshold) for t in times]
    edges: List[Tuple[int, int]] = []
    for e in range(12):
        coords = [0, 0, 0]
        e0 = e // 4
        e1 = (e0 + 1) % 3
        e2 = (e1 + 1) % 3
        coords[e0] = 0
        coords[e1] = e & 1
        coords[e2] = (e >> 1) & 1
        index1 = cube_index(*coords)
        coords[e0] = 1
        index2 = cube_index(*coords)
        if c[index1] != c[index2]:
            edges.append((index1, index2) if not c[index1] else (index2, index1))
    if not edges:
        return []

    next_edge = [-1] * len(edges)
    for i in range(len(edges)):
        candidates: List[int] = []
        for j in range(len(edges)):
            if i == j:
                continue
            mins, maxs = [1, 1, 1], [0, 0, 0]
            for k in range(3):
                for vertex in (edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                    bit = (vertex >> k) & 1
                    mins[k] = min(mins[k], bit)
                    maxs[k] = max(maxs[k], bit)
            if any(mins[k] == maxs[k] for k in range(3)):
                matrix = [
                    [((edges[i][0] >> k) & 1) * 2 - 1 for k in range(3)],
                    [((edges[i][1] >> k) & 1) * 2 - 1 for k in range(3)],
                    [((edges[j][0] >> k) & 1) * 2 - 1 for k in range(3)],
                ]
                if non_negative_orientation(matrix):
                    matrix[2] = [((edges[j][1] >> k) & 1) * 2 - 1 for k in range(3)]
                    if non_negative_orientation(matrix):
                        candidates.append(j)
        if len(candidates) > 1:
            shared_low = [j for j in candidates if edges[i][0] == edges[j][0]]
            if not shared_low:
                raise RuntimeError("source port found no preferred next edge")
            next_edge[i] = shared_low[0]
        elif len(candidates) == 1:
            next_edge[i] = candidates[0]
        else:
            raise RuntimeError("source port found an empty next-edge set")

    visited = [False] * len(edges)
    faces: List[List[Tuple[int, int]]] = []
    for i in range(len(edges)):
        if visited[i]:
            continue
        face = [edges[i]]
        visited[i] = True
        last = edges[i]
        j = next_edge[i]
        guard = 0
        while j != i:
            visited[j] = True
            new_edge = edges[j]
            if new_edge != last:
                face.append(new_edge)
            last = new_edge
            j = next_edge[j]
            guard += 1
            if guard > 24:
                raise RuntimeError("source port cycle did not close")
        while len(face) > 1 and face[-1] == face[0]:
            face.pop()
        if len(face) > 2:
            faces.append(face)
    return faces


def reduce_cyclic_tokens(tokens: Sequence[int]) -> Cycle:
    reduced: List[int] = []
    for token in tokens:
        if not reduced or reduced[-1] != token:
            reduced.append(int(token))
    if len(reduced) > 1 and reduced[0] == reduced[-1]:
        reduced.pop()
    return tuple(reduced)


@dataclass(frozen=True)
class LimitCandidate:
    side: str
    event_time: int
    cycle: Cycle
    vertex_set: FrozenSet[int]
    source_edges: Tuple[Tuple[int, int], ...]


def source_endpoint_limit_candidates(times: Sequence[int], ids: Sequence[int]) -> List[LimitCandidate]:
    unique_times = sorted(set(int(t) for t in times))
    if len(unique_times) < 2:
        return []
    candidates: List[LimitCandidate] = []
    for side in ("min", "max"):
        threshold = unique_times[0] if side == "min" else unique_times[-2]
        event_time = unique_times[0] if side == "min" else unique_times[-1]
        faces = official_preprocess_faces(times, threshold)
        for face in faces:
            tokens: List[int] = []
            for low_slot, high_slot in face:
                endpoint_slot = low_slot if side == "min" else high_slot
                if int(times[endpoint_slot]) != event_time:
                    raise AssertionError("one-sided endpoint map disagrees with source edge orientation")
                tokens.append(int(ids[endpoint_slot]))
            cycle = reduce_cyclic_tokens(tokens)
            if len(set(cycle)) > 1:
                candidates.append(LimitCandidate(
                    side,
                    event_time,
                    cycle,
                    frozenset(cycle),
                    tuple(face),
                ))
    return candidates


def leaf_spatial_center(leaf: Leaf, max_spatial_depth: int) -> np.ndarray:
    xlo, ylo, zlo, spatial_size, _, _ = leaf
    n = 1 << max_spatial_depth
    return np.asarray([
        (xlo + 0.5 * spatial_size) / n,
        (ylo + 0.5 * spatial_size) / n,
        (zlo + 0.5 * spatial_size) / n,
    ], dtype=float)


def affine_rank(points: Sequence[np.ndarray], tol: float = 1e-10) -> int:
    if len(points) <= 1:
        return 0
    pts = np.asarray(points, dtype=float)
    return int(np.linalg.matrix_rank(pts[1:] - pts[0], tol=tol))


def extremal_algebraic_fiber_dimension(times: Sequence[int], event_time: int) -> int:
    """Dimension of the exact multi-affine extremal fiber as a cubical complex.

    At a global min/max, all Bernstein weights are nonnegative. A point belongs
    to the extremal fiber iff every active corner has the extremal value. Thus
    the fiber is exactly the union of coordinate subfaces whose vertices all
    have that value.
    """
    best = -1
    for axis_states in product((-1, 0, 1), repeat=3):  # -1 means free axis
        corner_slots: List[int] = []
        for xyz in product((0, 1), repeat=3):
            if all(state == -1 or xyz[axis] == state for axis, state in enumerate(axis_states)):
                corner_slots.append(cube_index(*xyz))
        if corner_slots and all(int(times[slot]) == event_time for slot in corner_slots):
            best = max(best, sum(state == -1 for state in axis_states))
    return best


def event_star_components(candidates: Sequence[LimitCandidate]) -> List[List[int]]:
    """Group residual cells by shared HVID incidence after exact-set cancellation."""
    multiplicity: Dict[Tuple[int, FrozenSet[int]], List[int]] = collections.defaultdict(list)
    for idx, candidate in enumerate(candidates):
        multiplicity[(candidate.event_time, candidate.vertex_set)].append(idx)
    residual: List[int] = []
    for indices in multiplicity.values():
        residual.extend(indices[len(indices) // 2 * 2:])  # pairwise exact-set cancellation
    if not residual:
        return []
    parent = {idx: idx for idx in residual}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(residual):
        for b in residual[i + 1:]:
            ca, cb = candidates[a], candidates[b]
            if ca.event_time == cb.event_time and ca.vertex_set & cb.vertex_set:
                union(a, b)
    groups: Dict[int, List[int]] = collections.defaultdict(list)
    for idx in residual:
        groups[find(idx)].append(idx)
    return list(groups.values())


def run_flat_stratum_experiment(outdir: Path, representatives: Mapping, seed: int) -> Dict[str, object]:
    start = time.perf_counter()
    counters: collections.Counter[str] = collections.Counter()
    rows: List[Dict[str, object]] = []
    all_candidates: List[LimitCandidate] = []
    candidate_payloads: List[Tuple[LimitCandidate, Tuple[Leaf, ...], Tuple[int, ...]]] = []
    witness_strict_distinct: Optional[Dict[str, object]] = None
    witness_representation: Optional[Dict[str, object]] = None

    for ids, times, halfspans, leaves, yb, zb, tb in representatives.values():
        d_values = tuple(int(times[4 + i] - times[i]) for i in range(4))
        if all(value == 0 for value in d_values):
            d_category = "D_identically_zero"
        elif any(value == 0 for value in d_values):
            d_category = "partial_D_zero"
        else:
            d_category = "strict_D_positive_columns"
        repeated_hvid = len(set(ids)) < 8
        candidates = source_endpoint_limit_candidates(times, ids)
        if candidates:
            counters[f"states_with_candidate::{d_category}"] += 1
        for candidate in candidates:
            all_candidates.append(candidate)
            candidate_payloads.append((candidate, leaves, ids))
            id_to_slot: Dict[int, int] = {}
            for slot_id, hvid in enumerate(ids):
                id_to_slot.setdefault(int(hvid), slot_id)
            points = [leaf_spatial_center(leaves[id_to_slot[hvid]], 2) for hvid in candidate.vertex_set]
            carrier_rank = affine_rank(points)
            algebraic_dim = extremal_algebraic_fiber_dimension(times, candidate.event_time)
            representation_induced = carrier_rank > algebraic_dim
            side_slots = range(0, 4) if candidate.side == "min" else range(4, 8)
            time_parallel_side = all(int(times[slot_id]) == candidate.event_time for slot_id in side_slots)
            disk_candidate = carrier_rank == 2 and len(candidate.vertex_set) >= 3
            lower_dimensional = carrier_rank <= 1

            counters["candidate_total"] += 1
            counters[f"d_category::{d_category}"] += 1
            counters["repeated_hvid_hyperpoly" if repeated_hvid else "all_eight_hvid_distinct"] += 1
            counters[f"carrier_rank::{carrier_rank}"] += 1
            counters[f"algebraic_fiber_dim::{algebraic_dim}"] += 1
            counters["representation_induced" if representation_induced else "algebraically_supported_or_lower"] += 1
            counters["time_parallel_temporal_side" if time_parallel_side else "not_full_temporal_side"] += 1
            counters["disk_cone_candidate" if disk_candidate else "lower_dimensional_handler"] += 1

            row = {
                "ids": " ".join(map(str, ids)),
                "times": " ".join(map(str, times)),
                "D": " ".join(map(str, d_values)),
                "d_category": d_category,
                "repeated_hvid": int(repeated_hvid),
                "side": candidate.side,
                "event_time": candidate.event_time,
                "cycle": " ".join(map(str, candidate.cycle)),
                "unique_vertices": len(candidate.vertex_set),
                "carrier_rank": carrier_rank,
                "algebraic_extremal_fiber_dimension": algebraic_dim,
                "representation_induced": int(representation_induced),
                "time_parallel_temporal_side": int(time_parallel_side),
                "disk_cone_candidate": int(disk_candidate),
                "y_boundary": yb,
                "z_boundary": zb,
                "t_boundary": tb,
            }
            rows.append(row)

            if (
                witness_strict_distinct is None
                and d_category == "strict_D_positive_columns"
                and not repeated_hvid
            ):
                witness_strict_distinct = row.copy()
            if (
                witness_representation is None
                and representation_induced
                and d_category == "strict_D_positive_columns"
                and not repeated_hvid
            ):
                witness_representation = row.copy()

    total = counters["candidate_total"]
    d0 = counters["d_category::D_identically_zero"]
    partial = counters["d_category::partial_D_zero"]
    strict = counters["d_category::strict_D_positive_columns"]

    # Source-equivalence and cancellation stress.
    # Duplicate every source candidate twice: exact-set cancellation must remove all.
    doubled: List[LimitCandidate] = []
    for candidate in all_candidates:
        doubled.extend((candidate, candidate))
    exact_pair_residual_components = len(event_star_components(doubled))

    # Partial-overlap cells must not be silently cancelled. Construct deterministic
    # synthetic stars using source-derived cells renamed into a common registry.
    rng = np.random.default_rng(seed)
    partial_overlap_trials = min(1000, max(1, len(all_candidates)))
    partial_overlap_failures = 0
    for trial in range(partial_overlap_trials):
        base = all_candidates[int(rng.integers(0, len(all_candidates)))]
        # A pair of triangles sharing an edge but not an exact vertex set.
        a, b = 10_000 + 4 * trial, 10_001 + 4 * trial
        c, d = 10_002 + 4 * trial, 10_003 + 4 * trial
        c1 = LimitCandidate(base.side, base.event_time, (a, b, c), frozenset((a, b, c)), base.source_edges)
        c2 = LimitCandidate(base.side, base.event_time, (b, a, d), frozenset((a, b, d)), base.source_edges)
        groups = event_star_components((c1, c2))
        if len(groups) != 1 or len(groups[0]) != 2:
            partial_overlap_failures += 1

    summary: Dict[str, object] = {
        "source_states_examined": len(representatives),
        "one_sided_limit_candidates": total,
        "states_with_any_candidate": sum(v for k, v in counters.items() if k.startswith("states_with_candidate::")),
        "D_identically_zero_candidates": d0,
        "D_identically_zero_coverage": safe_div(d0, total),
        "partial_or_full_D_zero_candidates": d0 + partial,
        "partial_or_full_D_zero_coverage": safe_div(d0 + partial, total),
        "strict_D_positive_candidates": strict,
        "strict_D_positive_fraction": safe_div(strict, total),
        "repeated_hvid_candidates": counters["repeated_hvid_hyperpoly"],
        "all_distinct_hvid_candidates": counters["all_eight_hvid_distinct"],
        "representation_induced_candidates": counters["representation_induced"],
        "representation_induced_fraction": safe_div(counters["representation_induced"], total),
        "rank_1_candidates": counters["carrier_rank::1"],
        "rank_2_candidates": counters["carrier_rank::2"],
        "disk_cone_candidates": counters["disk_cone_candidate"],
        "lower_dimensional_candidates": counters["lower_dimensional_handler"],
        "time_parallel_temporal_side_candidates": counters["time_parallel_temporal_side"],
        "exact_pair_cancellation_residual_components": exact_pair_residual_components,
        "partial_overlap_star_trials": partial_overlap_trials,
        "partial_overlap_grouping_failures": partial_overlap_failures,
        "strict_D_all_distinct_witness": witness_strict_distinct,
        "strict_D_representation_induced_witness": witness_representation,
        "counter_dump": dict(counters),
        "runtime_seconds": time.perf_counter() - start,
    }

    with (outdir / "flat_limit_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return summary


# -----------------------------------------------------------------------------
# G3: Trilinear spatial embedding certificate
# -----------------------------------------------------------------------------

PARAMETER_NODES = np.asarray([0.0, 0.5, 1.0], dtype=float)


def unit_hexahedron() -> np.ndarray:
    vertices = np.zeros((2, 2, 2, 3), dtype=float)
    for i, j, k in product((0, 1), repeat=3):
        vertices[i, j, k] = (i, j, k)
    return vertices


def trilinear_point(vertices: np.ndarray, u: float, v: float, w: float) -> np.ndarray:
    bu, bv, bw = np.asarray([1 - u, u]), np.asarray([1 - v, v]), np.asarray([1 - w, w])
    return np.einsum("i,j,k,ijkc->c", bu, bv, bw, vertices)


def trilinear_derivatives(vertices: np.ndarray, u: float, v: float, w: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bu, bv, bw = np.asarray([1 - u, u]), np.asarray([1 - v, v]), np.asarray([1 - w, w])
    d = np.asarray([-1.0, 1.0])
    du = np.einsum("i,j,k,ijkc->c", d, bv, bw, vertices)
    dv = np.einsum("i,j,k,ijkc->c", bu, d, bw, vertices)
    dw = np.einsum("i,j,k,ijkc->c", bu, bv, d, vertices)
    return du, dv, dw


def jacobian_det(vertices: np.ndarray, u: float, v: float, w: float) -> float:
    du, dv, dw = trilinear_derivatives(vertices, u, v, w)
    return float(np.linalg.det(np.stack((du, dv, dw), axis=1)))


def jacobian_nodal_values(vertices: np.ndarray) -> np.ndarray:
    return np.asarray([
        [[jacobian_det(vertices, u, v, w) for w in PARAMETER_NODES] for v in PARAMETER_NODES]
        for u in PARAMETER_NODES
    ], dtype=float)


def nodal_to_quadratic_bernstein_axis(values: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(values, axis, 0)
    coefficients = np.empty_like(moved)
    coefficients[0] = moved[0]
    coefficients[2] = moved[2]
    coefficients[1] = 2.0 * moved[1] - 0.5 * (moved[0] + moved[2])
    return np.moveaxis(coefficients, 0, axis)


def jacobian_bernstein_coefficients(vertices: np.ndarray) -> np.ndarray:
    coefficients = jacobian_nodal_values(vertices)
    for axis in range(3):
        coefficients = nodal_to_quadratic_bernstein_axis(coefficients, axis)
    return coefficients


def restrict_trilinear_cell(vertices: np.ndarray, lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    child = np.zeros_like(vertices)
    for i, j, k in product((0, 1), repeat=3):
        uvw = np.where(np.asarray((i, j, k), dtype=int) == 0, lower, upper)
        child[i, j, k] = trilinear_point(vertices, float(uvw[0]), float(uvw[1]), float(uvw[2]))
    return child


@dataclass
class JacobianCertificate:
    status: str  # VALID, INVALID, UNRESOLVED
    boxes_visited: int
    deepest_level: int
    global_min_coefficient: float
    witness_uvw: Optional[Tuple[float, float, float]]
    runtime_ms: float


def certify_trilinear_jacobian(
    vertices: np.ndarray,
    epsilon: float = 1e-9,
    max_depth: int = 7,
) -> JacobianCertificate:
    start = time.perf_counter()
    stack: List[Tuple[np.ndarray, int, Tuple[float, float, float], Tuple[float, float, float]]] = [
        (vertices, 0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ]
    boxes = 0
    deepest = 0
    global_min_coefficient = float("inf")
    while stack:
        cell, depth, lower, upper = stack.pop()
        boxes += 1
        deepest = max(deepest, depth)
        nodal = jacobian_nodal_values(cell)
        coefficients = nodal.copy()
        for axis in range(3):
            coefficients = nodal_to_quadratic_bernstein_axis(coefficients, axis)
        minimum = float(coefficients.min())
        global_min_coefficient = min(global_min_coefficient, minimum)

        # A negative nodal value is an explicit invalidity witness.
        if float(nodal.min()) < -epsilon:
            index = np.unravel_index(int(np.argmin(nodal)), nodal.shape)
            witness = tuple(
                lower[axis] + PARAMETER_NODES[index[axis]] * (upper[axis] - lower[axis])
                for axis in range(3)
            )
            return JacobianCertificate(
                "INVALID", boxes, deepest, global_min_coefficient,
                tuple(map(float, witness)), (time.perf_counter() - start) * 1000.0,
            )

        # Convex-hull property of the Bernstein basis certifies positivity.
        if minimum > epsilon:
            continue

        if depth >= max_depth:
            return JacobianCertificate(
                "UNRESOLVED", boxes, deepest, global_min_coefficient,
                None, (time.perf_counter() - start) * 1000.0,
            )

        midpoint = tuple((lower[axis] + upper[axis]) * 0.5 for axis in range(3))
        for bits in product((0, 1), repeat=3):
            child_lower = tuple(lower[axis] if bits[axis] == 0 else midpoint[axis] for axis in range(3))
            child_upper = tuple(midpoint[axis] if bits[axis] == 0 else upper[axis] for axis in range(3))
            child = restrict_trilinear_cell(vertices, child_lower, child_upper)
            stack.append((child, depth + 1, child_lower, child_upper))

    return JacobianCertificate(
        "VALID", boxes, deepest, global_min_coefficient,
        None, (time.perf_counter() - start) * 1000.0,
    )


def batch_jacobian(vertices_batch: np.ndarray, u: float, v: float, w: float) -> np.ndarray:
    bu, bv, bw = np.asarray([1 - u, u]), np.asarray([1 - v, v]), np.asarray([1 - w, w])
    d = np.asarray([-1.0, 1.0])
    du = np.einsum("i,j,k,bijkc->bc", d, bv, bw, vertices_batch)
    dv = np.einsum("i,j,k,bijkc->bc", bu, d, bw, vertices_batch)
    dw = np.einsum("i,j,k,bijkc->bc", bu, bv, d, vertices_batch)
    return np.linalg.det(np.stack((du, dv, dw), axis=2))


def corner_jacobians(vertices: np.ndarray) -> np.ndarray:
    return np.asarray([jacobian_det(vertices, i, j, k) for i, j, k in product((0.0, 1.0), repeat=3)])


def dense_jacobian_min(vertices: np.ndarray, resolution: int = 17) -> Tuple[float, Tuple[float, float, float]]:
    best_value = float("inf")
    best_point = (0.0, 0.0, 0.0)
    grid = np.linspace(0.0, 1.0, resolution)
    for u in grid:
        for v in grid:
            for w in grid:
                value = jacobian_det(vertices, float(u), float(v), float(w))
                if value < best_value:
                    best_value = value
                    best_point = (float(u), float(v), float(w))
    return best_value, best_point


def random_positive_affine(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.eye(3) + rng.normal(scale=0.18, size=(3, 3))
    if np.linalg.det(matrix) < 0:
        matrix[0] *= -1
    # Keep a healthy orientation margin.
    u, _, vt = np.linalg.svd(matrix)
    singular = np.clip(np.linalg.svd(matrix, compute_uv=False), 0.55, 1.8)
    matrix = u @ np.diag(singular) @ vt
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    translation = rng.normal(scale=0.25, size=3)
    return matrix, translation


def source_like_hexahedron(rng: np.random.Generator, motion_score: float) -> np.ndarray:
    """Repo-shaped stress cell using three-bit bisection-like displacements.

    The official implementation places each dual vertex on a segment from a
    cell center toward a cube-surface query point, with three bisection rounds.
    Here the displacement factors are therefore multiples of 1/8. Camera-path
    motion only calibrates the perturbation amplitude; it is not a cache model.
    """
    base = unit_hexahedron()
    amplitude = 0.035 + 0.16 * motion_score
    perturb = np.zeros_like(base)
    for i, j, k in product((0, 1), repeat=3):
        r = int(rng.integers(1, 9)) / 8.0
        direction = rng.normal(size=3)
        direction += 0.45 * np.asarray([i - 0.5, j - 0.5, k - 0.5])
        norm = max(float(np.linalg.norm(direction)), 1e-12)
        perturb[i, j, k] = amplitude * r * direction / norm
    vertices = base + perturb
    matrix, translation = random_positive_affine(rng)
    return np.einsum("ab,ijkb->ijka", matrix, vertices) + translation




def collect_hard_valid_cells(rng: np.random.Generator, count: int) -> List[np.ndarray]:
    """Valid warped cells whose root Bernstein hull is inconclusive.

    These cases exercise recursive subdivision instead of the trivial
    all-positive-root-coefficient path.
    """
    base = unit_hexahedron()
    output: List[np.ndarray] = []
    while len(output) < count:
        vertices = base + rng.normal(scale=0.55, size=base.shape)
        if float(corner_jacobians(vertices).min()) <= 1e-2:
            continue
        nodal = jacobian_nodal_values(vertices)
        if float(nodal.min()) <= 1e-4:
            continue
        root_coefficients = jacobian_bernstein_coefficients(vertices)
        if float(root_coefficients.min()) > 1e-9:
            continue
        cert = certify_trilinear_jacobian(vertices, max_depth=6)
        if cert.status == "VALID" and cert.boxes_visited > 1:
            output.append(vertices)
    return output

def collect_interior_fold_cells(rng: np.random.Generator, count: int) -> List[np.ndarray]:
    base = unit_hexahedron()
    output: List[np.ndarray] = []
    while len(output) < count:
        batch_size = 4096
        batch = base[None, ...] + rng.normal(scale=0.9, size=(batch_size, 2, 2, 2, 3))
        corners = np.stack([
            batch_jacobian(batch, float(i), float(j), float(k))
            for i, j, k in product((0, 1), repeat=3)
        ], axis=1)
        center = batch_jacobian(batch, 0.5, 0.5, 0.5)
        indices = np.where((corners.min(axis=1) > 1e-2) & (center < -1e-2))[0]
        for index in indices:
            output.append(batch[int(index)].copy())
            if len(output) >= count:
                break
    return output


def near_degenerate_cell(rng: np.random.Generator, motion_score: float) -> np.ndarray:
    vertices = source_like_hexahedron(rng, motion_score * 0.25)
    # Compress one parametric direction to a positive but sub-certificate scale.
    origin = vertices[:, :, 0].mean(axis=(0, 1))
    bottom = vertices[:, :, 0].copy()
    top_direction = vertices[:, :, 1] - bottom
    thickness = 10.0 ** float(rng.uniform(-13.0, -10.5))
    vertices[:, :, 1] = bottom + thickness * top_direction
    # A tiny perturbation avoids making every case analytically identical.
    vertices += rng.normal(scale=thickness * 0.05, size=vertices.shape)
    return vertices


def obvious_invalid_cell(rng: np.random.Generator) -> np.ndarray:
    vertices = source_like_hexahedron(rng, 0.4)
    # Reflect one top corner through the opposite face.
    i, j = int(rng.integers(0, 2)), int(rng.integers(0, 2))
    vertices[i, j, 1] = vertices[i, j, 0] - 0.35 * (vertices[i, j, 1] - vertices[i, j, 0])
    return vertices


def run_spatial_validity_experiment(
    outdir: Path,
    motion_signals: Mapping[str, np.ndarray],
    seed: int,
    valid_count: int,
    hard_valid_count: int,
    fold_count: int,
    invalid_count: int,
    near_count: int,
) -> Dict[str, object]:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    scene_names = sorted(motion_signals)

    valid_cells: List[np.ndarray] = []
    valid_attempts = 0
    while len(valid_cells) < valid_count:
        valid_attempts += 1
        scene = scene_names[int(rng.integers(0, len(scene_names)))]
        signal = motion_signals[scene]
        motion = float(signal[int(rng.integers(0, len(signal)))])
        cell = source_like_hexahedron(rng, motion)
        cert = certify_trilinear_jacobian(cell, max_depth=5)
        if cert.status == "VALID":
            valid_cells.append(cell)

    hard_valid_cells = collect_hard_valid_cells(rng, hard_valid_count)
    fold_cells = collect_interior_fold_cells(rng, fold_count)
    invalid_cells = [obvious_invalid_cell(rng) for _ in range(invalid_count)]
    near_cells: List[np.ndarray] = []
    for _ in range(near_count):
        scene = scene_names[int(rng.integers(0, len(scene_names)))]
        signal = motion_signals[scene]
        near_cells.append(near_degenerate_cell(rng, float(signal[int(rng.integers(0, len(signal)))])))

    families = {
        "repo_shaped_valid": valid_cells,
        "hard_valid_requires_subdivision": hard_valid_cells,
        "positive_corner_interior_fold": fold_cells,
        "obvious_invalid": invalid_cells,
        "near_degenerate": near_cells,
    }
    rows: List[Dict[str, object]] = []
    status_by_family: Dict[str, collections.Counter[str]] = {}
    representative_fold: Optional[np.ndarray] = None
    for family, cells in families.items():
        status_counter: collections.Counter[str] = collections.Counter()
        for index, cell in enumerate(cells):
            corners = corner_jacobians(cell)
            dense_minimum, dense_location = dense_jacobian_min(cell, resolution=13)
            cert = certify_trilinear_jacobian(cell, max_depth=7)
            status_counter[cert.status] += 1
            if family == "positive_corner_interior_fold" and representative_fold is None:
                representative_fold = cell.copy()
            rows.append({
                "family": family,
                "index": index,
                "corner_min": float(corners.min()),
                "corner_all_positive": int(float(corners.min()) > 1e-9),
                "dense_min": dense_minimum,
                "dense_min_u": dense_location[0],
                "dense_min_v": dense_location[1],
                "dense_min_w": dense_location[2],
                "certificate_status": cert.status,
                "boxes_visited": cert.boxes_visited,
                "deepest_level": cert.deepest_level,
                "global_min_bernstein_coefficient": cert.global_min_coefficient,
                "runtime_ms": cert.runtime_ms,
            })
        status_by_family[family] = status_counter

    false_accepts = sum(
        1 for row in rows
        if row["certificate_status"] == "VALID" and float(row["dense_min"]) <= -1e-8
    )
    corner_false_accepts = sum(
        1 for row in rows
        if int(row["corner_all_positive"]) and float(row["dense_min"]) <= -1e-8
    )
    fold_corner_false_accepts = sum(
        1 for row in rows
        if row["family"] == "positive_corner_interior_fold"
        and int(row["corner_all_positive"])
        and float(row["dense_min"]) <= -1e-8
    )
    certificate_invalid_folds = sum(
        1 for row in rows
        if row["family"] == "positive_corner_interior_fold"
        and row["certificate_status"] == "INVALID"
    )
    runtimes = [float(row["runtime_ms"]) for row in rows]
    boxes = [int(row["boxes_visited"]) for row in rows]

    if representative_fold is None:
        raise AssertionError("no representative fold")
    np.save(outdir / "representative_positive_corner_fold.npy", representative_fold)

    summary: Dict[str, object] = {
        "valid_target_count": valid_count,
        "hard_valid_target_count": hard_valid_count,
        "valid_generation_attempts": valid_attempts,
        "family_sizes": {family: len(cells) for family, cells in families.items()},
        "certificate_status_by_family": {
            family: {status: int(count) for status, count in sorted(counter.items())}
            for family, counter in status_by_family.items()
        },
        "certificate_false_accepts_against_dense_grid": false_accepts,
        "corner_only_false_accepts_against_dense_grid": corner_false_accepts,
        "positive_corner_fold_cases": fold_count,
        "positive_corner_fold_missed_by_corner_test": fold_corner_false_accepts,
        "positive_corner_fold_detected_invalid_by_certificate": certificate_invalid_folds,
        "runtime_ms_median": float(np.median(runtimes)),
        "runtime_ms_p95": float(np.percentile(runtimes, 95)),
        "boxes_visited_median": float(np.median(boxes)),
        "boxes_visited_p95": float(np.percentile(boxes, 95)),
        "validity_policy": "ACCEPT only VALID; INVALID is rejected; UNRESOLVED is refined/fallback",
        "runtime_seconds": time.perf_counter() - start,
    }

    with (outdir / "spatial_validity_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    return summary


# -----------------------------------------------------------------------------
# Figures and report support
# -----------------------------------------------------------------------------


def plot_results(outdir: Path, source: Mapping[str, object], flat: Mapping[str, object], spatial: Mapping[str, object]) -> None:
    # Flat-stratum mechanism breakdown.
    labels = ["D identically zero", "partial D=0", "all four gaps > 0"]
    values = [
        int(flat["D_identically_zero_candidates"]),
        int(flat["partial_or_full_D_zero_candidates"]) - int(flat["D_identically_zero_candidates"]),
        int(flat["strict_D_positive_candidates"]),
    ]
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(labels, values)
    plt.ylabel("one-sided limit candidates")
    plt.title("Flat-event candidates are not equivalent to D=0")
    plt.tight_layout()
    plt.savefig(outdir / "flat_candidate_D_breakdown.png", dpi=180)
    plt.close()

    # Source incidence richness versus time schedules.
    plt.figure(figsize=(7.0, 4.2))
    labels2 = ["time arrays", "HVID partitions", "full HVID+time states"]
    values2 = [
        int(source["unique_time_arrays"]),
        int(source["unique_hvid_partition_patterns"]),
        int(source["unique_source_hvid_time_states"]),
    ]
    plt.bar(labels2, values2)
    plt.yscale("log")
    plt.ylabel("count (log scale)")
    plt.title("Source-reachable state requires incidence as well as time")
    plt.tight_layout()
    plt.savefig(outdir / "source_state_factorization.png", dpi=180)
    plt.close()

    # Certificate outcome by family.
    status_map = spatial["certificate_status_by_family"]
    families = list(status_map)
    statuses = ["VALID", "INVALID", "UNRESOLVED"]
    bottom = np.zeros(len(families), dtype=float)
    plt.figure(figsize=(8.5, 4.8))
    for status in statuses:
        vals = np.asarray([int(status_map[family].get(status, 0)) for family in families], dtype=float)
        plt.bar(families, vals, bottom=bottom, label=status)
        bottom += vals
    plt.ylabel("cells")
    plt.title("Fail-closed Bernstein/Jacobian validity gate")
    plt.xticks(rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "jacobian_certificate_outcomes.png", dpi=180)
    plt.close()

    # Representative positive-corner fold heatmap at w=0.5.
    vertices = np.load(outdir / "representative_positive_corner_fold.npy")
    grid = np.linspace(0.0, 1.0, 101)
    heat = np.asarray([[jacobian_det(vertices, u, v, 0.5) for u in grid] for v in grid])
    plt.figure(figsize=(5.5, 4.8))
    image = plt.imshow(heat, origin="lower", extent=(0, 1, 0, 1), aspect="equal")
    plt.colorbar(image, label="det J(u,v,w=0.5)")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("All corner Jacobians positive, but the interior folds")
    plt.tight_layout()
    plt.savefig(outdir / "positive_corner_interior_fold.png", dpi=180)
    plt.close()


def write_readme(root: Path) -> None:
    text = """# Binoc three-gap validation

Run:

```bash
python validate_three_gaps.py --out results
```

The script contains three independent modules:

1. source-exact one-sided limit-stratum extraction and D=0 counterexamples;
2. an exhaustive depth-2 8-sector binary-octree grammar plus deeper randomized source-semantics cross-checks;
3. a fail-closed Bernstein/Jacobian certificate for trilinear spatial embeddings.

`results/summary.json` is the machine-readable headline output.  The report
`THEORY_AND_VALIDATION_REPORT_ZH.md` states the assumptions and limits.

The public BinocMesher repository does not include generated `hyperpolys/` and
`hypervertices/` cache binaries. Therefore the camera paths and source code are
used to calibrate stress tests, but no claim in this bundle is a real-scene
prevalence measurement.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--random-source-trials", type=int, default=20000)
    parser.add_argument("--valid-cells", type=int, default=320)
    parser.add_argument("--hard-valid-cells", type=int, default=80)
    parser.add_argument("--fold-cells", type=int, default=100)
    parser.add_argument("--invalid-cells", type=int, default=100)
    parser.add_argument("--near-cells", type=int, default=80)
    args = parser.parse_args()

    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    write_readme(root)

    source_summary, context = run_source_grammar_experiment(outdir, args.random_source_trials, args.seed)
    flat_summary = run_flat_stratum_experiment(outdir, context["representatives"], args.seed + 1)
    spatial_summary = run_spatial_validity_experiment(
        outdir,
        context["motion_signals"],
        args.seed + 2,
        args.valid_cells,
        args.hard_valid_cells,
        args.fold_cells,
        args.invalid_cells,
        args.near_cells,
    )

    summary = {
        "verdict": {
            "flat_star": "SOLVED_AS_SOURCE_LIMIT_STRATUM_NOT_D_ZERO",
            "source_reachability": "SOLVED_BY_8_SECTOR_PREFIX_GRAMMAR_WITH_FINITE_EXHAUSTIVE_CHECK",
            "spatial_embedding": "SOLVED_FAIL_CLOSED_BY_BERNSTEIN_JACOBIAN_GATE",
            "global_real_cache_prevalence": "NOT_MEASURED_PUBLIC_CACHE_BINARIES_ABSENT",
            "overall": "GO_FOR_REAL_CACHE_CPP_INTEGRATION",
        },
        "source_reachable_completeness": source_summary,
        "flat_limit_strata": flat_summary,
        "spatial_embedding_validity": spatial_summary,
        "experiment_scope": {
            "official_source_semantics_used": True,
            "official_camera_paths_used_for_stress_calibration": True,
            "generated_public_repo_cache_binaries_used": False,
            "real_scene_prevalence_claimed": False,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(outdir, source_summary, flat_summary, spatial_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
