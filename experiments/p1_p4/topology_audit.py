#!/usr/bin/env python3
"""Independent topology and triangle-intersection audit for production meshes.

The five top-level gates emitted by :func:`audit_mesh_pair` are intentionally
plain integer counters.  In particular, ``new_nonincident_intersections`` is
not an AABB overlap count: AABBs are only a sweep-line broad phase and every
candidate is passed to a triangle/triangle narrow phase.  The narrow phase
handles both non-coplanar intersections and coplanar overlap after a stable
2-D projection.

The module only requires NumPy.  It can be imported by a production runner or
used directly on NPZ, JSON, or triangular OBJ meshes::

    python3 topology_audit.py --baseline before.npz --treatment after.npz \
        --output topology.json

NPZ files must contain ``vertices`` and ``faces`` arrays.  JSON files use the
same keys.  OBJ faces must already be triangles.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Counter, Sequence

import numpy as np


INTERSECTION_ALGORITHM = (
    "sweep_aabb_broad_phase+plane_segment_triangle+coplanar_projected_2d"
    "+degenerate_primitive_fail_closed"
)

PointKey = tuple[int, int, int]
TriangleKey = tuple[PointKey, PointKey, PointKey]
IntersectionKey = tuple[TriangleKey, TriangleKey]
ChainKey = tuple[PointKey, ...]


@dataclass(frozen=True)
class _PreparedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    epsilon: float
    quantization: float


@dataclass(frozen=True)
class _IntersectionAudit:
    broadphase_candidates: int
    narrowphase_tests: int
    intersections: int
    pair_keys: Counter[IntersectionKey]


def _prepare_mesh(
    vertices: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray,
    epsilon: float | None,
) -> _PreparedMesh:
    vertex_array = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    face_array = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    if vertex_array.size == 0:
        vertex_array = np.empty((0, 3), dtype=np.float64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if face_array.size == 0:
        face_array = np.empty((0, 3), dtype=np.int64)
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if not np.isfinite(vertex_array).all():
        raise ValueError("vertices contain a non-finite coordinate")
    if face_array.size:
        if int(face_array.min()) < 0 or int(face_array.max()) >= len(vertex_array):
            raise ValueError("face vertex index is out of range")

    if len(vertex_array):
        extent = np.ptp(vertex_array, axis=0)
        scale = max(1.0, float(np.linalg.norm(extent)))
    else:
        scale = 1.0
    resolved_epsilon = float(epsilon) if epsilon is not None else scale * 1.0e-10
    if not math.isfinite(resolved_epsilon) or resolved_epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return _PreparedMesh(
        vertices=vertex_array,
        faces=face_array,
        epsilon=resolved_epsilon,
        quantization=resolved_epsilon * 8.0,
    )


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _oriented_face_key(face: Sequence[int]) -> tuple[int, int, int]:
    a, b, c = (int(value) for value in face)
    return min((a, b, c), (b, c, a), (c, a, b))


def _edge_incidence(faces: np.ndarray) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = collections.Counter()
    for a_raw, b_raw, c_raw in faces:
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        counts[_canonical_edge(a, b)] += 1
        counts[_canonical_edge(b, c)] += 1
        counts[_canonical_edge(c, a)] += 1
    return counts


def _count_nonmanifold_vertices(faces: np.ndarray) -> int:
    """Count vertices whose simplicial link is neither one path nor one cycle."""

    incident: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for a_raw, b_raw, c_raw in faces:
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        incident[a].append((b, c))
        incident[b].append((c, a))
        incident[c].append((a, b))

    failures = 0
    for link_edges in incident.values():
        multiplicity: Counter[tuple[int, int]] = collections.Counter(
            _canonical_edge(a, b) for a, b in link_edges
        )
        if any(count != 1 for count in multiplicity.values()):
            failures += 1
            continue

        adjacency: dict[int, set[int]] = collections.defaultdict(set)
        for a, b in multiplicity:
            adjacency[a].add(b)
            adjacency[b].add(a)
        if not adjacency:
            failures += 1
            continue

        unseen = set(adjacency)
        components = 0
        while unseen:
            components += 1
            stack = [unseen.pop()]
            while stack:
                current = stack.pop()
                for neighbour in adjacency[current]:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        stack.append(neighbour)

        degrees = [len(adjacency[node]) for node in adjacency]
        degree_one = sum(degree == 1 for degree in degrees)
        is_path = degree_one == 2 and all(degree in (1, 2) for degree in degrees)
        is_cycle = degree_one == 0 and all(degree == 2 for degree in degrees)
        if components != 1 or not (is_path or is_cycle):
            failures += 1
    return failures


def _quantize_point(point: np.ndarray, quantum: float) -> PointKey:
    values = np.rint(np.asarray(point, dtype=np.float64) / quantum).astype(np.int64)
    return int(values[0]), int(values[1]), int(values[2])


def _triangle_geometry_key(triangle: np.ndarray, quantum: float) -> TriangleKey:
    points = sorted(_quantize_point(point, quantum) for point in triangle)
    return points[0], points[1], points[2]


def _intersection_geometry_key(
    first: np.ndarray, second: np.ndarray, quantum: float
) -> IntersectionKey:
    a = _triangle_geometry_key(first, quantum)
    b = _triangle_geometry_key(second, quantum)
    return (a, b) if a <= b else (b, a)


def _orient_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    orientation_epsilon: float,
    coordinate_epsilon: float,
) -> bool:
    if abs(_orient_2d(start, end, point)) > orientation_epsilon:
        return False
    return bool(
        np.all(point >= np.minimum(start, end) - coordinate_epsilon)
        and np.all(point <= np.maximum(start, end) + coordinate_epsilon)
    )


def _segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    orientation_epsilon: float,
    coordinate_epsilon: float,
) -> bool:
    o1 = _orient_2d(a, b, c)
    o2 = _orient_2d(a, b, d)
    o3 = _orient_2d(c, d, a)
    o4 = _orient_2d(c, d, b)
    if (
        (o1 > orientation_epsilon and o2 < -orientation_epsilon)
        or (o1 < -orientation_epsilon and o2 > orientation_epsilon)
    ) and (
        (o3 > orientation_epsilon and o4 < -orientation_epsilon)
        or (o3 < -orientation_epsilon and o4 > orientation_epsilon)
    ):
        return True
    if abs(o1) <= orientation_epsilon and _point_on_segment_2d(
        c, a, b, orientation_epsilon, coordinate_epsilon
    ):
        return True
    if abs(o2) <= orientation_epsilon and _point_on_segment_2d(
        d, a, b, orientation_epsilon, coordinate_epsilon
    ):
        return True
    if abs(o3) <= orientation_epsilon and _point_on_segment_2d(
        a, c, d, orientation_epsilon, coordinate_epsilon
    ):
        return True
    if abs(o4) <= orientation_epsilon and _point_on_segment_2d(
        b, c, d, orientation_epsilon, coordinate_epsilon
    ):
        return True
    return False


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, epsilon: float) -> bool:
    signs = [
        _orient_2d(triangle[index], triangle[(index + 1) % 3], point)
        for index in range(3)
    ]
    return not (any(value > epsilon for value in signs) and any(value < -epsilon for value in signs))


def _coplanar_triangles_intersect(
    first: np.ndarray, second: np.ndarray, normal: np.ndarray, epsilon: float
) -> bool:
    drop_axis = int(np.argmax(np.abs(normal)))
    keep_axes = [axis for axis in range(3) if axis != drop_axis]
    a = first[:, keep_axes]
    b = second[:, keep_axes]
    orientation_epsilon = epsilon * max(
        1.0,
        float(np.ptp(np.vstack((a, b)), axis=0).max(initial=0.0)),
    )
    for first_index in range(3):
        for second_index in range(3):
            if _segments_intersect_2d(
                a[first_index],
                a[(first_index + 1) % 3],
                b[second_index],
                b[(second_index + 1) % 3],
                orientation_epsilon,
                epsilon,
            ):
                return True
    return _point_in_triangle_2d(a[0], b, orientation_epsilon) or _point_in_triangle_2d(
        b[0], a, orientation_epsilon
    )


def _point_in_triangle_3d(
    point: np.ndarray, triangle: np.ndarray, epsilon: float
) -> bool:
    edge0 = triangle[1] - triangle[0]
    edge1 = triangle[2] - triangle[0]
    normal = np.cross(edge0, edge1)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= epsilon * epsilon:
        return False
    if abs(float(np.dot(normal, point - triangle[0]))) > epsilon * normal_length:
        return False
    offset = point - triangle[0]
    d00 = float(np.dot(edge0, edge0))
    d01 = float(np.dot(edge0, edge1))
    d11 = float(np.dot(edge1, edge1))
    d20 = float(np.dot(offset, edge0))
    d21 = float(np.dot(offset, edge1))
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= epsilon * epsilon:
        return False
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    u = 1.0 - v - w
    barycentric_epsilon = epsilon / max(
        1.0, float(np.linalg.norm(edge0)), float(np.linalg.norm(edge1))
    )
    return u >= -barycentric_epsilon and v >= -barycentric_epsilon and w >= -barycentric_epsilon


def _coplanar_segment_triangle_intersection(
    start: np.ndarray,
    end: np.ndarray,
    triangle: np.ndarray,
    normal: np.ndarray,
    epsilon: float,
) -> bool:
    """Test a segment already known to lie in a non-degenerate triangle plane.

    A degenerate input triangle is geometrically a segment or a point.  Its
    longest edge can cross a regular triangle while all three source vertices
    remain outside.  Point-only containment therefore is not a sufficient
    coplanar narrow phase.
    """

    drop_axis = int(np.argmax(np.abs(normal)))
    keep_axes = [axis for axis in range(3) if axis != drop_axis]
    segment_start = start[keep_axes]
    segment_end = end[keep_axes]
    projected_triangle = triangle[:, keep_axes]
    extent = np.ptp(
        np.vstack((segment_start, segment_end, projected_triangle)), axis=0
    )
    orientation_epsilon = epsilon * max(
        1.0, float(extent.max(initial=0.0))
    )
    if _point_in_triangle_2d(
        segment_start, projected_triangle, orientation_epsilon
    ) or _point_in_triangle_2d(
        segment_end, projected_triangle, orientation_epsilon
    ):
        return True
    return any(
        _segments_intersect_2d(
            segment_start,
            segment_end,
            projected_triangle[index],
            projected_triangle[(index + 1) % 3],
            orientation_epsilon,
            epsilon,
        )
        for index in range(3)
    )


def _segment_triangle_intersection(
    start: np.ndarray, end: np.ndarray, triangle: np.ndarray, epsilon: float
) -> bool:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= epsilon * epsilon:
        return False
    direction = end - start
    denominator = float(np.dot(normal, direction))
    numerator = float(np.dot(normal, triangle[0] - start))
    if abs(denominator) <= epsilon * normal_length:
        if abs(numerator) > epsilon * normal_length:
            return False
        return _coplanar_segment_triangle_intersection(
            start, end, triangle, normal, epsilon
        )
    parameter = numerator / denominator
    segment_epsilon = epsilon / max(1.0, float(np.linalg.norm(direction)))
    if parameter < -segment_epsilon or parameter > 1.0 + segment_epsilon:
        return False
    return _point_in_triangle_3d(start + parameter * direction, triangle, epsilon)


def _segment_segment_distance_squared(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Squared distance between two closed 3-D segments (Ericson, RTCD)."""

    d1 = first_end - first_start
    d2 = second_end - second_start
    offset = first_start - second_start
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, offset))
    tiny = np.finfo(np.float64).eps
    if a <= tiny and e <= tiny:
        return float(np.dot(offset, offset))
    if a <= tiny:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, offset))
        if e <= tiny:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denominator = a * e - b * b
            s = float(np.clip((b * f - c * e) / denominator, 0.0, 1.0)) if denominator else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    closest = offset + s * d1 - t * d2
    return float(np.dot(closest, closest))


def _degenerate_triangle_intersection(
    first: np.ndarray,
    second: np.ndarray,
    first_degenerate: bool,
    second_degenerate: bool,
    epsilon: float,
) -> bool:
    if not first_degenerate:
        for index in range(3):
            if _segment_triangle_intersection(
                second[index], second[(index + 1) % 3], first, epsilon
            ):
                return True
        return any(_point_in_triangle_3d(point, first, epsilon) for point in second)
    if not second_degenerate:
        for index in range(3):
            if _segment_triangle_intersection(
                first[index], first[(index + 1) % 3], second, epsilon
            ):
                return True
        return any(_point_in_triangle_3d(point, second, epsilon) for point in first)
    threshold = epsilon * epsilon
    for first_index in range(3):
        for second_index in range(3):
            if _segment_segment_distance_squared(
                first[first_index],
                first[(first_index + 1) % 3],
                second[second_index],
                second[(second_index + 1) % 3],
            ) <= threshold:
                return True
    return False


def triangles_intersect(
    first: Sequence[Sequence[float]] | np.ndarray,
    second: Sequence[Sequence[float]] | np.ndarray,
    epsilon: float,
) -> bool:
    """Return whether two closed triangles intersect, including coplanar contact."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("triangles must have shape (3, 3)")
    normal_a = np.cross(a[1] - a[0], a[2] - a[0])
    normal_b = np.cross(b[1] - b[0], b[2] - b[0])
    length_a = float(np.linalg.norm(normal_a))
    length_b = float(np.linalg.norm(normal_b))
    degenerate_a = length_a <= epsilon * epsilon
    degenerate_b = length_b <= epsilon * epsilon
    if degenerate_a or degenerate_b:
        return _degenerate_triangle_intersection(a, b, degenerate_a, degenerate_b, epsilon)

    distances_b = np.dot(b - a[0], normal_a) / length_a
    distances_a = np.dot(a - b[0], normal_b) / length_b
    if np.all(distances_b > epsilon) or np.all(distances_b < -epsilon):
        return False
    if np.all(distances_a > epsilon) or np.all(distances_a < -epsilon):
        return False

    scale = max(
        1.0,
        float(np.linalg.norm(np.ptp(np.vstack((a, b)), axis=0))),
    )
    relative_epsilon = epsilon / scale
    normals_parallel = float(np.linalg.norm(np.cross(normal_a, normal_b))) <= (
        relative_epsilon * length_a * length_b
    )
    if normals_parallel:
        if float(np.max(np.abs(distances_b))) > epsilon:
            return False
        return _coplanar_triangles_intersect(a, b, normal_a, epsilon)

    for index in range(3):
        if _segment_triangle_intersection(a[index], a[(index + 1) % 3], b, epsilon):
            return True
        if _segment_triangle_intersection(b[index], b[(index + 1) % 3], a, epsilon):
            return True
    return any(_point_in_triangle_3d(point, b, epsilon) for point in a) or any(
        _point_in_triangle_3d(point, a, epsilon) for point in b
    )


def _audit_intersections(mesh: _PreparedMesh) -> _IntersectionAudit:
    if len(mesh.faces) < 2:
        return _IntersectionAudit(0, 0, 0, collections.Counter())

    triangles = mesh.vertices[mesh.faces]
    minimum = triangles.min(axis=1)
    maximum = triangles.max(axis=1)
    total_extent = maximum.max(axis=0) - minimum.min(axis=0)
    sweep_axis = int(np.argmax(total_extent))
    order = np.argsort(minimum[:, sweep_axis], kind="stable")
    active: list[int] = []
    broadphase_candidates = 0
    narrowphase_tests = 0
    intersection_keys: Counter[IntersectionKey] = collections.Counter()

    for raw_current in order:
        current = int(raw_current)
        current_minimum = minimum[current]
        active = [
            other
            for other in active
            if maximum[other, sweep_axis] >= current_minimum[sweep_axis] - mesh.epsilon
        ]
        current_vertices = set(int(value) for value in mesh.faces[current])
        for other in active:
            if current_vertices.intersection(int(value) for value in mesh.faces[other]):
                continue
            if np.any(maximum[other] < current_minimum - mesh.epsilon) or np.any(
                maximum[current] < minimum[other] - mesh.epsilon
            ):
                continue
            broadphase_candidates += 1
            narrowphase_tests += 1
            if triangles_intersect(triangles[other], triangles[current], mesh.epsilon):
                intersection_keys[
                    _intersection_geometry_key(
                        triangles[other], triangles[current], mesh.quantization
                    )
                ] += 1
        active.append(current)

    return _IntersectionAudit(
        broadphase_candidates=broadphase_candidates,
        narrowphase_tests=narrowphase_tests,
        intersections=sum(intersection_keys.values()),
        pair_keys=intersection_keys,
    )


def _boundary_directed_edges(mesh: _PreparedMesh) -> list[tuple[PointKey, PointKey]]:
    incidence = _edge_incidence(mesh.faces)
    directed: list[tuple[PointKey, PointKey]] = []
    for a_raw, b_raw, c_raw in mesh.faces:
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        for start, end in ((a, b), (b, c), (c, a)):
            if incidence[_canonical_edge(start, end)] == 1:
                directed.append(
                    (
                        _quantize_point(mesh.vertices[start], mesh.quantization),
                        _quantize_point(mesh.vertices[end], mesh.quantization),
                    )
                )
    return directed


def _canonical_cycle(points: Sequence[PointKey]) -> ChainKey:
    values = tuple(points)
    return min(values[index:] + values[:index] for index in range(len(values)))


def _oriented_boundary_chains(mesh: _PreparedMesh) -> Counter[ChainKey]:
    edge_counts: Counter[tuple[PointKey, PointKey]] = collections.Counter(
        _boundary_directed_edges(mesh)
    )
    outgoing: dict[PointKey, Counter[PointKey]] = collections.defaultdict(collections.Counter)
    incoming: Counter[PointKey] = collections.Counter()
    for (start, end), count in edge_counts.items():
        outgoing[start][end] += count
        incoming[end] += count

    chains: Counter[ChainKey] = collections.Counter()
    while edge_counts:
        vertices = sorted({point for edge in edge_counts for point in edge})
        starts = [
            point
            for point in vertices
            if sum(outgoing[point].values()) > incoming[point]
        ]
        start = min(starts) if starts else min(edge_counts)[0]
        chain = [start]
        current = start
        closed = False
        while True:
            candidates = sorted(
                end
                for end, count in outgoing[current].items()
                if count > 0 and edge_counts[(current, end)] > 0
            )
            if not candidates:
                break
            end = candidates[0]
            edge_counts[(current, end)] -= 1
            outgoing[current][end] -= 1
            incoming[end] -= 1
            if edge_counts[(current, end)] == 0:
                del edge_counts[(current, end)]
            chain.append(end)
            current = end
            if current == start:
                closed = True
                break

        if closed:
            signature = _canonical_cycle(chain[:-1])
        else:
            signature = tuple(chain)
        chains[signature] += 1
    return chains


def _counter_positive_difference(
    treatment: Counter[object], baseline: Counter[object]
) -> int:
    return sum(max(0, treatment[key] - baseline[key]) for key in treatment)


def _counter_symmetric_difference(first: Counter[object], second: Counter[object]) -> int:
    return sum(abs(first[key] - second[key]) for key in set(first).union(second))


def _counter_digest(counter: Counter[object]) -> str:
    payload = repr(sorted(counter.items(), key=lambda item: repr(item[0]))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _intrinsic_audit(mesh: _PreparedMesh) -> tuple[dict[str, object], _IntersectionAudit, Counter[ChainKey]]:
    incidence = _edge_incidence(mesh.faces)
    oriented_faces: Counter[tuple[int, int, int]] = collections.Counter(
        _oriented_face_key(face) for face in mesh.faces
    )
    triangles = mesh.vertices[mesh.faces] if len(mesh.faces) else np.empty((0, 3, 3))
    doubled_areas = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        if len(triangles)
        else np.empty(0)
    )
    intersections = _audit_intersections(mesh)
    chains = _oriented_boundary_chains(mesh)
    summary: dict[str, object] = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "epsilon": mesh.epsilon,
        "nonmanifold_edges": sum(count > 2 for count in incidence.values()),
        "nonmanifold_vertices": _count_nonmanifold_vertices(mesh.faces),
        "duplicate_oriented_faces": sum(max(0, count - 1) for count in oriented_faces.values()),
        "degenerate_triangles": int(np.count_nonzero(doubled_areas <= mesh.epsilon * mesh.epsilon)),
        "repeated_vertex_index_faces": int(np.count_nonzero(
            (mesh.faces[:, 0] == mesh.faces[:, 1])
            | (mesh.faces[:, 1] == mesh.faces[:, 2])
            | (mesh.faces[:, 2] == mesh.faces[:, 0])
        )),
        "boundary_edges": sum(count == 1 for count in incidence.values()),
        "oriented_boundary_chains": sum(chains.values()),
        "boundary_chain_digest": _counter_digest(chains),
        "nonincident_intersections": intersections.intersections,
        "broadphase_candidates": intersections.broadphase_candidates,
        "narrowphase_tests": intersections.narrowphase_tests,
        "intersection_algorithm": INTERSECTION_ALGORITHM,
    }
    return summary, intersections, chains


def audit_mesh(
    vertices: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray,
    *,
    epsilon: float | None = None,
) -> dict[str, object]:
    """Audit one mesh; use :func:`audit_mesh_pair` for the five release gates."""

    mesh = _prepare_mesh(vertices, faces, epsilon)
    summary, _, _ = _intrinsic_audit(mesh)
    return summary


def audit_mesh_pair(
    baseline_vertices: Sequence[Sequence[float]] | np.ndarray,
    baseline_faces: Sequence[Sequence[int]] | np.ndarray,
    treatment_vertices: Sequence[Sequence[float]] | np.ndarray,
    treatment_faces: Sequence[Sequence[int]] | np.ndarray,
    *,
    epsilon: float | None = None,
) -> dict[str, object]:
    """Return the five production gates for a baseline/treatment mesh pair."""

    baseline = _prepare_mesh(baseline_vertices, baseline_faces, epsilon)
    treatment = _prepare_mesh(treatment_vertices, treatment_faces, epsilon)
    arrays_identical = (
        np.array_equal(baseline.vertices, treatment.vertices)
        and np.array_equal(baseline.faces, treatment.faces)
    )
    # A common tolerance is necessary for geometry-key comparisons.
    shared_epsilon = max(baseline.epsilon, treatment.epsilon)
    treatment = _prepare_mesh(treatment.vertices, treatment.faces, shared_epsilon)
    treatment_summary, treatment_intersections, treatment_chains = _intrinsic_audit(treatment)
    if arrays_identical:
        baseline_summary = dict(treatment_summary)
        baseline_intersections = treatment_intersections
        baseline_chains = treatment_chains.copy()
    else:
        baseline = _prepare_mesh(baseline.vertices, baseline.faces, shared_epsilon)
        baseline_summary, baseline_intersections, baseline_chains = _intrinsic_audit(
            baseline
        )

    relative_boundary_mismatches = _counter_symmetric_difference(
        baseline_chains, treatment_chains
    )
    new_geometric_intersections = _counter_positive_difference(
        treatment_intersections.pair_keys, baseline_intersections.pair_keys
    )
    # The production runner consumes a fixed five-counter schema.  Until that
    # schema has a dedicated degenerate-face gate, fold every treatment
    # degenerate into the existing intersection failure counter.  This is
    # deliberately fail-closed: even an isolated zero-area face must not make
    # a release topology audit pass merely because it has no second face to
    # intersect.  The components remain explicit below for diagnosis.
    degenerate_face_failures = int(treatment_summary["degenerate_triangles"])
    new_intersections = new_geometric_intersections + degenerate_face_failures
    gates = {
        "nonmanifold_edges": int(treatment_summary["nonmanifold_edges"]),
        "nonmanifold_vertices": int(treatment_summary["nonmanifold_vertices"]),
        "duplicate_oriented_faces": int(treatment_summary["duplicate_oriented_faces"]),
        "relative_boundary_mismatches": relative_boundary_mismatches,
        "new_nonincident_intersections": new_intersections,
    }
    passed = all(value == 0 for value in gates.values())
    return {
        "schema": "binoc-production-topology-audit-v1",
        **gates,
        "pass": passed,
        "verdict": "PASS_PRODUCTION_TOPOLOGY_GATES" if passed else "STOP_PRODUCTION_TOPOLOGY_GATES",
        "checks": {name: value == 0 for name, value in gates.items()},
        "intersection_algorithm": INTERSECTION_ALGORITHM,
        "degenerate_faces": degenerate_face_failures,
        "broadphase_candidates": int(treatment_summary["broadphase_candidates"]),
        "narrowphase_tests": int(treatment_summary["narrowphase_tests"]),
        "baseline": baseline_summary,
        "treatment": treatment_summary,
        "relative_boundary": {
            "algorithm": "oriented_boundary_chain_geometry_multiset",
            "baseline_chains": sum(baseline_chains.values()),
            "treatment_chains": sum(treatment_chains.values()),
            "baseline_digest": _counter_digest(baseline_chains),
            "treatment_digest": _counter_digest(treatment_chains),
        },
        "intersection_delta": {
            "algorithm": "positive_geometry_pair_multiset_difference",
            "baseline_intersections": baseline_intersections.intersections,
            "treatment_intersections": treatment_intersections.intersections,
            "baseline_broadphase_candidates": baseline_intersections.broadphase_candidates,
            "treatment_broadphase_candidates": treatment_intersections.broadphase_candidates,
            "baseline_narrowphase_tests": baseline_intersections.narrowphase_tests,
            "treatment_narrowphase_tests": treatment_intersections.narrowphase_tests,
            "new_geometric_intersections": new_geometric_intersections,
            "degenerate_face_failures": degenerate_face_failures,
            "gate_value_is_component_sum": True,
        },
    }


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: OBJ face is not triangular")
            indices: list[int] = []
            for token in fields[1:]:
                index = int(token.split("/", 1)[0])
                index = len(vertices) + index if index < 0 else index - 1
                indices.append(index)
            faces.append(indices)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return np.asarray(payload["vertices"]), np.asarray(payload["faces"])
    if suffix == ".json":
        payload = json.loads(path.read_text())
        return np.asarray(payload["vertices"]), np.asarray(payload["faces"])
    if suffix == ".obj":
        return _load_obj(path)
    raise ValueError(f"unsupported mesh format: {path.suffix}")


def _index_element_records(
    records: Any, *, side: str, mesh_id: str
) -> dict[int, dict[str, Any]]:
    """Validate and index one side of a mesh pair by stable element ID."""

    if not isinstance(records, list):
        raise ValueError(
            f"{mesh_id}: {side} elements must be a list, got "
            f"{type(records).__name__}"
        )
    if not records:
        raise ValueError(f"{mesh_id}: {side} has zero mesh elements")
    indexed: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"{mesh_id}: {side} element record {position} is not an object"
            )
        if "element" not in record:
            raise ValueError(
                f"{mesh_id}: {side} element record {position} lacks element ID"
            )
        raw_element = record["element"]
        if isinstance(raw_element, bool):
            raise ValueError(
                f"{mesh_id}: {side} element record {position} has boolean element ID"
            )
        try:
            element = int(raw_element)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{mesh_id}: {side} element record {position} has invalid element ID"
            ) from error
        if element < 0:
            raise ValueError(
                f"{mesh_id}: {side} element record {position} has negative element ID"
            )
        if element in indexed:
            raise ValueError(
                f"{mesh_id}: {side} has duplicate element ID {element}"
            )
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise ValueError(
                f"{mesh_id}: {side} element {element} lacks a nonempty path"
            )
        indexed[element] = record
    return indexed


def audit_mesh_pair_task(task: dict[str, Any]) -> dict[str, Any]:
    """Pickle-safe process worker used by the production controller."""

    gate_names = (
        "nonmanifold_edges",
        "nonmanifold_vertices",
        "duplicate_oriented_faces",
        "relative_boundary_mismatches",
        "new_nonincident_intersections",
    )
    pair_gates = {name: 0 for name in gate_names}
    element_audits: list[dict[str, Any]] = []
    mesh_id = str(task["mesh_id"])
    baseline_root = Path(task["baseline_root"])
    treatment_root = Path(task["treatment_root"])
    baseline_elements = _index_element_records(
        task.get("baseline_elements"), side="baseline", mesh_id=mesh_id
    )
    treatment_elements = _index_element_records(
        task.get("treatment_elements"), side="treatment", mesh_id=mesh_id
    )
    if set(baseline_elements) != set(treatment_elements):
        raise ValueError(
            f"{mesh_id}: baseline/treatment element ID sets differ: "
            f"{sorted(baseline_elements)} != {sorted(treatment_elements)}"
        )
    for element in sorted(baseline_elements):
        first_element = baseline_elements[element]
        second_element = treatment_elements[element]
        baseline_vertices, baseline_faces = load_mesh(
            baseline_root / str(first_element["path"])
        )
        treatment_vertices, treatment_faces = load_mesh(
            treatment_root / str(second_element["path"])
        )
        audit = audit_mesh_pair(
            baseline_vertices,
            baseline_faces,
            treatment_vertices,
            treatment_faces,
        )
        element_audits.append({
            "element": element,
            **audit,
        })
        for name in gate_names:
            pair_gates[name] += int(audit[name])
    return {
        "mesh_id": mesh_id,
        **pair_gates,
        "pass": all(value == 0 for value in pair_gates.values()),
        "elements": element_audits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float)
    args = parser.parse_args()

    baseline_vertices, baseline_faces = load_mesh(args.baseline)
    treatment_vertices, treatment_faces = load_mesh(args.treatment)
    result = audit_mesh_pair(
        baseline_vertices,
        baseline_faces,
        treatment_vertices,
        treatment_faces,
        epsilon=args.epsilon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["verdict"])
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
