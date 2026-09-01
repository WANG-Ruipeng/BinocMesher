#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

TV_DIR = Path(__file__).resolve().parents[1] / "tv0_tv4"
if str(TV_DIR) not in sys.path:
    sys.path.insert(0, str(TV_DIR))
from run_lightweight_profile import make_cameras, terrain  # type: ignore


@dataclass(frozen=True, order=True)
class SourceKey:
    node0: int
    group0: int
    node1: int
    group1: int

    @classmethod
    def canonical(cls, node0: int, group0: int, node1: int, group1: int) -> "SourceKey":
        first = (int(node0), int(group0))
        second = (int(node1), int(group1))
        if second < first:
            first, second = second, first
        return cls(first[0], first[1], second[0], second[1])

    def token(self) -> str:
        return f"{self.node0}:{self.group0}|{self.node1}:{self.group1}"


@dataclass(frozen=True, order=True)
class TriangleRef:
    element: int
    t_group: int
    t_start: int
    sorted_record_index: int
    interval_index: int
    face_index: int
    fan_index: int

    def values(self) -> tuple[int, ...]:
        return (
            self.element, self.t_group, self.t_start,
            self.sorted_record_index, self.interval_index,
            self.face_index, self.fan_index,
        )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_face(face: np.ndarray) -> tuple[int, int, int]:
    a, b, c = (int(value) for value in face)
    rotations = ((a, b, c), (b, c, a), (c, a, b))
    return min(rotations)


def canonical_mesh_hash(vertices: np.ndarray, faces: np.ndarray, tags: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
        np.ascontiguousarray(tags, dtype=np.int32),
    ):
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def oriented_face_multiset(faces: np.ndarray):
    import collections
    return collections.Counter(
        canonical_face(face) for face in np.asarray(faces, dtype=np.int64)
    )


def make_mesher(repo: Path, cache_root: Path, slicing_time: float):
    os.environ["BINOC_EVENT_MODE"] = "1"
    os.environ["BINOC_PROVENANCE_V2"] = "1"
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from binocmesher import BinocMesher  # type: ignore

    cameras = make_cameras(24, width=640, height=360, step=0.3)
    return BinocMesher(
        cameras,
        bounds=[-1000.0, 1000.0, -1000.0, 1000.0, -10.0, 10.0],
        slicing_time=slicing_time,
        pixels_per_cube=180,
        pixels_per_cube_coarse=360,
        pixels_per_cube_outview=720,
        min_dist=0.1,
        simplify_occluded=False,
        relax_margin=0,
        boundary_margin=1,
        relax_iters=0,
        n_coarse_nodes=200000,
        bisection_iters=1,
        fading_time=1.0 / 24.0,
        seed_stride=2,
        medium_group=20000,
        fine_group=5000,
        bisection_group=1000000,
        path=cache_root,
    )


def initialize_mesher(repo: Path, cache_root: Path):
    # A normal cache-reuse call initializes all C++ parameters and validates the
    # slicing cache contract.  Its mesh is discarded before the exact call.
    mesher = make_mesher(repo, cache_root, slicing_time=(0.5 + 12) / 24.0)
    mesher([terrain])
    return mesher


def exact_mesh(
    repo: Path,
    cache_root: Path,
    exact_time: Fraction,
    *,
    plan: Path | None = None,
    audit: Path | None = None,
    trace: Path | None = None,
    omp_threads: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    for name in (
        "BINOC_SOURCE_SPLICE_PLAN",
        "BINOC_SOURCE_SPLICE_AUDIT",
        "BINOC_SOURCE_SPLICE_TRACE",
    ):
        os.environ.pop(name, None)

    # Initialize/validate the ordinary cache without an intervention.  The SSP1
    # plan is exact-rational only and is enabled immediately before the exact
    # C ABI call below.
    mesher = initialize_mesher(repo, cache_root)
    if plan is not None:
        os.environ["BINOC_SOURCE_SPLICE_PLAN"] = str(plan.resolve())
    if audit is not None:
        os.environ["BINOC_SOURCE_SPLICE_AUDIT"] = str(audit.resolve())
    if trace is not None:
        os.environ["BINOC_SOURCE_SPLICE_TRACE"] = str(trace.resolve())
    from binocmesher.utils.interface import AsInt  # type: ignore

    n_elements = 1
    v_counts = np.zeros(n_elements, dtype=np.int32)
    f_counts = np.zeros(n_elements, dtype=np.int32)
    status = int(mesher.run_slicing_rational(
        int(exact_time.numerator), int(exact_time.denominator),
        AsInt(v_counts), AsInt(f_counts), False,
    ))
    if status != 0:
        raw = mesher.slicing_last_error()
        detail = raw.decode("utf-8", errors="replace") if raw else f"status {status}"
        raise RuntimeError(f"run_slicing_rational failed closed: {detail}")

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    for element in range(n_elements):
        v = np.zeros((int(v_counts[element]), 3), dtype=np.float64)
        f = np.zeros((int(f_counts[element]), 3), dtype=np.int32)
        t = np.zeros(int(v_counts[element]), dtype=np.int32)
        mesher.slicing_output(
            element, mesher.AF(v), AsInt(f), AsInt(t))
        vertices.append(v)
        faces.append(f)
        tags.append(t)
    mesher.slicing_clean_up()

    metadata = {
        "exact_time": {
            "numerator": exact_time.numerator,
            "denominator": exact_time.denominator,
        },
        "counts": [
            {"vertices": len(vertices[i]), "faces": len(faces[i])}
            for i in range(n_elements)
        ],
        "mesh_hashes": [
            canonical_mesh_hash(vertices[i], faces[i], tags[i])
            for i in range(n_elements)
        ],
        "plan": None if plan is None else str(plan),
        "trace": None if trace is None else str(trace),
        "omp_threads": omp_threads,
    }
    metadata["mesh_sha256"] = metadata["mesh_hashes"][0]
    metadata["pass"] = True
    return vertices, faces, tags, metadata


def save_mesh_npz(
    path: Path,
    vertices: list[np.ndarray],
    faces: list[np.ndarray],
    tags: list[np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=vertices[0],
        faces=faces[0],
        tags=tags[0],
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def load_mesh_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["vertices"]),
            np.asarray(data["faces"]),
            np.asarray(data["tags"]),
            json.loads(str(data["metadata"].item())),
        )

# ---------------------------------------------------------------------------
# Independent mesh-audit helpers used by the source-splice validation.
# ---------------------------------------------------------------------------

def canonical_oriented_face(face: np.ndarray | tuple[int, int, int]) -> tuple[int, int, int]:
    """Canonicalize cyclic rotation while preserving triangle orientation."""
    a, b, c = (int(value) for value in face)
    return min((a, b, c), (b, c, a), (c, a, b))


def mesh_digest(vertices: np.ndarray, faces: np.ndarray, tags: np.ndarray) -> str:
    return canonical_mesh_hash(vertices, faces, tags)


def mesh_topology(vertices: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    """Return finite combinatorial audits for an indexed triangle mesh."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return {
            "V": 0,
            "E": 0,
            "F": 0,
            "chi": 0,
            "components": 0,
            "boundary_edges": 0,
            "boundary_components": 0,
            "boundary_loops": 0,
            "nonmanifold_edges": 0,
            "duplicate_faces": 0,
            "degenerate_faces": 0,
        }

    edge_incidence: dict[tuple[int, int], int] = {}
    unoriented_faces: dict[tuple[int, int, int], int] = {}
    degenerate = 0
    valid_faces: list[tuple[int, int, int]] = []
    for row in faces:
        a, b, c = map(int, row)
        if a == b or b == c or c == a:
            degenerate += 1
            continue
        if min(a, b, c) < 0 or max(a, b, c) >= len(vertices):
            degenerate += 1
            continue
        p0, p1, p2 = vertices[[a, b, c]]
        scale = max(1.0, float(np.max(np.abs([p0, p1, p2]))))
        if float(np.linalg.norm(np.cross(p1 - p0, p2 - p0))) <= 1e-12 * scale * scale:
            degenerate += 1
            continue
        valid_faces.append((a, b, c))
        key = tuple(sorted((a, b, c)))
        unoriented_faces[key] = unoriented_faces.get(key, 0) + 1
        for x, y in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((x, y)))
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1

    used = sorted({value for face in valid_faces for value in face})
    parent = {value: value for value in used}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            following = parent[value]
            parent[value] = root
            value = following
        return root

    def union(first: int, second: int) -> None:
        if first not in parent or second not in parent:
            return
        a, b = find(first), find(second)
        if a != b:
            parent[b] = a

    for a, b, c in valid_faces:
        union(a, b)
        union(b, c)
        union(c, a)

    boundary = [edge for edge, count in edge_incidence.items() if count == 1]
    boundary_nodes = sorted({value for edge in boundary for value in edge})
    boundary_parent = {value: value for value in boundary_nodes}

    def boundary_find(value: int) -> int:
        root = value
        while boundary_parent[root] != root:
            root = boundary_parent[root]
        while boundary_parent[value] != value:
            following = boundary_parent[value]
            boundary_parent[value] = root
            value = following
        return root

    for a, b in boundary:
        ra, rb = boundary_find(a), boundary_find(b)
        if ra != rb:
            boundary_parent[rb] = ra

    boundary_components = (
        len({boundary_find(value) for value in boundary_nodes})
        if boundary_nodes else 0
    )
    return {
        "V": len(used),
        "E": len(edge_incidence),
        "F": len(valid_faces),
        "chi": len(used) - len(edge_incidence) + len(valid_faces),
        "components": len({find(value) for value in used}) if used else 0,
        "boundary_edges": len(boundary),
        "boundary_components": boundary_components,
        "boundary_loops": boundary_components,
        "nonmanifold_edges": sum(count > 2 for count in edge_incidence.values()),
        "duplicate_faces": sum(
            count - 1 for count in unoriented_faces.values() if count > 1
        ),
        "degenerate_faces": degenerate,
    }


def _project_triangle(points: np.ndarray, axis: int) -> np.ndarray:
    keep = [index for index in range(3) if index != axis]
    return points[:, keep]


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) -
                 (b[1] - a[1]) * (c[0] - a[0]))


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, eps: float) -> bool:
    values = [
        _orient2d(triangle[index], triangle[(index + 1) % 3], point)
        for index in range(3)
    ]
    return (
        all(value >= -eps for value in values) or
        all(value <= eps for value in values)
    )


def _segments_intersect_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float
) -> bool:
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    def on_segment(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> bool:
        return (
            abs(_orient2d(x, y, p)) <= eps and
            min(x[0], y[0]) - eps <= p[0] <= max(x[0], y[0]) + eps and
            min(x[1], y[1]) - eps <= p[1] <= max(x[1], y[1]) + eps
        )

    return any((
        on_segment(a, b, c), on_segment(a, b, d),
        on_segment(c, d, a), on_segment(c, d, b),
    ))


def _coplanar_triangles_intersect(first: np.ndarray, second: np.ndarray, normal: np.ndarray,
                                   eps: float) -> bool:
    axis = int(np.argmax(np.abs(normal)))
    first2 = _project_triangle(first, axis)
    second2 = _project_triangle(second, axis)
    for i in range(3):
        for j in range(3):
            if _segments_intersect_2d(
                first2[i], first2[(i + 1) % 3],
                second2[j], second2[(j + 1) % 3], eps,
            ):
                return True
    return (
        _point_in_triangle_2d(first2[0], second2, eps) or
        _point_in_triangle_2d(second2[0], first2, eps)
    )


def _segment_triangle_intersection(
    first: np.ndarray, second: np.ndarray, triangle: np.ndarray, eps: float
) -> bool:
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    direction = second - first
    pvec = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, pvec))
    if abs(determinant) <= eps:
        return False
    inverse = 1.0 / determinant
    tvec = first - triangle[0]
    u = float(np.dot(tvec, pvec)) * inverse
    if u < -eps or u > 1.0 + eps:
        return False
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inverse
    if v < -eps or u + v > 1.0 + eps:
        return False
    parameter = float(np.dot(edge2, qvec)) * inverse
    return -eps <= parameter <= 1.0 + eps


def _triangles_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    scale = max(1.0, float(np.max(np.abs(np.concatenate((first, second))))))
    eps = 1e-10 * scale
    if np.any(np.max(first, axis=0) < np.min(second, axis=0) - eps) or \
       np.any(np.max(second, axis=0) < np.min(first, axis=0) - eps):
        return False
    n1 = np.cross(first[1] - first[0], first[2] - first[0])
    n2 = np.cross(second[1] - second[0], second[2] - second[0])
    norm1 = float(np.linalg.norm(n1))
    norm2 = float(np.linalg.norm(n2))
    if norm1 <= eps or norm2 <= eps:
        return False
    distances_second = (second - first[0]) @ n1
    distances_first = (first - second[0]) @ n2
    plane_eps1 = eps * norm1
    plane_eps2 = eps * norm2
    if (np.all(distances_second > plane_eps1) or
        np.all(distances_second < -plane_eps1) or
        np.all(distances_first > plane_eps2) or
        np.all(distances_first < -plane_eps2)):
        return False
    parallel = float(np.linalg.norm(np.cross(n1, n2))) <= eps * norm1 * norm2
    coplanar = parallel and np.max(np.abs(distances_second)) <= plane_eps1
    if coplanar:
        return _coplanar_triangles_intersect(first, second, n1, eps)
    for triangle, other in ((first, second), (second, first)):
        for index in range(3):
            if _segment_triangle_intersection(
                triangle[index], triangle[(index + 1) % 3], other, eps
            ):
                return True
    return False


def patch_nonincident_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    patch_face_indices: list[int],
) -> list[tuple[int, int]]:
    """Audit a small replacement patch against all nonincident mesh faces."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    patch_set = set(int(index) for index in patch_face_indices)
    result: list[tuple[int, int]] = []
    for first_index in sorted(patch_set):
        first_ids = set(map(int, faces[first_index]))
        first = vertices[faces[first_index]]
        for second_index, second_face in enumerate(faces):
            # Patch faces are a distinguished set, so every external face
            # must be tested even when its array index is smaller.
            if second_index in patch_set:
                continue
            second_ids = set(map(int, second_face))
            if first_ids & second_ids:
                continue
            second = vertices[second_face]
            if _triangles_intersect(first, second):
                result.append((first_index, int(second_index)))
    return result
