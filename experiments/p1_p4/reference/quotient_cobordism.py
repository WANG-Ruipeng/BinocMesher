#!/usr/bin/env python3
"""Universal quotient-cobordism construction and P0 quotient solver.

No event-shape dispatcher appears in the core compiler. Every input consists of
triangulated one-sided limits and simplicial maps into one shared critical
complex. A single staircase mapping-cylinder formula builds all event tets.
For flat births/deaths, one bounded search considers all tree quotients with at
most ``max_target_vertices`` and minimizes one quadratic motion energy.
"""
from __future__ import annotations

import collections
import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Hashable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np

import event_complex as ec

Face = Tuple[int, int, int]
Tet = Tuple[Hashable, Hashable, Hashable, Hashable]


@dataclass(frozen=True)
class Surface:
    positions: np.ndarray
    triangles: Tuple[Face, ...]
    name: str = "surface"

    @property
    def n_vertices(self) -> int:
        return int(len(self.positions))


@dataclass(frozen=True)
class QuotientCandidate:
    labels: Tuple[int, ...]
    target_edges: Tuple[Tuple[int, int], ...]
    target_vertices: int
    tetrahedra: Tuple[Tet, ...]
    volume_audit: ec.VolumeAudit


@dataclass
class EmbeddedCandidate:
    candidate: QuotientCandidate
    target_positions: np.ndarray
    action: float
    action_normalized: float
    tetrahedron_ratio: float
    objective: float


@dataclass
class GeometryAudit:
    min_gram_volume: float
    regular_slice_failures: int
    self_intersection_failures: int
    critical_edge_intersections: int
    area_slope: float
    area_at_smallest_sample: float
    passed: bool


# -----------------------------------------------------------------------------
# Universal simplicial mapping cylinder
# -----------------------------------------------------------------------------

def mapping_cylinder_tetrahedra(
    triangles: Sequence[Sequence[int]],
    quotient: Mapping[int, int],
    side: str,
    source_rank: Mapping[int, int] | None = None,
) -> Tuple[Tet, ...]:
    """Triangulate a simplicial mapping cylinder with one staircase formula.

    ``source_rank`` is a *global* total order on source vertices.  Earlier
    versions used the raw integer vertex ID as the secondary ordering key.
    That made the selected tetrahedralization (and therefore the complexity
    objective) depend on arbitrary HVID relabeling.  The fixed implementation
    accepts an explicit, relabeling-invariant source order.

    For an ordered source triangle ``v0 <= v1 <= v2`` (ordered first by target
    rank and then by ``source_rank``), generate
    ``[v0..vi, q(vi)..q(v2)]`` for i=0,1,2, remove repeated target vertices,
    and keep the 4-vertex simplices.  It automatically emits 1, 2, or 3
    tetrahedra when the triangle image has rank 0, 1, or 2.
    """
    if source_rank is None:
        source_rank = {int(v): int(v) for tri in triangles for v in tri}
    raw: List[Tet] = []
    for triangle in triangles:
        vertices = [int(v) for v in triangle]
        if len(set(vertices)) != 3:
            raise ValueError("degenerate source triangle")
        missing = [v for v in vertices if v not in quotient or v not in source_rank]
        if missing:
            raise KeyError(f"missing quotient/source rank for vertices {missing}")
        ordered = sorted(
            vertices,
            key=lambda v: (int(quotient[v]), int(source_rank[v])),
        )
        bottom = [(side, v) for v in ordered]
        top = [("critical", int(quotient[v])) for v in ordered]
        for pivot in range(3):
            sequence = bottom[: pivot + 1] + top[pivot:]
            simplex: List[Hashable] = []
            for vertex in sequence:
                if vertex not in simplex:
                    simplex.append(vertex)
            if len(simplex) == 4:
                raw.append(tuple(simplex))  # type: ignore[arg-type]
    seen = set()
    output: List[Tet] = []
    for tet in raw:
        canonical = tuple(sorted(tet, key=repr))
        if canonical not in seen:
            seen.add(canonical)
            output.append(tet)
    return tuple(output)

def double_mapping_cobordism(
    lower_triangles: Sequence[Sequence[int]],
    upper_triangles: Sequence[Sequence[int]],
    lower_quotient: Mapping[int, int],
    upper_quotient: Mapping[int, int],
    lower_source_rank: Mapping[int, int] | None = None,
    upper_source_rank: Mapping[int, int] | None = None,
) -> Tuple[Tet, ...]:
    return (
        mapping_cylinder_tetrahedra(
            lower_triangles, lower_quotient, "lower", lower_source_rank
        )
        + mapping_cylinder_tetrahedra(
            upper_triangles, upper_quotient, "upper", upper_source_rank
        )
    )


# -----------------------------------------------------------------------------
# Enumerating small graph quotients, without shape labels
# -----------------------------------------------------------------------------

def surface_edges(triangles: Sequence[Sequence[int]]) -> Tuple[Tuple[int, int], ...]:
    output = set()
    for triangle in triangles:
        for a, b in itertools.combinations(map(int, triangle), 2):
            output.add((a, b) if a < b else (b, a))
    return tuple(sorted(output))



def _rounded_pairwise_distances(points: np.ndarray, digits: int = 12) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    return np.round(np.sum(delta * delta, axis=-1), digits)


def canonical_source_ranks(
    surface: Surface,
    max_permutations: int = 500_000,
) -> Tuple[Tuple[int, ...], ...]:
    """Return all canonically tied source-vertex total orders.

    The descriptor uses only the abstract triangle complex and pairwise metric,
    so it is invariant to source/HVID relabeling and rigid transformations.
    Vertices are first partitioned by inexpensive invariant signatures; only
    permutations inside tied color classes are enumerated.  All orders that
    attain the lexicographically minimal descriptor are returned, so exact
    geometric/combinatorial symmetries are not broken by raw IDs.

    The intended Binoc endpoint domain has at most four vertices; the larger
    synthetic P0 grids (nine vertices) also remain comfortably below the cap.
    The routine fails closed instead of silently truncating.
    """
    points = np.asarray(surface.positions, dtype=float)
    n = surface.n_vertices
    if n == 0:
        return (tuple(),)
    if sorted({int(v) for tri in surface.triangles for v in tri}) != list(range(n)):
        raise ValueError("surface vertices must be dense 0..n-1")
    edges = surface_edges(surface.triangles)
    degree = collections.Counter(v for e in edges for v in e)
    incident = collections.Counter(v for tri in surface.triangles for v in tri)
    distances = _rounded_pairwise_distances(points)
    signatures: Dict[Tuple, List[int]] = collections.defaultdict(list)
    for v in range(n):
        signatures[(degree[v], incident[v], tuple(sorted(distances[v].tolist())))].append(v)
    classes = [tuple(sorted(values)) for _, values in sorted(signatures.items(), key=lambda item: repr(item[0]))]
    permutations = 1
    for cls in classes:
        permutations *= math.factorial(len(cls))
    if permutations > max_permutations:
        raise RuntimeError(
            f"canonical source-order search requires {permutations} permutations "
            f"(cap={max_permutations}); refine/fallback instead of truncating"
        )

    triangle_sets = [frozenset(map(int, tri)) for tri in surface.triangles]
    best_descriptor = None
    best_ranks: List[Tuple[int, ...]] = []
    per_class = [itertools.permutations(cls) for cls in classes]
    for parts in itertools.product(*per_class):
        order = tuple(v for part in parts for v in part)  # rank -> original vertex
        rank = [0] * n
        for r, v in enumerate(order):
            rank[v] = r
        # Descriptor of the reordered metric and simplicial complex.
        metric = tuple(
            float(distances[order[i], order[j]])
            for i in range(n)
            for j in range(i + 1, n)
        )
        triangles = tuple(
            sorted(tuple(sorted(rank[v] for v in tri)) for tri in triangle_sets)
        )
        descriptor = (triangles, metric)
        ranks = tuple(rank)
        if best_descriptor is None or descriptor < best_descriptor:
            best_descriptor = descriptor
            best_ranks = [ranks]
        elif descriptor == best_descriptor:
            best_ranks.append(ranks)
    return tuple(sorted(set(best_ranks)))


def target_trees(k: int) -> Tuple[nx.Graph, ...]:
    """Return every distinct *labelled* realization of each unlabeled tree.

    The staircase cylinder uses the numeric target order.  Keeping only one
    NetworkX labeling of an unlabeled tree silently discards valid orderings
    where, for example, the center of a 3-path has target rank 1 instead of 0.
    All k! relabelings are therefore generated and edge-set deduplicated.
    """
    if k == 1:
        graph = nx.Graph()
        graph.add_node(0)
        return (graph,)
    labelled: Dict[Tuple[Tuple[int, int], ...], nx.Graph] = {}
    for base in nx.generators.nonisomorphic_trees(k):
        base = nx.convert_node_labels_to_integers(base, ordering="sorted")
        for permutation in itertools.permutations(range(k)):
            mapping = {old: int(permutation[old]) for old in range(k)}
            edges = tuple(
                sorted(
                    (min(mapping[a], mapping[b]), max(mapping[a], mapping[b]))
                    for a, b in base.edges
                )
            )
            if edges in labelled:
                continue
            graph = nx.Graph()
            graph.add_nodes_from(range(k))
            graph.add_edges_from(edges)
            labelled[edges] = graph
    return tuple(labelled[key] for key in sorted(labelled))


def target_automorphisms(tree: nx.Graph) -> Tuple[Dict[int, int], ...]:
    matcher = nx.algorithms.isomorphism.GraphMatcher(tree, tree)
    return tuple(dict(mapping) for mapping in matcher.isomorphisms_iter())


def canonicalize_target_labels(labels: Sequence[int], tree: nx.Graph) -> Tuple[int, ...]:
    return min(
        tuple(int(mapping[int(label)]) for label in labels)
        for mapping in target_automorphisms(tree)
    )


def enumerate_tree_maps(
    n_vertices: int,
    triangles: Sequence[Sequence[int]],
    tree: nx.Graph,
    max_maps: int = 250_000,
) -> Tuple[Tuple[int, ...], ...]:
    """Enumerate *all* surjective simplicial maps to one labelled target tree.

    Earlier code canonicalized target-tree automorphisms before constructing
    the mapping cylinder.  That is unsound because the staircase
    tetrahedralization depends on the target total order: two automorphism-
    equivalent labelings can have different tetrahedron counts.  We retain all
    labelings and only deduplicate identical final candidates.
    """
    domain = nx.Graph()
    domain.add_nodes_from(range(n_vertices))
    domain.add_edges_from(surface_edges(triangles))
    order = sorted(domain.nodes, key=lambda v: (-domain.degree[v], v))
    closed_neighbors = {v: set(tree.neighbors(v)) | {v} for v in tree.nodes}
    assignment: Dict[int, int] = {}
    raw: List[Tuple[int, ...]] = []
    truncated = False

    def recurse(index: int) -> None:
        nonlocal truncated
        if len(raw) >= max_maps:
            truncated = True
            return
        if index == len(order):
            labels = tuple(int(assignment[v]) for v in range(n_vertices))
            if len(set(labels)) == len(tree):
                raw.append(labels)
            return
        vertex = order[index]
        candidates = set(tree.nodes)
        for neighbor in domain.neighbors(vertex):
            if neighbor in assignment:
                candidates &= closed_neighbors[assignment[neighbor]]
        for label in sorted(candidates):
            assignment[vertex] = int(label)
            recurse(index + 1)
            del assignment[vertex]
            if truncated:
                return

    recurse(0)
    if truncated:
        raise RuntimeError(
            f"tree-map enumeration exceeded cap={max_maps}; "
            "fail closed instead of silently returning an incomplete quotient set"
        )
    return tuple(sorted(set(raw)))


@lru_cache(maxsize=None)
def enumerate_valid_candidates_cached(
    n_vertices: int,
    triangles: Tuple[Face, ...],
    max_target_vertices: int,
    source_ranks: Tuple[Tuple[int, ...], ...],
) -> Tuple[QuotientCandidate, ...]:
    output: List[QuotientCandidate] = []
    seen = set()
    for k in range(1, min(max_target_vertices, n_vertices) + 1):
        for tree in target_trees(k):
            target_edges = tuple(sorted((min(a, b), max(a, b)) for a, b in tree.edges))
            for labels in enumerate_tree_maps(n_vertices, triangles, tree):
                quotient = {vertex: int(labels[vertex]) for vertex in range(n_vertices)}
                for rank_tuple in source_ranks:
                    source_rank = {vertex: int(rank_tuple[vertex]) for vertex in range(n_vertices)}
                    tetrahedra = mapping_cylinder_tetrahedra(
                        triangles, quotient, "upper", source_rank
                    )
                    audit = ec.audit_volume(tetrahedra)
                    if not audit.valid_relative_3_manifold:
                        continue
                    # Exact candidate identity; target automorphisms/order variants
                    # are intentionally retained if they induce a different complex.
                    key = (
                        labels,
                        target_edges,
                        tuple(sorted((tuple(sorted(t, key=repr)) for t in tetrahedra), key=repr)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    output.append(
                        QuotientCandidate(
                            labels=labels,
                            target_edges=target_edges,
                            target_vertices=k,
                            tetrahedra=tetrahedra,
                            volume_audit=audit,
                        )
                    )
    output.sort(
        key=lambda candidate: (
            candidate.target_vertices,
            len(candidate.tetrahedra),
            quotient_partition_signature(candidate),
            candidate.target_edges,
            candidate.labels,
            tuple(sorted((tuple(sorted(t, key=repr)) for t in candidate.tetrahedra), key=repr)),
        )
    )
    return tuple(output)


def enumerate_valid_candidates(
    surface: Surface,
    max_target_vertices: int = 4,
) -> Tuple[QuotientCandidate, ...]:
    triangles = tuple(tuple(map(int, triangle)) for triangle in surface.triangles)
    ranks = canonical_source_ranks(surface)
    return enumerate_valid_candidates_cached(
        surface.n_vertices, triangles, max_target_vertices, ranks
    )


# -----------------------------------------------------------------------------
# One minimum-action embedding
# -----------------------------------------------------------------------------

def triangle_areas(surface: Surface) -> np.ndarray:
    p = np.asarray(surface.positions, dtype=float)
    return np.asarray(
        [0.5 * np.linalg.norm(np.cross(p[b] - p[a], p[c] - p[a])) for a, b, c in surface.triangles],
        dtype=float,
    )


def integrated_action(surface: Surface, labels: Sequence[int], target: np.ndarray) -> float:
    points = np.asarray(surface.positions, dtype=float)
    areas = triangle_areas(surface)
    energy = 0.0
    for area, triangle in zip(areas, surface.triangles):
        displacement = [points[v] - target[int(labels[v])] for v in triangle]
        energy += area / 6.0 * (
            sum(float(np.dot(d, d)) for d in displacement)
            + sum(
                float(np.dot(displacement[i], displacement[j]))
                for i, j in itertools.combinations(range(3), 2)
            )
        )
    return float(max(energy, 0.0))


def solve_minimum_action(
    surface: Surface,
    candidate: QuotientCandidate,
    edge_regularization: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    k = candidate.target_vertices
    labels = candidate.labels
    points = np.asarray(surface.positions, dtype=float)
    areas = triangle_areas(surface)
    hessian = np.zeros((k, k), dtype=float)
    rhs = np.zeros((k, 3), dtype=float)

    # For linearly interpolated displacement d on a triangle:
    # mean ||d||^2 = (sum ||d_i||^2 + sum_{i<j} d_i.d_j) / 6.
    for area, triangle in zip(areas, surface.triangles):
        vertices = [int(v) for v in triangle]
        labs = [int(labels[v]) for v in vertices]
        xyz = [points[v] for v in vertices]
        for i in range(3):
            weight = area / 6.0
            hessian[labs[i], labs[i]] += weight
            rhs[labs[i]] += weight * xyz[i]
        for i, j in itertools.combinations(range(3), 2):
            weight = area / 6.0
            li, lj = labs[i], labs[j]
            if li == lj:
                hessian[li, li] += weight
            else:
                hessian[li, lj] += weight / 2.0
                hessian[lj, li] += weight / 2.0
            rhs[li] += weight * xyz[j] / 2.0
            rhs[lj] += weight * xyz[i] / 2.0

    scale = max(float(areas.sum()), 1e-12)
    for a, b in candidate.target_edges:
        weight = edge_regularization * scale
        hessian[a, a] += weight
        hessian[b, b] += weight
        hessian[a, b] -= weight
        hessian[b, a] -= weight
    hessian += np.eye(k) * scale * 1e-12
    target = np.linalg.solve(hessian, rhs)
    return target, integrated_action(surface, labels, target)


def point_action(surface: Surface) -> float:
    candidate = enumerate_valid_candidates(surface, 1)[0]
    return solve_minimum_action(surface, candidate)[1]


def per_face_centroid_action(surface: Surface) -> float:
    points = np.asarray(surface.positions, dtype=float)
    areas = triangle_areas(surface)
    total = 0.0
    for area, triangle in zip(areas, surface.triangles):
        xyz = points[np.asarray(triangle, dtype=int)]
        center = xyz.mean(axis=0)
        displacement = xyz - center
        total += area / 6.0 * (
            sum(float(np.dot(d, d)) for d in displacement)
            + sum(
                float(np.dot(displacement[i], displacement[j]))
                for i, j in itertools.combinations(range(3), 2)
            )
        )
    return float(total)


def continuous_pca_line_action(surface: Surface) -> float:
    points = np.asarray(surface.positions, dtype=float)
    areas = triangle_areas(surface)
    weights = np.zeros(surface.n_vertices, dtype=float)
    for area, triangle in zip(areas, surface.triangles):
        for vertex in triangle:
            weights[vertex] += area / 3.0
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / max(float(weights.sum()), 1e-12)
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    projected = center + np.outer(centered @ axis, axis)
    return integrated_action(surface, tuple(range(surface.n_vertices)), projected)


def score_candidates(
    surface: Surface,
    complexity_weight: float,
    max_target_vertices: int = 4,
) -> Tuple[EmbeddedCandidate, ...]:
    point = point_action(surface)
    n_faces = max(len(surface.triangles), 1)
    output: List[EmbeddedCandidate] = []
    for candidate in enumerate_valid_candidates(surface, max_target_vertices):
        target, action = solve_minimum_action(surface, candidate)
        normalized = action / max(point, 1e-15)
        tetrahedron_ratio = len(candidate.tetrahedra) / n_faces
        objective = normalized + complexity_weight * tetrahedron_ratio
        output.append(
            EmbeddedCandidate(
                candidate=candidate,
                target_positions=target,
                action=action,
                action_normalized=normalized,
                tetrahedron_ratio=tetrahedron_ratio,
                objective=objective,
            )
        )
    output.sort(key=embedded_tie_key)
    return tuple(output)


def embedded_tie_key(result: EmbeddedCandidate) -> Tuple:
    block_sizes = tuple(sorted(collections.Counter(result.candidate.labels).values()))
    centroid_signature = tuple(
        tuple(np.round(row, 10)) for row in sorted(result.target_positions.tolist())
    )
    return (
        round(result.objective, 13),
        round(result.action, 13),
        len(result.candidate.tetrahedra),
        result.candidate.target_vertices,
        block_sizes,
        quotient_partition_signature(result.candidate),
        centroid_signature,
        tuple(sorted((tuple(sorted(t, key=repr)) for t in result.candidate.tetrahedra), key=repr)),
    )


def choose_candidate(
    surface: Surface,
    complexity_weight: float,
    max_target_vertices: int = 4,
) -> EmbeddedCandidate:
    candidates = score_candidates(surface, complexity_weight, max_target_vertices)
    if not candidates:
        raise RuntimeError("no valid quotient; a point quotient should exist for every disk")
    return candidates[0]


# -----------------------------------------------------------------------------
# Geometry realization and local embedding audit
# -----------------------------------------------------------------------------

def gram_3volume(points4: np.ndarray) -> float:
    edges = (points4[1:] - points4[0]).T
    determinant = max(float(np.linalg.det(edges.T @ edges)), 0.0)
    return math.sqrt(determinant) / 6.0


def embedded_vertices4(surface: Surface, embedded: EmbeddedCandidate) -> Dict[Hashable, np.ndarray]:
    output: Dict[Hashable, np.ndarray] = {}
    for index, point in enumerate(surface.positions):
        output[("upper", index)] = np.r_[point, 1.0]
    for index, point in enumerate(embedded.target_positions):
        output[("critical", index)] = np.r_[point, 0.0]
    return output


def slice_tetrahedral_complex(
    vertices4: Mapping[Hashable, np.ndarray],
    tetrahedra: Sequence[Sequence[Hashable]],
    tau: float,
) -> Tuple[np.ndarray, Tuple[Face, ...]]:
    edge_cache: Dict[Tuple[Hashable, Hashable], int] = {}
    vertices: List[np.ndarray] = []
    faces: List[Face] = []

    def edge_key(a: Hashable, b: Hashable) -> Tuple[Hashable, Hashable]:
        return tuple(sorted((a, b), key=repr))  # type: ignore[return-value]

    for tet in tetrahedra:
        crossing: List[int] = []
        for a, b in itertools.combinations(tet, 2):
            pa, pb = vertices4[a], vertices4[b]
            ta, tb = float(pa[3]), float(pb[3])
            if (ta < tau < tb) or (tb < tau < ta):
                key = edge_key(a, b)
                if key not in edge_cache:
                    weight = (tau - ta) / (tb - ta)
                    edge_cache[key] = len(vertices)
                    vertices.append((1.0 - weight) * pa[:3] + weight * pb[:3])
                crossing.append(edge_cache[key])
        crossing = list(dict.fromkeys(crossing))
        if len(crossing) == 3:
            faces.append(tuple(crossing))  # type: ignore[arg-type]
        elif len(crossing) == 4:
            xyz = np.asarray([vertices[index] for index in crossing])
            center = xyz.mean(axis=0)
            _, _, vh = np.linalg.svd(xyz - center, full_matrices=False)
            angle = np.arctan2((xyz - center) @ vh[1], (xyz - center) @ vh[0])
            cycle = [crossing[index] for index in np.argsort(angle)]
            d02 = np.linalg.norm(vertices[cycle[0]] - vertices[cycle[2]])
            d13 = np.linalg.norm(vertices[cycle[1]] - vertices[cycle[3]])
            if d02 <= d13:
                faces.extend(((cycle[0], cycle[1], cycle[2]), (cycle[0], cycle[2], cycle[3])))
            else:
                faces.extend(((cycle[1], cycle[2], cycle[3]), (cycle[1], cycle[3], cycle[0])))
        elif crossing:
            raise RuntimeError(f"unexpected slice polygon with {len(crossing)} vertices")
    return np.asarray(vertices, dtype=float), tuple(faces)


def segment_triangle_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    triangle: np.ndarray,
    eps: float = 1e-9,
) -> bool:
    direction = p1 - p0
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    h = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, h))
    if abs(determinant) < eps:
        return False
    inverse = 1.0 / determinant
    s = p0 - triangle[0]
    u = inverse * float(np.dot(s, h))
    if u < -eps or u > 1.0 + eps:
        return False
    q = np.cross(s, edge1)
    v = inverse * float(np.dot(direction, q))
    if v < -eps or u + v > 1.0 + eps:
        return False
    t = inverse * float(np.dot(edge2, q))
    return -eps <= t <= 1.0 + eps


def triangles_intersect(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> bool:
    if np.any(a.max(axis=0) < b.min(axis=0) - eps) or np.any(b.max(axis=0) < a.min(axis=0) - eps):
        return False
    normal_a = np.cross(a[1] - a[0], a[2] - a[0])
    normal_b = np.cross(b[1] - b[0], b[2] - b[0])
    length_a, length_b = np.linalg.norm(normal_a), np.linalg.norm(normal_b)
    if length_a < eps or length_b < eps:
        return True
    normal_a /= length_a
    normal_b /= length_b
    if np.linalg.norm(np.cross(normal_a, normal_b)) < 1e-6:
        if float(np.abs((b - a[0]) @ normal_a).max()) > eps:
            return False
        from shapely.geometry import Polygon
        drop = int(np.argmax(np.abs(normal_a)))
        keep = [axis for axis in range(3) if axis != drop]
        intersection = Polygon(a[:, keep]).intersection(Polygon(b[:, keep]))
        return (not intersection.is_empty) and (intersection.area > eps or intersection.length > eps)
    for source, target in ((a, b), (b, a)):
        for index in range(3):
            if segment_triangle_intersection(source[index], source[(index + 1) % 3], target, eps):
                return True
    return False


def count_nonincident_intersections(vertices: np.ndarray, faces: Sequence[Sequence[int]]) -> int:
    failures = 0
    for i, j in itertools.combinations(range(len(faces)), 2):
        if set(faces[i]) & set(faces[j]):
            continue
        if triangles_intersect(vertices[np.asarray(faces[i])], vertices[np.asarray(faces[j])]):
            failures += 1
    return failures


def count_target_edge_intersections(target: np.ndarray, edges: Sequence[Tuple[int, int]]) -> int:
    def segment_distance(p1, q1, p2, q2) -> float:
        u, v, w = q1 - p1, q2 - p2, p1 - p2
        a, b, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
        d, e = np.dot(u, w), np.dot(v, w)
        denominator = a * c - b * b
        s = np.clip((b * e - c * d) / denominator, 0.0, 1.0) if denominator > 1e-15 else 0.0
        t = np.clip((b * s + e) / max(c, 1e-15), 0.0, 1.0)
        s = np.clip((b * t - d) / max(a, 1e-15), 0.0, 1.0)
        return float(np.linalg.norm((p1 + s * u) - (p2 + t * v)))

    scale = max(float(np.linalg.norm(target.max(axis=0) - target.min(axis=0))), 1.0)
    failures = 0
    for edge_a, edge_b in itertools.combinations(edges, 2):
        if set(edge_a) & set(edge_b):
            continue
        if segment_distance(
            target[edge_a[0]], target[edge_a[1]], target[edge_b[0]], target[edge_b[1]]
        ) < 1e-8 * scale:
            failures += 1
    return failures


def geometry_audit(surface: Surface, embedded: EmbeddedCandidate) -> GeometryAudit:
    vertices4 = embedded_vertices4(surface, embedded)
    tetrahedra = embedded.candidate.tetrahedra
    volumes = [gram_3volume(np.asarray([vertices4[vertex] for vertex in tet])) for tet in tetrahedra]
    min_volume = min(volumes) if volumes else 0.0
    slice_failures = 0
    intersection_failures = 0
    near_critical: List[Tuple[float, float]] = []
    for tau in (0.02, 0.04, 0.08, 0.16, 0.35, 0.65, 0.9):
        vertices, faces = slice_tetrahedral_complex(vertices4, tetrahedra, tau)
        audit = ec.audit_surface(faces)
        if not (
            audit.nonmanifold_edges == 0
            and audit.vertex_link_failures == 0
            and audit.orientable
            and audit.boundary_graph_valid
            and audit.connected_components == 1
            and audit.topology_class == "DISK"
        ):
            slice_failures += 1
        intersection_failures += count_nonincident_intersections(vertices, faces)
        area = 0.0
        for a, b, c in faces:
            area += 0.5 * float(np.linalg.norm(np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])))
        if tau <= 0.16:
            near_critical.append((tau, max(area, 1e-30)))
    slope = float(
        np.polyfit(np.log([x for x, _ in near_critical]), np.log([y for _, y in near_critical]), 1)[0]
    )
    target_intersections = count_target_edge_intersections(
        embedded.target_positions, embedded.candidate.target_edges
    )
    scale = max(float(np.linalg.norm(surface.positions.max(axis=0) - surface.positions.min(axis=0))), 1.0)
    passed = (
        min_volume > 1e-11 * scale**3
        and slice_failures == 0
        and intersection_failures == 0
        and target_intersections == 0
        and slope > 0.5
    )
    return GeometryAudit(
        min_gram_volume=float(min_volume),
        regular_slice_failures=slice_failures,
        self_intersection_failures=intersection_failures,
        critical_edge_intersections=target_intersections,
        area_slope=slope,
        area_at_smallest_sample=near_critical[0][1],
        passed=passed,
    )


def quotient_partition_signature(candidate: QuotientCandidate) -> Tuple:
    blocks: Dict[int, List[int]] = collections.defaultdict(list)
    for vertex, label in enumerate(candidate.labels):
        blocks[int(label)].append(vertex)
    block_sets = {label: frozenset(vertices) for label, vertices in blocks.items()}
    vertices = tuple(sorted(block_sets.values(), key=lambda block: (len(block), tuple(sorted(block)))))
    edges = tuple(
        sorted(
            (
                tuple(
                    sorted(
                        (block_sets[a], block_sets[b]),
                        key=lambda block: (len(block), tuple(sorted(block))),
                    )
                )
                for a, b in candidate.target_edges
            ),
            key=repr,
        )
    )
    return vertices, edges
