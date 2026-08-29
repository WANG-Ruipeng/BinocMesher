#!/usr/bin/env python3
"""Exact combinatorial kernels for incidence-compatible event-star transitions.

This module deliberately separates three statements:

1. A finite event-face complex is a valid PL surface patch (combinatorial).
2. A supported patch can be compiled to a local simplicial spacetime 3-complex.
3. A particular geometric embedding is nondegenerate / injective (geometric).

The primary supported kernels are:

* STAR_CONE: a connected disk or sphere is coned to one *global* event apex.
  Unlike per-face centroid extrusion, every shared event vertex has one path.
* PRODUCT: a triangulated surface patch with fixed combinatorics is connected
  to a second embedding by the staircase triangulation of K x [0,1].

Unsupported/nonmanifold inputs fail closed rather than being silently repaired.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from itertools import combinations
from math import sqrt
from typing import Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

Vertex = Hashable
Edge = Tuple[Vertex, Vertex]
Triangle = Tuple[Vertex, Vertex, Vertex]
Tetrahedron = Tuple[Vertex, Vertex, Vertex, Vertex]
Vec3 = Tuple[Fraction, Fraction, Fraction]
Vec4 = Tuple[Fraction, Fraction, Fraction, Fraction]


def key(v: Vertex) -> str:
    return repr(v)


def canonical_edge(a: Vertex, b: Vertex) -> Edge:
    return (a, b) if key(a) <= key(b) else (b, a)


def canonical_simplex(vertices: Sequence[Vertex]) -> Tuple[Vertex, ...]:
    return tuple(sorted(vertices, key=key))


def permutation_parity_to_sorted(values: Sequence[Vertex]) -> int:
    """Sign (+1/-1) of the permutation from values to repr-sorted order."""
    order = sorted(range(len(values)), key=lambda i: key(values[i]))
    inversions = sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))
    return -1 if inversions % 2 else 1


def graph_components(vertices: Iterable[Vertex], edges: Iterable[Edge]) -> int:
    vertices = list(vertices)
    adjacency: Dict[Vertex, Set[Vertex]] = defaultdict(set)
    for v in vertices:
        adjacency[v]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: Set[Vertex] = set()
    count = 0
    for start in vertices:
        if start in seen:
            continue
        count += 1
        seen.add(start)
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
    return count


def path_or_circle(vertices: Iterable[Vertex], edges: Iterable[Edge]) -> Tuple[bool, bool, int, Dict[int, int]]:
    vertices = set(vertices)
    edges = list(edges)
    if not vertices:
        return False, False, 0, {}
    degrees: Counter[Vertex] = Counter()
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    components = graph_components(vertices, edges)
    hist = Counter(degrees.values())
    is_interval = (
        components == 1
        and len(edges) == len(vertices) - 1
        and hist[1] == 2
        and all(d in (1, 2) for d in degrees.values())
    )
    is_circle = (
        components == 1
        and len(edges) == len(vertices)
        and all(d == 2 for d in degrees.values())
    )
    return is_interval, is_circle, components, dict(sorted(hist.items()))


def clean_cycle(cycle: Sequence[Vertex]) -> Tuple[Vertex, ...]:
    out: List[Vertex] = []
    for v in cycle:
        if not out or out[-1] != v:
            out.append(v)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return tuple(out)


def rotate_cycle_to_min(cycle: Sequence[Vertex]) -> Tuple[Vertex, ...]:
    cycle = clean_cycle(cycle)
    if not cycle:
        return ()
    i = min(range(len(cycle)), key=lambda j: key(cycle[j]))
    return tuple(cycle[i:]) + tuple(cycle[:i])


def triangulate_polygon(cycle: Sequence[Vertex]) -> Tuple[Triangle, ...]:
    """Canonical abstract fan triangulation preserving the supplied orientation."""
    cycle = rotate_cycle_to_min(cycle)
    if len(cycle) < 3:
        raise ValueError("polygon needs at least three vertices")
    if len(set(cycle)) != len(cycle):
        raise ValueError("polygon has a nonconsecutive repeated vertex")
    v0 = cycle[0]
    return tuple((v0, cycle[i], cycle[i + 1]) for i in range(1, len(cycle) - 1))


@dataclass(frozen=True)
class VertexLink:
    vertex: Vertex
    incident_triangles: int
    link_vertices: int
    link_edges: int
    components: int
    degree_histogram: Dict[int, int]
    is_interval: bool
    is_circle: bool
    is_boundary_vertex: bool
    valid: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vertex"] = repr(self.vertex)
        return d


@dataclass
class SurfaceAudit:
    vertices: int
    edges: int
    triangles: int
    euler_characteristic: int
    connected_components: int
    boundary_edges: int
    internal_edges: int
    nonmanifold_edges: int
    boundary_loops: int
    boundary_graph_valid: bool
    orientable: bool
    duplicate_triangles: int
    vertex_link_failures: int
    topology_class: str
    valid_relative_surface: bool
    vertex_links: List[VertexLink] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vertex_links"] = [x.to_dict() for x in self.vertex_links]
        return d


def triangle_oriented_edge_sign(tri: Triangle, edge: Edge) -> int:
    a, b = edge
    for i in range(3):
        u, v = tri[i], tri[(i + 1) % 3]
        if u == a and v == b:
            return 1
        if u == b and v == a:
            return -1
    raise AssertionError((tri, edge))


def surface_orientable(triangles: Sequence[Triangle], edge_to_tris: Mapping[Edge, List[int]]) -> bool:
    signs: Dict[int, int] = {}
    adjacency: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for edge, incident in edge_to_tris.items():
        if len(incident) != 2:
            continue
        i, j = incident
        di = triangle_oriented_edge_sign(triangles[i], edge)
        dj = triangle_oriented_edge_sign(triangles[j], edge)
        # s_i d_i = - s_j d_j
        relation = -di * dj  # s_j = relation * s_i
        adjacency[i].append((j, relation))
        adjacency[j].append((i, relation))
    for start in range(len(triangles)):
        if start in signs:
            continue
        signs[start] = 1
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j, relation in adjacency[i]:
                expected = relation * signs[i]
                if j in signs:
                    if signs[j] != expected:
                        return False
                else:
                    signs[j] = expected
                    queue.append(j)
    return True


def audit_surface(triangles: Sequence[Sequence[Vertex]]) -> SurfaceAudit:
    tris: List[Triangle] = [tuple(t) for t in triangles]  # type: ignore[list-item]
    if any(len(t) != 3 or len(set(t)) != 3 for t in tris):
        raise ValueError("all triangles must contain exactly three distinct vertices")

    vertices = set(v for t in tris for v in t)
    edge_counts: Counter[Edge] = Counter()
    edge_to_tris: Dict[Edge, List[int]] = defaultdict(list)
    for i, tri in enumerate(tris):
        for j in range(3):
            e = canonical_edge(tri[j], tri[(j + 1) % 3])
            edge_counts[e] += 1
            edge_to_tris[e].append(i)

    canonical_tris = Counter(canonical_simplex(t) for t in tris)
    duplicate_triangles = sum(count - 1 for count in canonical_tris.values() if count > 1)
    boundary = {e for e, count in edge_counts.items() if count == 1}
    internal = {e for e, count in edge_counts.items() if count == 2}
    nonmanifold = {e for e, count in edge_counts.items() if count > 2}

    boundary_vertices = set(v for e in boundary for v in e)
    boundary_degrees: Counter[Vertex] = Counter()
    for a, b in boundary:
        boundary_degrees[a] += 1
        boundary_degrees[b] += 1
    boundary_graph_valid = not boundary or all(d == 2 for d in boundary_degrees.values())
    boundary_loops = graph_components(boundary_vertices, boundary) if boundary and boundary_graph_valid else 0

    vlinks: List[VertexLink] = []
    for v in sorted(vertices, key=key):
        link_edges: List[Edge] = []
        incident = 0
        for tri in tris:
            if v not in tri:
                continue
            incident += 1
            others = [x for x in tri if x != v]
            link_edges.append(canonical_edge(others[0], others[1]))
        link_vertices = set(x for e in link_edges for x in e)
        is_interval, is_circle, components, hist = path_or_circle(link_vertices, link_edges)
        is_boundary = v in boundary_vertices
        valid = is_interval if is_boundary else is_circle
        vlinks.append(
            VertexLink(
                vertex=v,
                incident_triangles=incident,
                link_vertices=len(link_vertices),
                link_edges=len(link_edges),
                components=components,
                degree_histogram=hist,
                is_interval=is_interval,
                is_circle=is_circle,
                is_boundary_vertex=is_boundary,
                valid=valid,
            )
        )

    components = graph_components(vertices, edge_counts)
    chi = len(vertices) - len(edge_counts) + len(tris)
    orientable = surface_orientable(tris, edge_to_tris)
    vertex_failures = sum(not x.valid for x in vlinks)

    if not tris:
        topology = "EMPTY"
    elif duplicate_triangles:
        topology = "DUPLICATE_MULTIPLICITY"
    elif nonmanifold:
        topology = "NONMANIFOLD_EDGE"
    elif vertex_failures:
        topology = "NONMANIFOLD_VERTEX"
    elif components != 1:
        topology = "MULTI_COMPONENT"
    elif not boundary_graph_valid:
        topology = "INVALID_BOUNDARY"
    elif not orientable:
        topology = "NONORIENTABLE"
    elif boundary_loops == 0 and chi == 2:
        topology = "SPHERE"
    elif boundary_loops == 1 and chi == 1:
        topology = "DISK"
    elif boundary_loops == 2 and chi == 0:
        topology = "ANNULUS"
    else:
        topology = "OTHER_ORIENTABLE_SURFACE"

    valid_surface = (
        topology in {"DISK", "SPHERE", "ANNULUS", "OTHER_ORIENTABLE_SURFACE"}
        and not nonmanifold
        and vertex_failures == 0
        and components == 1
        and orientable
        and boundary_graph_valid
        and duplicate_triangles == 0
    )

    return SurfaceAudit(
        vertices=len(vertices),
        edges=len(edge_counts),
        triangles=len(tris),
        euler_characteristic=chi,
        connected_components=components,
        boundary_edges=len(boundary),
        internal_edges=len(internal),
        nonmanifold_edges=len(nonmanifold),
        boundary_loops=boundary_loops,
        boundary_graph_valid=boundary_graph_valid,
        orientable=orientable,
        duplicate_triangles=duplicate_triangles,
        vertex_link_failures=vertex_failures,
        topology_class=topology,
        valid_relative_surface=valid_surface,
        vertex_links=vlinks,
    )


@dataclass(frozen=True)
class EdgeLink3D:
    edge: Edge
    boundary_edge: bool
    link_vertices: int
    link_edges: int
    components: int
    degree_histogram: Dict[int, int]
    is_interval: bool
    is_circle: bool
    valid: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge"] = [repr(self.edge[0]), repr(self.edge[1])]
        return d


@dataclass(frozen=True)
class VertexLink3D:
    vertex: Vertex
    boundary_vertex: bool
    topology_class: str
    valid: bool
    audit: SurfaceAudit

    def to_dict(self) -> dict:
        return {
            "vertex": repr(self.vertex),
            "boundary_vertex": self.boundary_vertex,
            "topology_class": self.topology_class,
            "valid": self.valid,
            "audit": self.audit.to_dict(),
        }


@dataclass
class VolumeAudit:
    vertices: int
    edges: int
    faces: int
    tetrahedra: int
    euler_characteristic: int
    connected_components: int
    boundary_faces: int
    internal_faces: int
    nonmanifold_faces: int
    duplicate_tetrahedra: int
    orientable: bool
    edge_link_failures: int
    vertex_link_failures: int
    valid_relative_3_manifold: bool
    edge_links: List[EdgeLink3D] = field(default_factory=list)
    vertex_links: List[VertexLink3D] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_links"] = [x.to_dict() for x in self.edge_links]
        d["vertex_links"] = [x.to_dict() for x in self.vertex_links]
        return d


def oriented_face_sign(tet: Tetrahedron, face: Tuple[Vertex, Vertex, Vertex]) -> int:
    face_set = set(face)
    for omitted in range(4):
        induced = tuple(tet[i] for i in range(4) if i != omitted)
        if set(induced) == face_set:
            return ((-1) ** omitted) * permutation_parity_to_sorted(induced)
    raise AssertionError((tet, face))


def volume_orientable(tets: Sequence[Tetrahedron], face_to_tets: Mapping[Tuple[Vertex, Vertex, Vertex], List[int]]) -> bool:
    signs: Dict[int, int] = {}
    adjacency: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for face, incident in face_to_tets.items():
        if len(incident) != 2:
            continue
        i, j = incident
        di = oriented_face_sign(tets[i], face)
        dj = oriented_face_sign(tets[j], face)
        relation = -di * dj
        adjacency[i].append((j, relation))
        adjacency[j].append((i, relation))
    for start in range(len(tets)):
        if start in signs:
            continue
        signs[start] = 1
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j, relation in adjacency[i]:
                expected = relation * signs[i]
                if j in signs:
                    if signs[j] != expected:
                        return False
                else:
                    signs[j] = expected
                    queue.append(j)
    return True


def audit_volume(tetrahedra: Sequence[Sequence[Vertex]]) -> VolumeAudit:
    tets: List[Tetrahedron] = [tuple(t) for t in tetrahedra]  # type: ignore[list-item]
    if any(len(t) != 4 or len(set(t)) != 4 for t in tets):
        raise ValueError("all tetrahedra must contain exactly four distinct vertices")

    vertices = set(v for t in tets for v in t)
    edges: Set[Edge] = set()
    face_counts: Counter[Tuple[Vertex, Vertex, Vertex]] = Counter()
    face_to_tets: Dict[Tuple[Vertex, Vertex, Vertex], List[int]] = defaultdict(list)
    for i, tet in enumerate(tets):
        for a, b in combinations(tet, 2):
            edges.add(canonical_edge(a, b))
        for face in combinations(tet, 3):
            f = canonical_simplex(face)
            face_counts[f] += 1
            face_to_tets[f].append(i)

    duplicate_tets = sum(c - 1 for c in Counter(canonical_simplex(t) for t in tets).values() if c > 1)
    boundary_faces = {f for f, c in face_counts.items() if c == 1}
    internal_faces = {f for f, c in face_counts.items() if c == 2}
    nonmanifold_faces = {f for f, c in face_counts.items() if c > 2}
    boundary_vertices = set(v for f in boundary_faces for v in f)
    boundary_edges = set(canonical_edge(a, b) for f in boundary_faces for a, b in combinations(f, 2))

    # Tet components by shared triangular faces, as required for a pure 3-complex.
    tet_adj: Dict[int, Set[int]] = defaultdict(set)
    for incident in face_to_tets.values():
        if len(incident) == 2:
            a, b = incident
            tet_adj[a].add(b)
            tet_adj[b].add(a)
    components = 0
    seen: Set[int] = set()
    for i in range(len(tets)):
        if i in seen:
            continue
        components += 1
        seen.add(i)
        queue = deque([i])
        while queue:
            j = queue.popleft()
            for k in tet_adj[j]:
                if k not in seen:
                    seen.add(k)
                    queue.append(k)

    edge_links: List[EdgeLink3D] = []
    for edge in sorted(edges, key=lambda e: (key(e[0]), key(e[1]))):
        link_edges: List[Edge] = []
        for tet in tets:
            if set(edge) <= set(tet):
                opp = [v for v in tet if v not in edge]
                link_edges.append(canonical_edge(opp[0], opp[1]))
        link_vertices = set(v for e in link_edges for v in e)
        is_interval, is_circle, comps, hist = path_or_circle(link_vertices, link_edges)
        is_boundary = edge in boundary_edges
        valid = is_interval if is_boundary else is_circle
        edge_links.append(
            EdgeLink3D(
                edge=edge,
                boundary_edge=is_boundary,
                link_vertices=len(link_vertices),
                link_edges=len(link_edges),
                components=comps,
                degree_histogram=hist,
                is_interval=is_interval,
                is_circle=is_circle,
                valid=valid,
            )
        )

    vertex_links: List[VertexLink3D] = []
    for v in sorted(vertices, key=key):
        link_tris: List[Triangle] = []
        for tet in tets:
            if v in tet:
                link_tris.append(tuple(x for x in tet if x != v))  # type: ignore[arg-type]
        audit = audit_surface(link_tris)
        is_boundary = v in boundary_vertices
        valid = audit.topology_class == ("DISK" if is_boundary else "SPHERE")
        vertex_links.append(
            VertexLink3D(
                vertex=v,
                boundary_vertex=is_boundary,
                topology_class=audit.topology_class,
                valid=valid,
                audit=audit,
            )
        )

    orientable = volume_orientable(tets, face_to_tets)
    chi = len(vertices) - len(edges) + len(face_counts) - len(tets)
    valid = (
        bool(tets)
        and not nonmanifold_faces
        and duplicate_tets == 0
        and components == 1
        and orientable
        and all(e.valid for e in edge_links)
        and all(v.valid for v in vertex_links)
    )

    return VolumeAudit(
        vertices=len(vertices),
        edges=len(edges),
        faces=len(face_counts),
        tetrahedra=len(tets),
        euler_characteristic=chi,
        connected_components=components,
        boundary_faces=len(boundary_faces),
        internal_faces=len(internal_faces),
        nonmanifold_faces=len(nonmanifold_faces),
        duplicate_tetrahedra=duplicate_tets,
        orientable=orientable,
        edge_link_failures=sum(not e.valid for e in edge_links),
        vertex_link_failures=sum(not v.valid for v in vertex_links),
        valid_relative_3_manifold=valid,
        edge_links=edge_links,
        vertex_links=vertex_links,
    )


@dataclass
class CompileResult:
    status: str
    kernel: str
    reason: str
    input_audit: SurfaceAudit
    tetrahedra: Tuple[Tetrahedron, ...]
    output_audit: Optional[VolumeAudit]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "kernel": self.kernel,
            "reason": self.reason,
            "input_audit": self.input_audit.to_dict(),
            "tetrahedra": [[repr(v) for v in t] for t in self.tetrahedra],
            "output_audit": self.output_audit.to_dict() if self.output_audit else None,
        }


def compile_star_cone(triangles: Sequence[Triangle], apex: Vertex = ("event_apex", 0)) -> CompileResult:
    audit = audit_surface(triangles)
    if audit.topology_class not in {"DISK", "SPHERE"}:
        return CompileResult(
            status="UNSUPPORTED",
            kernel="STAR_CONE",
            reason=f"cone apex link would be {audit.topology_class}, not a disk/sphere",
            input_audit=audit,
            tetrahedra=(),
            output_audit=None,
        )
    if apex in {v for t in triangles for v in t}:
        raise ValueError("apex must be a new global event vertex")
    tets = tuple((apex, tri[0], tri[1], tri[2]) for tri in triangles)
    out = audit_volume(tets)
    status = "PASS" if out.valid_relative_3_manifold else "FAIL_CERTIFICATE"
    return CompileResult(
        status=status,
        kernel="STAR_CONE",
        reason="certified" if status == "PASS" else "constructed cone failed 3-manifold audit",
        input_audit=audit,
        tetrahedra=tets,
        output_audit=out,
    )


def product_vertex(v: Vertex, side: int) -> Tuple[str, int, Vertex]:
    return ("product", side, v)


def compile_product(triangles: Sequence[Triangle]) -> CompileResult:
    """Staircase triangulation of K x [0,1] using one global vertex order.

    For each ordered base triangle a<b<c, the triangular prism is split into:
      [a0,b0,c0,c1], [a0,b0,b1,c1], [a0,a1,b1,c1].
    The restriction to every shared edge is therefore globally consistent.
    """
    audit = audit_surface(triangles)
    if not audit.valid_relative_surface:
        return CompileResult(
            status="UNSUPPORTED",
            kernel="PRODUCT",
            reason=f"input is not a connected orientable PL surface: {audit.topology_class}",
            input_audit=audit,
            tetrahedra=(),
            output_audit=None,
        )
    tets: List[Tetrahedron] = []
    for tri in triangles:
        a, b, c = sorted(tri, key=key)
        a0, b0, c0 = (product_vertex(v, 0) for v in (a, b, c))
        a1, b1, c1 = (product_vertex(v, 1) for v in (a, b, c))
        tets.extend(
            [
                (a0, b0, c0, c1),
                (a0, b0, b1, c1),
                (a0, a1, b1, c1),
            ]
        )
    out = audit_volume(tets)
    status = "PASS" if out.valid_relative_3_manifold else "FAIL_CERTIFICATE"
    return CompileResult(
        status=status,
        kernel="PRODUCT",
        reason="certified" if status == "PASS" else "staircase product failed 3-manifold audit",
        input_audit=audit,
        tetrahedra=tuple(tets),
        output_audit=out,
    )


def add3(a: Sequence[Fraction], b: Sequence[Fraction]) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def sub3(a: Sequence[Fraction], b: Sequence[Fraction]) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def scale3(s: Fraction, a: Sequence[Fraction]) -> Vec3:
    return tuple(s * x for x in a)  # type: ignore[return-value]


def cross3(a: Sequence[Fraction], b: Sequence[Fraction]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm3_float(a: Sequence[Fraction]) -> float:
    return sqrt(float(sum(x * x for x in a)))


def triangle_area_float(a: Sequence[Fraction], b: Sequence[Fraction], c: Sequence[Fraction]) -> float:
    return 0.5 * norm3_float(cross3(sub3(b, a), sub3(c, a)))


def surface_area_float(triangles: Sequence[Triangle], positions: Mapping[Vertex, Vec3]) -> float:
    return sum(triangle_area_float(positions[a], positions[b], positions[c]) for a, b, c in triangles)


def occurrence_weighted_apex(triangles: Sequence[Triangle], positions: Mapping[Vertex, Vec3]) -> Vec3:
    occurrence = [v for tri in triangles for v in tri]
    total: Vec3 = (Fraction(0), Fraction(0), Fraction(0))
    for v in occurrence:
        total = add3(total, positions[v])
    return scale3(Fraction(1, len(occurrence)), total)


def cone_slice_positions(positions: Mapping[Vertex, Vec3], apex: Vec3, s: Fraction) -> Dict[Vertex, Vec3]:
    return {v: add3(apex, scale3(s, sub3(p, apex))) for v, p in positions.items()}


def verify_cone_area_law(
    triangles: Sequence[Triangle], positions: Mapping[Vertex, Vec3], apex: Vec3,
    samples: Sequence[Fraction] = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)),
    tolerance: float = 1e-12,
) -> dict:
    base = surface_area_float(triangles, positions)
    rows = []
    passed = True
    for s in samples:
        current = cone_slice_positions(positions, apex, s)
        area = surface_area_float(triangles, current)
        expected = float(s * s) * base
        error = abs(area - expected)
        passed &= error <= tolerance * max(1.0, base)
        rows.append({"s": str(s), "area": area, "expected": expected, "abs_error": error})
    return {"passed": bool(passed), "base_area": base, "samples": rows}


def shared_trajectory_gap(
    positions: Mapping[Vertex, Vec3], apex_a: Vec3, apex_b: Vec3, shared: Iterable[Vertex], s: Fraction
) -> float:
    max_gap = 0.0
    for v in shared:
        pa = add3(apex_a, scale3(s, sub3(positions[v], apex_a)))
        pb = add3(apex_b, scale3(s, sub3(positions[v], apex_b)))
        max_gap = max(max_gap, norm3_float(sub3(pa, pb)))
    return max_gap

@dataclass
class NormalizedCompileResult:
    status: str
    reason: str
    input_polygon_count: int
    simple_loop_count: int
    dropped_lower_dimensional_cycles: int
    vertex_splits: int
    raw_surface_audit: Optional[SurfaceAudit]
    normalized_surface_audit: Optional[SurfaceAudit]
    component_audits: List[SurfaceAudit]
    component_kernels: List[CompileResult]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "input_polygon_count": self.input_polygon_count,
            "simple_loop_count": self.simple_loop_count,
            "dropped_lower_dimensional_cycles": self.dropped_lower_dimensional_cycles,
            "vertex_splits": self.vertex_splits,
            "raw_surface_audit": self.raw_surface_audit.to_dict() if self.raw_surface_audit else None,
            "normalized_surface_audit": self.normalized_surface_audit.to_dict() if self.normalized_surface_audit else None,
            "component_audits": [x.to_dict() for x in self.component_audits],
            "component_kernels": [x.to_dict() for x in self.component_kernels],
        }


def split_closed_walk_at_repeats(cycle: Sequence[Vertex]) -> List[Tuple[Vertex, ...]]:
    """Decompose a closed boundary walk at nonconsecutive repeated vertices.

    The result is a collection of simple cycles plus possibly lower-dimensional
    cycles. This is a combinatorial normalization; using it as geometry requires
    a component-separation certificate from the occupancy representation.
    """
    cycle = clean_cycle(cycle)
    if not cycle:
        return []
    first_pos: Dict[Vertex, int] = {}
    for j, v in enumerate(cycle):
        if v in first_pos:
            i = first_pos[v]
            loop = cycle[i:j]
            remainder = cycle[: i + 1] + cycle[j + 1 :]
            result: List[Tuple[Vertex, ...]] = []
            result.extend(split_closed_walk_at_repeats(loop))
            result.extend(split_closed_walk_at_repeats(remainder))
            return result
        first_pos[v] = j
    return [tuple(cycle)]


def triangle_components_by_edges(triangles: Sequence[Triangle]) -> List[List[int]]:
    edge_to_tris: Dict[Edge, List[int]] = defaultdict(list)
    for i, tri in enumerate(triangles):
        for j in range(3):
            edge_to_tris[canonical_edge(tri[j], tri[(j + 1) % 3])].append(i)
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for incident in edge_to_tris.values():
        for a, b in combinations(incident, 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen: Set[int] = set()
    result: List[List[int]] = []
    for start in range(len(triangles)):
        if start in seen:
            continue
        comp: List[int] = []
        seen.add(start)
        queue = deque([start])
        while queue:
            i = queue.popleft()
            comp.append(i)
            for j in adjacency[i]:
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
        result.append(sorted(comp))
    return result


def split_nonmanifold_vertex_links(triangles: Sequence[Triangle]) -> Tuple[Tuple[Triangle, ...], int]:
    """Duplicate a vertex once per connected component of its triangle star.

    Two incident triangles are in the same sheet at v iff they can be connected
    through edges containing v. This repairs vertex-only pinches but deliberately
    does not pair or split edges of degree > 2.
    """
    tris = [tuple(t) for t in triangles]
    incident: Dict[Vertex, List[int]] = defaultdict(list)
    for i, tri in enumerate(tris):
        for v in tri:
            incident[v].append(i)

    replacements: Dict[Tuple[int, Vertex], Vertex] = {}
    split_count = 0
    for v, indices in incident.items():
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        for a, b in combinations(indices, 2):
            other_a = set(tris[a]) - {v}
            other_b = set(tris[b]) - {v}
            if other_a & other_b:  # shared edge containing v
                adjacency[a].add(b)
                adjacency[b].add(a)
        comps: List[List[int]] = []
        seen: Set[int] = set()
        for start in indices:
            if start in seen:
                continue
            comp: List[int] = []
            seen.add(start)
            queue = deque([start])
            while queue:
                i = queue.popleft()
                comp.append(i)
                for j in adjacency[i]:
                    if j not in seen:
                        seen.add(j)
                        queue.append(j)
            comps.append(comp)
        if len(comps) <= 1:
            continue
        split_count += len(comps) - 1
        for comp_id, comp in enumerate(comps):
            new_v: Vertex = ("sheet_split", v, comp_id)
            for tri_index in comp:
                replacements[(tri_index, v)] = new_v

    out: List[Triangle] = []
    for i, tri in enumerate(tris):
        out.append(tuple(replacements.get((i, v), v) for v in tri))  # type: ignore[arg-type]
    return tuple(out), split_count


def compile_normalized_same_side(
    polygon_cycles: Sequence[Sequence[Vertex]], apex_prefix: Vertex = "normalized_apex"
) -> NormalizedCompileResult:
    loops: List[Tuple[Vertex, ...]] = []
    dropped = 0
    for cycle in polygon_cycles:
        for loop in split_closed_walk_at_repeats(cycle):
            if len(loop) < 3 or len(set(loop)) < 3:
                dropped += 1
            else:
                loops.append(loop)

    triangles: List[Triangle] = []
    for loop_id, loop in enumerate(loops):
        try:
            triangles.extend(triangulate_polygon_cell(loop, ("normalized", loop_id)))
        except ValueError:
            return NormalizedCompileResult(
                status="UNSUPPORTED",
                reason="polygon normalization did not produce simple cycles",
                input_polygon_count=len(polygon_cycles),
                simple_loop_count=len(loops),
                dropped_lower_dimensional_cycles=dropped,
                vertex_splits=0,
                raw_surface_audit=None,
                normalized_surface_audit=None,
                component_audits=[],
                component_kernels=[],
            )

    if not triangles:
        return NormalizedCompileResult(
            status="PASS_LOWER_DIMENSIONAL_NO_2D_KERNEL",
            reason="all carriers collapse below dimension two; no finite-area kernel is emitted",
            input_polygon_count=len(polygon_cycles),
            simple_loop_count=0,
            dropped_lower_dimensional_cycles=dropped,
            vertex_splits=0,
            raw_surface_audit=None,
            normalized_surface_audit=None,
            component_audits=[],
            component_kernels=[],
        )

    raw = audit_surface(triangles)
    if raw.nonmanifold_edges or raw.duplicate_triangles or not raw.orientable:
        return NormalizedCompileResult(
            status="UNSUPPORTED",
            reason=f"requires noncanonical edge/multiplicity repair: {raw.topology_class}",
            input_polygon_count=len(polygon_cycles),
            simple_loop_count=len(loops),
            dropped_lower_dimensional_cycles=dropped,
            vertex_splits=0,
            raw_surface_audit=raw,
            normalized_surface_audit=raw,
            component_audits=[raw],
            component_kernels=[],
        )

    normalized, split_count = split_nonmanifold_vertex_links(triangles)
    normalized_audit = audit_surface(normalized)
    components = triangle_components_by_edges(normalized)
    component_audits: List[SurfaceAudit] = []
    kernels: List[CompileResult] = []
    for component_id, indices in enumerate(components):
        comp_tris = tuple(normalized[i] for i in indices)
        comp_audit = audit_surface(comp_tris)
        component_audits.append(comp_audit)
        kernel = compile_star_cone(comp_tris, apex=(apex_prefix, component_id))
        kernels.append(kernel)

    if all(k.status == "PASS" for k in kernels):
        if split_count:
            status = "PASS_WITH_COMPONENT_SPLIT"
        elif len(components) > 1:
            status = "PASS_WITH_EVENT_STAR_COMPONENT_SPLIT"
        elif dropped:
            status = "PASS_WITH_LOWER_DIMENSIONAL_PRUNING"
        else:
            status = "PASS_DIRECT"
        reason = "all normalized sheet components compile to certified disk/sphere cones"
    else:
        status = "UNSUPPORTED"
        reason = "at least one normalized component is not cone-supported"

    return NormalizedCompileResult(
        status=status,
        reason=reason,
        input_polygon_count=len(polygon_cycles),
        simple_loop_count=len(loops),
        dropped_lower_dimensional_cycles=dropped,
        vertex_splits=split_count,
        raw_surface_audit=raw,
        normalized_surface_audit=normalized_audit,
        component_audits=component_audits,
        component_kernels=kernels,
    )

@dataclass
class PolygonProductResult:
    status: str
    reason: str
    input_sizes: Tuple[int, int]
    common_boundary_vertices: int
    endpoint_steiner_vertices: Tuple[int, int]
    reference_triangles: Tuple[Triangle, ...]
    product: Optional[CompileResult]
    endpoint_parameter_maps: Tuple[List[dict], List[dict]]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "input_sizes": list(self.input_sizes),
            "common_boundary_vertices": self.common_boundary_vertices,
            "endpoint_steiner_vertices": list(self.endpoint_steiner_vertices),
            "reference_triangles": [[repr(v) for v in t] for t in self.reference_triangles],
            "product": self.product.to_dict() if self.product else None,
            "endpoint_parameter_maps": self.endpoint_parameter_maps,
        }


def boundary_parameter_map(cycle: Sequence[Vertex], parameters: Sequence[Fraction]) -> List[dict]:
    n = len(cycle)
    result=[]
    for p in parameters:
        value=p*n
        i=int(value) % n
        alpha=value-int(value)
        j=(i+1)%n
        result.append({
            "parameter": str(p),
            "edge": [repr(cycle[i]),repr(cycle[j])],
            "alpha": str(alpha),
            "is_original_vertex": alpha == 0,
        })
    return result


def compile_polygon_mapping_cylinder(cycle0: Sequence[Vertex], cycle1: Sequence[Vertex]) -> PolygonProductResult:
    """Common-boundary refinement + product for two simple polygonal disks.

    The theorem-level precondition is that the caller supplies an
    orientation-preserving boundary homeomorphism. Uniform perimeter parameters
    encode one such abstract homeomorphism. Geometric embeddedness of the morph
    remains a separate certificate (e.g. common-reference normal graphs).
    """
    c0=clean_cycle(cycle0); c1=clean_cycle(cycle1)
    if len(c0)<3 or len(c1)<3 or len(set(c0))!=len(c0) or len(set(c1))!=len(c1):
        return PolygonProductResult(
            status="UNSUPPORTED", reason="endpoint is not a simple polygonal disk",
            input_sizes=(len(c0),len(c1)), common_boundary_vertices=0,
            endpoint_steiner_vertices=(0,0), reference_triangles=(), product=None,
            endpoint_parameter_maps=([],[]),
        )
    params=sorted(set([Fraction(i,len(c0)) for i in range(len(c0))] + [Fraction(i,len(c1)) for i in range(len(c1))]))
    boundary=[("ref_boundary",str(p)) for p in params]
    center=("ref_center",0)
    triangles=tuple((center,boundary[i],boundary[(i+1)%len(boundary)]) for i in range(len(boundary)))
    product=compile_product(triangles)
    status="PASS_WITH_BOUNDARY_HOMEOMORPHISM" if product.status=="PASS" else "FAIL_CERTIFICATE"
    return PolygonProductResult(
        status=status,
        reason="certified abstract mapping cylinder; geometric correspondence is caller-certified" if status.startswith("PASS") else product.reason,
        input_sizes=(len(c0),len(c1)),
        common_boundary_vertices=len(params),
        endpoint_steiner_vertices=(len(params)-len(c0),len(params)-len(c1)),
        reference_triangles=triangles, product=product,
        endpoint_parameter_maps=(boundary_parameter_map(c0,params),boundary_parameter_map(c1,params)),
    )

@dataclass
class RegistryNormalizedResult:
    status: str
    reason: str
    removed_duplicate_triangle_occurrences: int
    normalized: NormalizedCompileResult

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "removed_duplicate_triangle_occurrences": self.removed_duplicate_triangle_occurrences,
            "normalized": self.normalized.to_dict(),
        }


def compile_registry_normalized_same_side(
    polygon_cycles: Sequence[Sequence[Vertex]], apex_prefix: Vertex = "registry_apex"
) -> RegistryNormalizedResult:
    """Common-refinement multiplicity cancellation followed by manifoldization.

    Exact duplicate 2-simplices are removed *entirely*, modeling an internal
    coincident interface. This operation is valid only when a global
    occupancy-side/incidence registry certifies that the copies are not two
    intentionally distinct surface sheets.
    """
    loops: List[Tuple[Vertex, ...]] = []
    dropped = 0
    for cycle in polygon_cycles:
        for loop in split_closed_walk_at_repeats(cycle):
            if len(loop) < 3 or len(set(loop)) < 3:
                dropped += 1
            else:
                loops.append(loop)
    triangles: List[Triangle] = []
    for loop_id, loop in enumerate(loops):
        triangles.extend(triangulate_polygon_cell(loop, ("registry", loop_id)))
    if not triangles:
        normalized = compile_normalized_same_side(polygon_cycles, apex_prefix)
        return RegistryNormalizedResult(
            status=normalized.status,
            reason=normalized.reason,
            removed_duplicate_triangle_occurrences=0,
            normalized=normalized,
        )

    groups: Dict[Tuple[Vertex, ...], List[Triangle]] = defaultdict(list)
    for tri in triangles:
        groups[canonical_simplex(tri)].append(tri)
    retained: List[Triangle] = []
    removed = 0
    for occurrences in groups.values():
        if len(occurrences) > 1:
            removed += len(occurrences)
        else:
            retained.append(occurrences[0])

    if not retained:
        empty = NormalizedCompileResult(
            status="PASS_INTERNAL_MULTIPLICITY_CANCELLED",
            reason="all 2D carriers were certified internal coincident interfaces",
            input_polygon_count=len(polygon_cycles),
            simple_loop_count=len(loops),
            dropped_lower_dimensional_cycles=dropped,
            vertex_splits=0,
            raw_surface_audit=audit_surface(triangles),
            normalized_surface_audit=None,
            component_audits=[],
            component_kernels=[],
        )
        return RegistryNormalizedResult(
            status=empty.status,
            reason=empty.reason,
            removed_duplicate_triangle_occurrences=removed,
            normalized=empty,
        )

    raw_after = audit_surface(retained)
    if raw_after.nonmanifold_edges or raw_after.duplicate_triangles or not raw_after.orientable:
        unsupported = NormalizedCompileResult(
            status="UNSUPPORTED",
            reason=f"registry reduction still leaves {raw_after.topology_class}",
            input_polygon_count=len(polygon_cycles),
            simple_loop_count=len(loops),
            dropped_lower_dimensional_cycles=dropped,
            vertex_splits=0,
            raw_surface_audit=raw_after,
            normalized_surface_audit=raw_after,
            component_audits=[raw_after],
            component_kernels=[],
        )
        return RegistryNormalizedResult(
            status="UNSUPPORTED", reason=unsupported.reason,
            removed_duplicate_triangle_occurrences=removed, normalized=unsupported,
        )

    split_tris, split_count = split_nonmanifold_vertex_links(retained)
    components = triangle_components_by_edges(split_tris)
    audits=[]; kernels=[]
    for component_id, ids in enumerate(components):
        comp=tuple(split_tris[i] for i in ids)
        audits.append(audit_surface(comp))
        kernels.append(compile_star_cone(comp, apex=(apex_prefix,component_id)))
    passed=all(k.status=="PASS" for k in kernels)
    if passed:
        status = "PASS_WITH_REGISTRY_CANCELLATION" if removed else (
            "PASS_WITH_COMPONENT_SPLIT" if split_count else (
                "PASS_WITH_EVENT_STAR_COMPONENT_SPLIT" if len(components) > 1 else (
                    "PASS_WITH_LOWER_DIMENSIONAL_PRUNING" if dropped else "PASS_DIRECT"
                )
            )
        )
        reason="registry-normalized components compile to certified cones"
    else:
        status="UNSUPPORTED"; reason="a registry-normalized component is not cone-supported"
    normalized=NormalizedCompileResult(
        status=status,reason=reason,input_polygon_count=len(polygon_cycles),
        simple_loop_count=len(loops),dropped_lower_dimensional_cycles=dropped,
        vertex_splits=split_count,raw_surface_audit=raw_after,
        normalized_surface_audit=audit_surface(split_tris),
        component_audits=audits,component_kernels=kernels,
    )
    return RegistryNormalizedResult(
        status=status,reason=reason,
        removed_duplicate_triangle_occurrences=removed,normalized=normalized,
    )


def triangulate_polygon_cell(cycle: Sequence[Vertex], face_id: Vertex) -> Tuple[Triangle, ...]:
    """Functorial polygon-cell subdivision with one private face center.

    Triangles are kept as-is. Every n-gon with n>3 is coned to a private
    face-center vertex. Unlike a vertex-order fan, this construction is
    invariant under relabeling and does not introduce a shared diagonal that
    another overlapping polygon might interpret differently.
    """
    cycle=clean_cycle(cycle)
    if len(cycle)<3 or len(set(cycle))!=len(cycle):
        raise ValueError("polygon cell must have a simple boundary cycle")
    if len(cycle)==3:
        return (tuple(cycle),)  # type: ignore[return-value]
    center=("face_center",face_id)
    return tuple((center,cycle[i],cycle[(i+1)%len(cycle)]) for i in range(len(cycle)))


def triangulate_polygon_cells(cycles: Sequence[Sequence[Vertex]]) -> Tuple[Triangle, ...]:
    triangles: List[Triangle]=[]
    for i,cycle in enumerate(cycles):
        triangles.extend(triangulate_polygon_cell(cycle,i))
    return tuple(triangles)
