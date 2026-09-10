"""Oracle-free connectivity controls for small, fixed-boundary XY graph meshes.

Predicates are floating-point with a fixed tolerance after global XY scaling.
These are bounded experimental controls, not an exact-predicate production kernel.
Boundary vertices must be listed first in CCW order for retriangulation/enumeration.
"""
from collections import deque

import numpy as np


EPS = 1e-12


def canonical_faces(faces):
    """Connectivity signature, ignoring face order and orientation."""
    return tuple(sorted(tuple(sorted(map(int, face))) for face in faces))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _orient(a, b, c):
    ab, ac = b-a, c-a
    return float(ab[0]*ac[1]-ab[1]*ac[0])


def _edge_map(faces):
    edges = {}
    for index, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(a), int(b))))
            edges.setdefault(edge, []).append(index)
    return edges


def _prepare(vertices, faces, boundary_count=None):
    vertices = np.asarray(vertices, dtype=float)
    raw_faces = np.asarray(faces)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3 and
             len(vertices) >= 3 and np.isfinite(vertices).all(), 'invalid vertices')
    _require(raw_faces.ndim == 2 and raw_faces.shape[1] == 3 and
             len(raw_faces) > 0 and np.isfinite(raw_faces).all() and
             np.equal(raw_faces, np.floor(raw_faces)).all(), 'invalid faces')
    faces = raw_faces.astype(int, copy=True)
    _require(np.all(faces >= 0) and np.all(faces < len(vertices)), 'invalid face indices')
    _require(set(faces.ravel()) == set(range(len(vertices))), 'unused vertex')
    _require(len(set(canonical_faces(faces))) == len(faces), 'duplicate face')
    scale = float(np.ptp(vertices[:, :2], axis=0).max())
    _require(scale > 0, 'degenerate XY extent')
    origin = vertices[:, :2].min(axis=0)
    xy = (vertices[:, :2]-origin)/scale
    for face in faces:
        _require(_orient(*xy[face]) > EPS, 'degenerate or reversed parameter face')
    edges = _edge_map(faces)
    _require(all(len(users) in (1, 2) for users in edges.values()), 'nonmanifold edge')
    for (a, b), users in edges.items():
        if len(users) == 2:
            c, d = [next(int(v) for v in faces[i] if v not in (a, b)) for i in users]
            _require(_orient(xy[a], xy[b], xy[c]) *
                     _orient(xy[a], xy[b], xy[d]) < 0,
                     'inconsistent interior adjacency')
    _require(len(vertices)-len(edges)+len(faces) == 1, 'not a disk mesh')
    if boundary_count is not None:
        m = int(boundary_count)
        _require(m == boundary_count and 3 <= m <= len(vertices), 'invalid boundary count')
        expected = {tuple(sorted((i, (i+1) % m))) for i in range(m)}
        actual = {edge for edge, users in edges.items() if len(users) == 1}
        _require(actual == expected, 'boundary changed')
        _require(len(faces) == m+2*(len(vertices)-m)-2, 'fixed-boundary face budget changed')
    return vertices, faces, xy, edges, origin, scale


def _positive_face(face, xy):
    a, b, c = map(int, face)
    area = _orient(xy[a], xy[b], xy[c])
    _require(abs(area) > EPS, 'degenerate replacement face')
    return (a, b, c) if area > 0 else (a, c, b)


def insert_point(vertices, faces, point_xyz):
    """Insert supplied geometry, returning faces with new ID len(vertices).

Interior triangle: 1 -> 3. Shared interior edge: 2 -> 4. Boundary,
duplicate, outside, nonfinite and numerically degenerate points are refused.
No oracle is called and input arrays are not changed.
"""
    vertices, faces, xy, edges, origin, scale = _prepare(vertices, faces)
    point = np.asarray(point_xyz, dtype=float)
    _require(point.shape == (3,) and np.isfinite(point).all(), 'invalid inserted point')
    q = (point[:2]-origin)/scale
    _require(np.all(np.linalg.norm(xy-q, axis=1) > EPS), 'duplicate inserted vertex')
    containing, hits = [], set()
    for index, face in enumerate(faces):
        signs = [_orient(xy[a], xy[b], q) for a, b in zip(face, np.roll(face, -1))]
        if min(signs) >= -EPS:
            containing.append(index)
            for j, value in enumerate(signs):
                if abs(value) <= EPS:
                    hits.add(tuple(sorted((int(face[j]), int(face[(j+1) % 3])))))
    _require(containing, 'inserted point outside mesh')
    _require(not any(len(edges[e]) == 1 for e in hits), 'boundary insertion refused')
    _require(len(hits) <= 1, 'ambiguous near-vertex insertion')
    new_id = len(vertices)
    all_xy = np.vstack((xy, q))
    if hits:
        edge = next(iter(hits))
        selected = edges[edge]
        _require(set(containing).issubset(selected), 'ambiguous edge insertion')
        replacements = []
        a, b = edge
        for index in selected:
            c = next(int(v) for v in faces[index] if v not in edge)
            replacements.extend(((a, new_id, c), (new_id, b, c)))
    else:
        _require(len(containing) == 1, 'overlapping containing triangles')
        selected = containing
        a, b, c = faces[selected[0]]
        replacements = ((a, b, new_id), (b, c, new_id), (c, a, new_id))
    kept = [tuple(face) for i, face in enumerate(faces) if i not in selected]
    result = np.asarray(kept+[_positive_face(face, all_xy) for face in replacements], dtype=int)
    _prepare(np.vstack((vertices, point)), result)
    return result


def _flip_candidates(faces, xy):
    """Yield legal strictly-convex interior 2 -> 2 flips in stable edge order."""
    edges = _edge_map(faces)
    for (a, b), users in sorted(edges.items()):
        if len(users) != 2:
            continue
        i, j = users
        c, d = [next(int(v) for v in faces[k] if v not in (a, b)) for k in users]
        if _orient(xy[a], xy[b], xy[c]) < 0:
            c, d = d, c
        alternative = tuple(sorted((c, d)))
        if alternative in edges:
            continue
        side_a, side_b = _orient(xy[c], xy[d], xy[a]), _orient(xy[c], xy[d], xy[b])
        if not ((side_a > EPS and side_b < -EPS) or (side_b > EPS and side_a < -EPS)):
            continue
        replacements = (_positive_face((c, d, a), xy), _positive_face((d, c, b), xy))
        yield (a, b), alternative, (i, j), replacements, (a, b, c, d)


def _apply_flip(faces, users, replacements):
    result = faces.copy()
    result[list(users)] = replacements
    return result


def _incircle(xy, ids):
    a, b, c, d = xy[list(ids)]
    a, b, c = a-d, b-d, c-d
    return float(np.dot(a, a)*(b[0]*c[1]-b[1]*c[0]) -
                 np.dot(b, b)*(a[0]*c[1]-a[1]*c[0]) +
                 np.dot(c, c)*(a[0]*b[1]-a[1]*b[0]))


def retriangulate_delaunay(vertices, faces, boundary_count):
    """Deterministic XY Delaunay flips; retain every vertex and boundary edge.

Near-cocircular alternatives use the lexicographically smaller diagonal.
Explicit state/cost limits fail closed instead of returning a partial result.
"""
    _, faces, xy, _, _, _ = _prepare(vertices, faces, boundary_count)
    seen = {canonical_faces(faces)}
    limit = max(100, 32*len(faces)*len(faces))
    stats = {'flips': 0, 'strict_flips': 0, 'cocircular_tie_flips': 0,
             'predicate_tests': 0, 'predicate_epsilon': EPS, 'oracle_queries': 0}
    for _ in range(limit):
        changed = False
        for edge, alternative, users, replacements, ids in _flip_candidates(faces, xy):
            value = _incircle(xy, ids)
            stats['predicate_tests'] += 1
            strict = value > EPS
            tie = abs(value) <= EPS and alternative < edge
            if not (strict or tie):
                continue
            faces = _apply_flip(faces, users, replacements)
            key = canonical_faces(faces)
            _require(key not in seen, 'Delaunay flip cycle; result refused')
            seen.add(key)
            stats['flips'] += 1
            stats['strict_flips' if strict else 'cocircular_tie_flips'] += 1
            changed = True
            break
        if not changed:
            _prepare(vertices, faces, boundary_count)
            stats['canonical_faces'] = canonical_faces(faces)
            return faces, stats
    raise ValueError('Delaunay flip budget exhausted; result refused')


def enumerate_triangulations(vertices, start_faces, boundary_count, max_states=4096):
    """Bounded flip-graph enumeration, diagnostic only (no geometry oracle).

Returns every reachable legal straight-line triangulation. For nondegenerate
planar point sets this is the ordinary connected triangulation flip graph.
Collinear subsets may restrict the graph; no universal completeness is claimed.
Raises if the state cap is exceeded rather than selecting from a partial set.
"""
    _, faces, xy, _, _, _ = _prepare(vertices, start_faces, boundary_count)
    _require(max_states >= 1, 'invalid enumeration state budget')
    initial = canonical_faces(faces)
    states, pending = {initial: faces}, deque([initial])
    transitions = 0
    while pending:
        key = pending.popleft()
        for _, _, users, replacements, _ in _flip_candidates(states[key], xy):
            transitions += 1
            candidate = _apply_flip(states[key], users, replacements)
            new_key = canonical_faces(candidate)
            if new_key in states:
                continue
            _require(len(states) < max_states, 'enumeration state budget exhausted; result refused')
            states[new_key] = candidate
            pending.append(new_key)
    return [states[key] for key in sorted(states)], {
        'states': len(states), 'directed_legal_flips': transitions,
        'state_budget': int(max_states), 'complete_reachable_graph': True,
        'oracle_queries': 0, 'predicate_epsilon': EPS}
