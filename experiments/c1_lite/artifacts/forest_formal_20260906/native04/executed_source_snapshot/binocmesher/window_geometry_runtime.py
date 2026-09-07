"""Exact, conservative checks of one *actual* binary32 graph-fan replacement.

This is a query-time checker, not a continuous-time certificate.  Positions
come from the real slicer; common features are actual global vertex IDs.
No shared neighbour is skipped.  Failure to separate is UNKNOWN, never safe.
"""
from collections import Counter, defaultdict
from fractions import Fraction as F

import numpy as np


def _p(point):
    return tuple(F.from_float(float(x)) for x in point)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _cross2(a, b, axes=(0, 1)):
    i, j = axes
    return a[i]*b[j]-a[j]*b[i]


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _edges(face):
    return [(face[i], face[(i+1) % len(face)]) for i in range(len(face))]


def _canonical(face):
    face = tuple(int(x) for x in face)
    return min(face[i:]+face[:i] for i in range(3))


def interface_check(faces, removed, cycle):
    """Original disk and retained collar in actual global-index space."""
    original = [tuple(int(x) for x in faces[i]) for i in removed]
    if len(original) != 2 or len(set(cycle)) != 4:
        return 'Expected two source faces and four distinct boundary IDs.'
    if any(len(set(f)) != 3 or not set(f) <= set(cycle) for f in original):
        return 'Original faces are not triangles on the declared boundary.'
    directed = Counter(e for f in original for e in _edges(f))
    boundary = Counter(_edges(cycle))
    if any(directed[e] != 1 for e in boundary):
        return 'Original oriented source boundary differs.'
    rest = directed-boundary
    if (len(rest) != 2 or sum(rest.values()) != 2 or
            any(n != 1 or rest[b, a] != 1 for (a, b), n in rest.items())):
        return 'Original source faces do not have one internal diagonal.'
    diagonal = set(next(iter(rest)))
    edges, links, seen = Counter(), defaultdict(list), set()
    for index, row in enumerate(faces):
        if index in removed:
            continue
        face = tuple(int(x) for x in row)
        shared = set(face) & set(cycle)
        if not shared:
            continue
        if len(set(face)) != 3 or diagonal <= set(face):
            return 'Repeated interface ID or retained reuse of source diagonal.'
        key = _canonical(face)
        if key in seen:
            return 'Duplicate oriented interface face.'
        seen.add(key)
        edges.update(_edges(face))
        for v in shared:
            j = face.index(v)
            links[v].append((face[(j+1) % 3], face[(j+2) % 3]))
    if any(edges[a, b] != 0 or edges[b, a] != 1 for a, b in _edges(cycle)):
        return 'Retained boundary edge attachment is not exactly opposite.'
    for i, v in enumerate(cycle):
        pairs = links[v]
        degree = Counter(x for pair in pairs for x in pair)
        incoming, outgoing = Counter(b for a, b in pairs), Counter(a for a, b in pairs)
        start, end = cycle[i-1], cycle[(i+1) % 4]
        if set(x for x, n in degree.items() if n == 1) != {start, end}:
            return 'Retained vertex link has wrong endpoints.'
        if any(n != (1 if x in (start, end) else 2) for x, n in degree.items()):
            return 'Retained vertex link has wrong degrees.'
        if any(incoming[x] != (0 if x == start else 1) or
               outgoing[x] != (0 if x == end else 1) for x in degree):
            return 'Retained vertex link has incompatible orientation.'
        adjacency = defaultdict(set)
        for a, b in pairs:
            adjacency[a].add(b); adjacency[b].add(a)
        reached, pending = set(), [start]
        while pending:
            x = pending.pop()
            if x not in reached:
                reached.add(x); pending.extend(adjacency[x]-reached)
        if reached != set(degree):
            return 'Retained vertex link is disconnected.'
    return None


def _normals():
    result = {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for k in range(1, 9):
            for x, y in ((1, k), (k, 1)):
                for sign in (-1, 1):
                    n = [0, 0, 0]; n[a], n[b] = x, sign*y
                    result.add(tuple(n))
    return tuple(sorted(result))


NORMALS = _normals()


def _contact(boundary, center, cycle, triangle, ids, orientation):
    shared = set(ids) & set(cycle)
    patch = (*boundary, center)
    if shared and not any(_cross(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))):
        return None, 'Degenerate retained triangle touches the interface.'
    # A plane through an actual boundary edge has structural zero at both
    # common endpoints, even when the source projection was changed by rounding.
    for i in range(4):
        j = (i+1) % 4
        edge_ids = {cycle[i], cycle[j]}
        if not shared <= edge_ids:
            continue
        origin, edge = boundary[i], _sub(boundary[j], boundary[i])
        for axes, tilt in (((1, 2), 0), *((axes, tilt) for axes in ((0, 2), (1, 2))
                                                  for tilt in (1, -1, 2, -2, 4, -4, 8, -8))):
            def value(point):
                d = _sub(point, origin)
                return orientation*_cross2(edge, d)+tilt*_cross2(edge, d, axes)
            if not all(value(point) > 0 for k, point in enumerate(patch) if k not in (i, j)):
                continue
            if all((point == boundary[cycle.index(vid)] if vid in shared else value(point) < 0)
                   for point, vid in zip(triangle, ids)):
                return ('RELATIVE_EDGE_PLANE' if shared else 'RELATIVE_EMPTY_PLANE'), None
    if len(shared) == 1:
        v = next(iter(shared)); i = cycle.index(v); origin = boundary[i]
        previous = _sub(origin, boundary[i-1]); following = _sub(boundary[(i+1) % 4], origin)
        for wa in range(1, 17):
            for wb in range(1, 17):
                def value(point):
                    d = _sub(point, origin)
                    return orientation*(wa*_cross2(previous, d)+wb*_cross2(following, d))
                if (all(value(point) > 0 for k, point in enumerate(patch) if k != i) and
                        all(point == origin if vid == v else value(point) < 0
                            for point, vid in zip(triangle, ids))):
                    return 'RELATIVE_VERTEX_PLANE', None
    if not shared:
        for normal in NORMALS:
            a = [_dot(normal, p) for p in patch]
            b = [_dot(normal, p) for p in triangle]
            if max(a) < min(b) or max(b) < min(a):
                return 'STRICT_FIXED_PLANE', None
    return None, 'No sufficient exact separator for the actual retained face.'


def check_actual_patch(vertices, faces, removed, cycle, center):
    """Validate the complete actual retained mesh against a four-triangle fan.

    Remote existing degeneracies may remain; this proves only local interface
    validity and no new fan/retained contact beyond common actual features.
    """
    report = {'status': 'UNKNOWN', 'scope': 'ONE_ACTUAL_QUERY_ONLY',
              'all_time_certificate': False, 'retained_faces_checked': 0,
              'certificate_counts': {}, 'reason': None}
    try:
        v, f = np.asarray(vertices), np.asarray(faces)
        if (v.ndim != 2 or v.shape[1] != 3 or not np.all(np.isfinite(v)) or
                f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer) or
                np.any(f < 0) or np.any(f >= len(v)) or len(removed) != 2 or
                any(i < 0 or i >= len(f) for i in removed) or len(cycle) != 4 or
                any(i < 0 or i >= len(v) for i in cycle) or not np.all(np.isfinite(center))):
            raise ValueError('Malformed actual mesh or patch indices.')
        # This public checker is specifically the spaceT=binary32 path, even
        # though slicing_output exports those exact values in a double array.
        if not np.array_equal(v, v.astype(np.float32).astype(v.dtype)):
            raise ValueError('Actual baseline coordinates are not binary32 values.')
        if not np.array_equal(np.asarray(center), np.asarray(center, dtype=np.float32).astype(float)):
            raise ValueError('Center is not the declared binary32 representation.')
        boundary = tuple(_p(v[i]) for i in cycle); c = _p(center)
        turns = [_cross2(_sub(boundary[(i+1) % 4], boundary[i]),
                         _sub(boundary[(i+2) % 4], boundary[(i+1) % 4])) for i in range(4)]
        orientation = 1 if turns[0] > 0 else -1
        if not all(orientation*x > 0 for x in turns):
            raise ValueError('Actual boundary is not a strictly convex XY graph.')
        if not all(orientation*_cross2(_sub(boundary[(i+1) % 4], boundary[i]), _sub(c, boundary[i])) > 0
                   for i in range(4)):
            raise ValueError('Actual center is not strictly inside the projected boundary.')
        error = interface_check(f, set(removed), tuple(cycle))
        if error:
            raise ValueError(error)
        patch = np.asarray([v[i] for i in cycle]+[center])
        low, high = patch.min(axis=0), patch.max(axis=0)
        counts = Counter()
        for index, row in enumerate(f):
            if index in removed:
                continue
            coords = v[row]
            # Only exact order comparisons of represented coordinates. No
            # rounded cross products, tolerances, or signed-volume shortcuts.
            if np.any(coords.min(axis=0) > high) or np.any(coords.max(axis=0) < low):
                kind = 'STRICT_ACTUAL_AABB'
            else:
                kind, error = _contact(boundary, c, tuple(cycle), tuple(_p(p) for p in coords),
                                       tuple(int(x) for x in row), orientation)
                if kind is None:
                    report.update(reason=error, failing_retained_face=index,
                                  certificate_counts=dict(counts))
                    return report
            counts[kind] += 1; report['retained_faces_checked'] += 1
        report.update(status='PASS', certificate_counts=dict(counts),
                      reason='Actual graph fan, interface and all retained contacts checked exactly.')
    except (ValueError, TypeError, IndexError, OverflowError) as error:
        report.update(status='REJECT', reason=str(error))
    return report
