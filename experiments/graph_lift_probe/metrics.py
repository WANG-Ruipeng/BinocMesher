"""Surface metrics for the bounded graph-lift probe.

Distances are floating-point point-to-triangle distances, never distances to
triangle vertices.  KD-tree pruning uses the triangle inequality: a triangle
closer than the current upper bound has its centroid within that bound plus
the triangle's maximum vertex-to-centroid radius.  Roundoff padding makes the
broad phase conservative in ordinary float64 arithmetic; this is not an
interval-arithmetic geometric certificate.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


_EPS = np.finfo(np.float64).eps


def _points(values, dimension, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != dimension:
        raise ValueError(f"{name} must have shape (N, {dimension})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _triangle_data(triangles):
    tri = np.array(triangles, dtype=np.float64, copy=True)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or len(tri) == 0:
        raise ValueError("triangles must have nonempty shape (T, 3, 3)")
    if not np.all(np.isfinite(tri)):
        raise ValueError("triangles contain nonfinite values")
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(edge1, edge2)
    double_area = np.linalg.norm(cross, axis=1)
    edge_scale2 = np.maximum.reduce(
        [np.einsum("ij,ij->i", edge1, edge1),
         np.einsum("ij,ij->i", edge2, edge2),
         np.einsum("ij,ij->i", edge2 - edge1, edge2 - edge1)]
    )
    if (not np.all(np.isfinite(double_area))
            or not np.all(np.isfinite(edge_scale2))
            or np.any(double_area <= 64.0 * _EPS * edge_scale2)):
        raise ValueError("degenerate or numerically singular triangles")
    normals = cross / double_area[:, None]
    return tri, cross, double_area, normals


def _paired_squared_distance(points, triangles):
    """One point per triangle; orthogonal projection or clipped edge minimum."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, points - a
    normal = np.cross(ab, ac)
    normal2 = np.einsum("ij,ij->i", normal, normal)
    u = np.einsum("ij,ij->i", np.cross(ap, ac), normal) / normal2
    v = np.einsum("ij,ij->i", np.cross(ab, ap), normal) / normal2
    inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    normal_dot = np.einsum("ij,ij->i", ap, normal)
    plane_d2 = normal_dot * normal_dot / normal2
    best_d2 = np.full(len(points), np.inf, dtype=np.float64)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        offset = points - start
        edge2 = np.einsum("ij,ij->i", edge, edge)
        fraction = np.clip(np.einsum("ij,ij->i", offset, edge) / edge2,
                           0.0, 1.0)
        residual = offset - fraction[:, None] * edge
        d2 = np.einsum("ij,ij->i", residual, residual)
        best_d2 = np.minimum(best_d2, d2)
    best_d2 = np.where(inside, np.minimum(best_d2, plane_d2), best_d2)
    if not np.all(np.isfinite(best_d2)) or np.any(best_d2 < 0.0):
        raise ValueError("point-to-triangle computation produced invalid distances")
    return best_d2


class TriangleSurface:
    """Immutable nondegenerate triangle soup with conservative nearest lookup."""

    def __init__(self, triangles):
        tri, _, _, normals = _triangle_data(triangles)
        self.triangles = tri
        self.normals = normals
        self.triangles.setflags(write=False)
        self.normals.setflags(write=False)
        self._centroids = tri.mean(axis=1)
        self._max_radius = float(np.linalg.norm(
            tri - self._centroids[:, None, :], axis=2).max())
        self._coordinate_scale = float(np.abs(tri).max())
        self._tree = cKDTree(self._centroids)

    def nearest(self, points):
        """Return distances and nearest face IDs; ties prefer the smallest ID.

        Up to 128 point neighborhoods are queried at once.  Ragged candidate
        pairs are evaluated in blocks capped near 200,000 pairs, avoiding an
        all-points-by-all-triangles allocation.
        """
        points = _points(points, 3, "points")
        count = len(points)
        if count == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
        result_d2 = np.empty(count, dtype=np.float64)
        result_face = np.empty(count, dtype=np.int64)
        for begin in range(0, count, 128):
            block = points[begin:begin + 128]
            _, seed_ids = self._tree.query(block, k=1, workers=1)
            seed_ids = np.asarray(seed_ids, dtype=np.int64)
            upper = np.sqrt(_paired_squared_distance(
                block, self.triangles[seed_ids]))
            coordinate_scale = np.maximum(np.abs(block).max(axis=1),
                                          self._coordinate_scale)
            pad = 128.0 * _EPS * (1.0 + coordinate_scale + upper
                                 + self._max_radius)
            radii = np.nextafter(upper + self._max_radius + pad, np.inf)
            neighborhoods = self._tree.query_ball_point(
                block, radii, workers=1, return_sorted=True)
            lengths = np.fromiter((len(ids) for ids in neighborhoods),
                                  dtype=np.int64, count=len(block))
            if np.any(lengths == 0):
                raise RuntimeError("conservative broad phase lost its seed face")
            local_begin = 0
            while local_begin < len(block):
                local_end = local_begin
                pair_count = 0
                while local_end < len(block):
                    next_count = pair_count + int(lengths[local_end])
                    if pair_count and next_count > 200_000:
                        break
                    pair_count = next_count
                    local_end += 1
                block_lengths = lengths[local_begin:local_end]
                face_ids = np.concatenate(neighborhoods[local_begin:local_end])
                face_ids = np.asarray(face_ids, dtype=np.int64)
                local_ids = np.repeat(np.arange(local_begin, local_end),
                                      block_lengths)
                d2 = _paired_squared_distance(block[local_ids],
                                              self.triangles[face_ids])
                starts = np.concatenate(([0], np.cumsum(block_lengths[:-1])))
                minimum = np.minimum.reduceat(d2, starts)
                repeated_minimum = np.repeat(minimum, block_lengths)
                tied_ids = np.where(d2 == repeated_minimum,
                                    face_ids, len(self.triangles))
                nearest_ids = np.minimum.reduceat(tied_ids, starts)
                target = slice(begin + local_begin, begin + local_end)
                result_d2[target] = minimum
                result_face[target] = nearest_ids
                local_begin = local_end
        return np.sqrt(result_d2), result_face


def pl_lift(uv_points, triangles):
    """Lift shared parameter points onto a positive-oriented PL graph.

    Returns XYZ points, upward unit face normals, and the surface-area
    Jacobian relative to du dv.  Input topology/overlap is checked by the
    experiment's mesh guards; this lookup rejects uncovered sample points.
    On shared edges the lowest face ID supplies the normal.
    """
    uv = _points(uv_points, 2, "uv_points")
    tri, cross, double_area, normals = _triangle_data(triangles)
    a = tri[:, 0, :2]
    edge1 = tri[:, 1, :2] - a
    edge2 = tri[:, 2, :2] - a
    determinant = cross[:, 2]
    xy_scale2 = np.maximum.reduce(
        [np.einsum("ij,ij->i", edge1, edge1),
         np.einsum("ij,ij->i", edge2, edge2),
         np.einsum("ij,ij->i", edge2 - edge1, edge2 - edge1)]
    )
    if np.any(determinant <= 64.0 * _EPS * xy_scale2):
        raise ValueError("PL graph requires positive, nondegenerate XY faces")
    xyz = np.empty((len(uv), 3), dtype=np.float64)
    xyz[:, :2] = uv
    lifted_normals = np.empty((len(uv), 3), dtype=np.float64)
    jacobian = np.empty(len(uv), dtype=np.float64)
    assigned = np.zeros(len(uv), dtype=bool)
    for face in range(len(tri)):
        remaining = np.flatnonzero(~assigned)
        if not len(remaining):
            break
        offset = uv[remaining] - a[face]
        b = (offset[:, 0] * edge2[face, 1]
             - offset[:, 1] * edge2[face, 0]) / determinant[face]
        c = (edge1[face, 0] * offset[:, 1]
             - edge1[face, 1] * offset[:, 0]) / determinant[face]
        tolerance = 128.0 * _EPS * (1.0 + xy_scale2[face] / determinant[face])
        inside = (b >= -tolerance) & (c >= -tolerance) & (b + c <= 1 + tolerance)
        chosen = remaining[inside]
        xyz[chosen, 2] = (tri[face, 0, 2]
                          + b[inside] * (tri[face, 1, 2] - tri[face, 0, 2])
                          + c[inside] * (tri[face, 2, 2] - tri[face, 0, 2]))
        lifted_normals[chosen] = normals[face]
        jacobian[chosen] = double_area[face] / determinant[face]
        assigned[chosen] = True
    if not np.all(assigned):
        raise ValueError(f"{int((~assigned).sum())} parameter points lie outside the PL mesh")
    if (not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(lifted_normals))
            or not np.all(np.isfinite(jacobian))):
        raise ValueError("PL lifting produced nonfinite outputs")
    return xyz, lifted_normals, jacobian


def weighted_stats(values, weights):
    """Area-weighted mean, inverse-ECDF P95, and positive-weight sampled max."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or len(values) == 0:
        raise ValueError("values and weights must be matching nonempty vectors")
    if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0) or not np.any(weights > 0.0)):
        raise ValueError("invalid values or weights")
    positive = weights > 0.0
    values, weights = values[positive], weights[positive]
    total = float(weights.sum())
    if not np.isfinite(total):
        raise ValueError("nonfinite total weight")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order] / total)
    index = min(int(np.searchsorted(cumulative, 0.95, side="left")),
                len(order) - 1)
    result = {"mean": float(np.dot(values, weights / total)),
              "p95": float(values[order[index]]),
              "sampled_max": float(values.max())}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("weighted statistics produced nonfinite values")
    return result
