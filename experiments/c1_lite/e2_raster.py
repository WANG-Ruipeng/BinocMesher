"""Small deterministic CPU pinhole rasterizer for scientific geometry probes.

OpenCV camera coordinates: +x right, +y down, +z forward. Pixel centers are
(x+.5, y+.5). No backface culling, in-view tags, RGB, shadows, or mesh mutation.
Normals are oriented WORLD-space flat geometric normals of the original face.
"""
from __future__ import annotations

import math
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _intersection(a, b, near):
    # Same shared edge gives bit-identical clipping coordinates regardless of
    # which adjacent face traverses it first or in which direction.
    if tuple(a) > tuple(b):
        a, b = b, a
    if a[2] == near:
        return a.copy()
    if b[2] == near:
        return b.copy()
    t = (near-a[2])/(b[2]-a[2])
    point = a+t*(b-a)
    point[2] = near
    return point


def _clip_near(triangle, near):
    polygon = []
    previous = triangle[-1]
    previous_inside = previous[2] >= near
    for current in triangle:
        current_inside = current[2] >= near
        if current_inside != previous_inside:
            polygon.append(_intersection(previous, current, near))
        if current_inside:
            polygon.append(current.copy())
        previous, previous_inside = current, current_inside
    unique = []
    for point in polygon:
        if not unique or not np.array_equal(point, unique[-1]):
            unique.append(point)
    if len(unique) > 1 and np.array_equal(unique[0], unique[-1]):
        unique.pop()
    if unique:
        start = min(range(len(unique)), key=lambda i: tuple(unique[i]))
        unique = unique[start:]+unique[:start]
    return unique


def _edge(a, b):
    # Reversing an edge negates all coefficients; shared-edge coverage does
    # not depend on a face's third vertex or subtraction around a pixel.
    return (a[1]-b[1], b[0]-a[0], a[0]*b[1]-a[1]*b[0])


def _top_left(a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    return dy < 0 or (dy == 0 and dx > 0)


def _project(camera_points, K):
    projected = camera_points @ K.T
    return projected[:, :2]/camera_points[:, 2, None]


def rasterize(vertices, faces, c2w, K, width=640, height=360, near=1e-6):
    """Return depth, normals, mask, face_id and deterministic integer stats.

    depth is camera Z (not ray length), float64, NaN on background. normals
    has shape (H,W,3), float64, NaN on background. face_id is the ORIGINAL
    face-array index, int64, -1 on background, including for clipped triangles.
    Depth ties use strict less-than: the first input face wins an exact tie.
    Numerically equivalent alternative triangulations may differ by roundoff;
    compare depths with a declared numerical tolerance, not face-ID hashes.
    """
    _require(isinstance(width, (int, np.integer)) and not isinstance(width, (bool, np.bool_))
             and isinstance(height, (int, np.integer)) and not isinstance(height, (bool, np.bool_)),
             'Width and height must be integers.')
    width, height = int(width), int(height)
    _require(0 < width <= 8192 and 0 < height <= 8192 and width*height <= 16_777_216,
             'Image dimensions exceed the bounded rasterizer profile.')
    near = float(near)
    _require(np.isfinite(near) and near >= 1e-6, 'Near plane must be finite and at least 1e-6.')
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    pose, intrinsics = np.asarray(c2w, dtype=np.float64), np.asarray(K, dtype=np.float64)
    _require(v.ndim == 2 and v.shape[1] == 3 and np.all(np.isfinite(v)),
             'Vertices must be a finite (n,3) array.')
    _require(f.ndim == 2 and f.shape[1] == 3 and f.dtype.kind in 'iu',
             'Faces must be an integer (m,3) array.')
    _require(not f.size or (int(f.min()) >= 0 and int(f.max()) < len(v)),
             'Face vertex index is out of range.')
    _require(pose.shape == (4, 4) and np.all(np.isfinite(pose)) and
             np.array_equal(pose[3], [0, 0, 0, 1]), 'c2w must be a finite affine 4x4 matrix.')
    _require(intrinsics.shape == (3, 3) and np.all(np.isfinite(intrinsics)) and
             np.array_equal(intrinsics[2], [0, 0, 1]) and
             abs(float(np.linalg.det(intrinsics[:2, :2]))) > 0,
             'K must be a finite nonsingular pinhole intrinsic matrix.')
    try:
        w2c = np.linalg.inv(pose)
    except np.linalg.LinAlgError as error:
        raise ValueError('c2w is singular.') from error
    camera = v @ w2c[:3, :3].T+w2c[:3, 3]
    _require(np.all(np.isfinite(camera)), 'Camera transform overflowed.')

    depth = np.full((height, width), np.inf, np.float64)
    normals = np.full((height, width, 3), np.nan, np.float64)
    face_id = np.full((height, width), -1, np.int64)
    stats = dict(input_vertices=len(v), input_faces=len(f), degenerate_faces=0,
                 behind_near_faces=0, near_clipped_faces=0,
                 viewport_rejected_faces=0, viewport_rejected_triangles=0,
                 projected_degenerate_triangles=0, raster_triangles=0,
                 depth_updates=0, covered_pixels=0)
    if not len(f):
        depth[:] = np.nan
        return dict(depth=depth, normals=normals, mask=face_id >= 0, face_id=face_id, stats=stats)

    # Canonical cyclic rotation preserves winding while making cyclic face
    # rotations compute exactly the same normals and screen edge coefficients.
    triangles = v[f]
    starts = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)[:, 0]
    ordered = np.take_along_axis(f, (starts[:, None]+np.arange(3)) % 3, axis=1)
    world = v[ordered]
    cross = np.cross(world[:, 1]-world[:, 0], world[:, 2]-world[:, 0])
    _require(np.all(np.isfinite(cross)), 'World triangle normal overflowed.')
    scale = np.max(np.abs(cross), axis=1)
    valid = scale > 0
    stats['degenerate_faces'] = int(np.count_nonzero(~valid))
    unit = np.zeros_like(cross)
    unit[valid] = cross[valid]/scale[valid, None]
    unit[valid] /= np.sqrt(np.sum(unit[valid]**2, axis=1))[:, None]
    camera_triangles = camera[ordered]
    front = camera_triangles[:, :, 2] >= near
    all_front, any_front = front.all(axis=1), front.any(axis=1)
    stats['behind_near_faces'] = int(np.count_nonzero(valid & ~any_front))
    stats['near_clipped_faces'] = int(np.count_nonzero(valid & any_front & ~all_front))
    candidates = valid & any_front
    screen = np.full((len(v), 2), np.nan)
    projected_vertices = camera[:, 2] >= near
    screen[projected_vertices] = _project(camera[projected_vertices], intrinsics)
    _require(np.all(np.isfinite(screen[projected_vertices])), 'Pinhole projection overflowed.')
    fully_visible_candidates = np.flatnonzero(valid & all_front)
    projected_triangles = screen[ordered[fully_visible_candidates]]
    lows, highs = projected_triangles.min(axis=1), projected_triangles.max(axis=1)
    outside = ((highs[:, 0] < .5) | (lows[:, 0] > width-.5) |
               (highs[:, 1] < .5) | (lows[:, 1] > height-.5))
    candidates[fully_visible_candidates[outside]] = False
    stats['viewport_rejected_faces'] = int(np.count_nonzero(outside))

    def draw(projected, z, original_face):
        e2 = _edge(projected[0], projected[1])
        area = e2[0]*projected[2, 0]+e2[1]*projected[2, 1]+e2[2]
        _require(np.isfinite(area), 'Projected area overflowed.')
        if area == 0:
            stats['projected_degenerate_triangles'] += 1
            return
        if area < 0:
            projected = projected[[0, 2, 1]]
            z = z[[0, 2, 1]]
        low, high = projected.min(axis=0), projected.max(axis=0)
        if high[0] < .5 or low[0] > width-.5 or high[1] < .5 or low[1] > height-.5:
            stats['viewport_rejected_triangles'] += 1
            return
        xmin = max(0, int(math.ceil(float(low[0])-.5)))
        xmax = min(width-1, int(math.floor(float(high[0])-.5)))
        ymin = max(0, int(math.ceil(float(low[1])-.5)))
        ymax = min(height-1, int(math.floor(float(high[1])-.5)))
        if xmin > xmax or ymin > ymax:
            stats['viewport_rejected_triangles'] += 1
            return
        edges = (_edge(projected[1], projected[2]), _edge(projected[2], projected[0]),
                 _edge(projected[0], projected[1]))
        # Recompute after winding normalization to keep numerator/denominator
        # arithmetic identical for front- and back-facing input triangles.
        area = edges[2][0]*projected[2, 0]+edges[2][1]*projected[2, 1]+edges[2][2]
        if area <= 0:
            stats['projected_degenerate_triangles'] += 1
            return
        xx = np.arange(xmin, xmax+1, dtype=float)[None, :]+.5
        yy = np.arange(ymin, ymax+1, dtype=float)[:, None]+.5
        values = [a*xx+b*yy+c for a, b, c in edges]
        top_left = (_top_left(projected[1], projected[2]), _top_left(projected[2], projected[0]),
                    _top_left(projected[0], projected[1]))
        covered = np.ones(values[0].shape, dtype=bool)
        for value, include_edge in zip(values, top_left):
            covered &= (value > 0) | ((value == 0) & include_edge)
        stats['raster_triangles'] += 1
        if not np.any(covered):
            return
        reciprocal_numerator = values[0]/z[0]+values[1]/z[1]+values[2]/z[2]
        candidate = np.full(covered.shape, np.inf, dtype=np.float64)
        np.divide(area, reciprocal_numerator, out=candidate,
                  where=covered & (reciprocal_numerator > 0))
        current = depth[ymin:ymax+1, xmin:xmax+1]
        replace = covered & np.isfinite(candidate) & (candidate < current)
        if np.any(replace):
            current[replace] = candidate[replace]
            normals[ymin:ymax+1, xmin:xmax+1][replace] = unit[original_face]
            face_id[ymin:ymax+1, xmin:xmax+1][replace] = original_face
            stats['depth_updates'] += int(np.count_nonzero(replace))

    for original_face in np.flatnonzero(candidates):
        original_face = int(original_face)
        if all_front[original_face]:
            draw(screen[ordered[original_face]], camera_triangles[original_face, :, 2], original_face)
        else:
            polygon = _clip_near(camera_triangles[original_face], near)
            if len(polygon) < 3:
                continue
            polygon = np.asarray(polygon)
            projected = _project(polygon, intrinsics)
            _require(np.all(np.isfinite(projected)), 'Clipped projection overflowed.')
            for i in range(1, len(polygon)-1):
                indices = [0, i, i+1]
                draw(projected[indices], polygon[indices, 2], original_face)
    mask = face_id >= 0
    depth[~mask] = np.nan
    stats['covered_pixels'] = int(np.count_nonzero(mask))
    return dict(depth=depth, normals=normals, mask=mask, face_id=face_id, stats=stats)
