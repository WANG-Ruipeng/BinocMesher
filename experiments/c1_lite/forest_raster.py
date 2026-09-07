"""Bounded deterministic CPU geometry raster for original Forest cameras.

No native mesher, Blender, GPU context, or scene file is opened here. The
existing Numba compiler runs serial, float64 arithmetic without fastmath or
disk caching. Mesh arrays are read only; only the current image is returned.
"""
from __future__ import annotations
import hashlib
import json
import math
import time

import numpy as np
from numba import njit

FOREST_CAMERA_CANONICAL_SHA = '5e986d2e2280e053d5ab4324bc58f4f7a36a3371c87ec75c634b11b929accd1d'
FOREST_CAMERA_FILE_SHA = '40535546e900f9fc9c6d6efab3e1538da8eb9a88021c9078ebec4697a53f23b7'
FOREST_SCENE_SHA = 'ba63ee3b72e800d67ebf81fa7a35997cb2366a7f3889c4dd243e0614c50902fe'
STAT_NAMES = ('degenerate_faces', 'behind_near_faces', 'beyond_far_faces',
              'near_clipped_faces', 'far_clipped_faces', 'viewport_rejected_faces',
              'projected_degenerate_triangles', 'raster_triangles',
              'depth_updates', 'pixel_tests')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def forest_image_camera_document(frozen_camera_inputs):
    """Create a NEW explicitly unrelaxed image-camera document.

    The frozen mesher camera is checked by canonical content, never modified.
    Its poses equal the original Blender camera's OpenCV matrices for all 64
    frames. K=1500 comes from the original 50 mm / 32x18 mm sensor, NOT a
    multiply/divide adjustment of the LOD focal length. See provenance note.
    """
    encoded = json.dumps(frozen_camera_inputs, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(hashlib.sha256(encoded.encode()).hexdigest() == FOREST_CAMERA_CANONICAL_SHA,
            'This camera adapter requires the frozen Forest64 camera input document.')
    source = json.loads(encoded)
    return {'schema': 'forest-original-image-camera-v1', 'intrinsics_role': 'UNRELAXED_IMAGE_CAMERA',
            'poses': source['poses'], 'image_intrinsics': [[[1500., 0., 480.], [0., 1500., 270.], [0., 0., 1.]] for _ in range(64)],
            'widths': source['widths'], 'heights': source['heights'], 'times_seconds': source['times_seconds'],
            'near': 0.10000000149011612, 'far': 10000.,
            'pixel_centers': 'x+0.5,y+0.5', 'depth_kind': 'CAMERA_Z_NOT_RAY_LENGTH',
            'provenance': {'source_camera_canonical_sha256': FOREST_CAMERA_CANONICAL_SHA,
                'source_camera_file_sha256': FOREST_CAMERA_FILE_SHA, 'coarse_scene_sha256': FOREST_SCENE_SHA,
                'all_64_scene_world_poses_equal_frozen_cv_poses': True,
                'projection': 'Ideal pinhole from saved 50mm lens and 32x18mm sensor; no Cycles subpixel/jitter equivalence claim.',
                'mesher_relax_1p05_input_unchanged': True}}


def _camera(document, index):
    require(type(index) is int and 0 <= index < len(document['poses']), 'Invalid zero-based original frame index.')
    require(document.get('intrinsics_role') == 'UNRELAXED_IMAGE_CAMERA',
            'Pass an explicit image-camera document, not the relaxed mesher camera.')
    n = len(document['poses'])
    require(all(len(document[k]) == n for k in ('image_intrinsics', 'widths', 'heights')), 'Camera array lengths differ.')
    width, height = document['widths'][index], document['heights'][index]
    require(type(width) is int and type(height) is int and 0 < width <= 4096 and 0 < height <= 4096
            and width*height <= 4_194_304, 'Image allocation exceeds the bounded raster profile.')
    pose = np.asarray(document['poses'][index], np.float64)
    k = np.ascontiguousarray(document['image_intrinsics'][index], dtype=np.float64)
    require(pose.shape == (4, 4) and np.isfinite(pose).all() and np.array_equal(pose[3], [0, 0, 0, 1]), 'Invalid camera-to-world matrix.')
    require(k.shape == (3, 3) and np.isfinite(k).all() and np.array_equal(k[2], [0, 0, 1])
            and abs(float(np.linalg.det(k[:2, :2]))) > 0, 'Invalid image intrinsics.')
    near, far = float(document['near']), float(document['far'])
    require(math.isfinite(near) and math.isfinite(far) and 1e-6 <= near < far, 'Invalid near/far clipping planes.')
    return pose, np.linalg.inv(pose), k, width, height, near, far


@njit(cache=False, fastmath=False, inline='always')
def _before(a, b):
    return a[0] < b[0] or (a[0] == b[0] and (a[1] < b[1] or (a[1] == b[1] and a[2] < b[2])))


@njit(cache=False, fastmath=False, inline='always')
def _append(poly, n, x, y, z):
    if n and poly[n-1, 0] == x and poly[n-1, 1] == y and poly[n-1, 2] == z:
        return n
    if n >= len(poly):
        raise ValueError('Clipping polygon exceeded its fixed scratch budget.')
    poly[n, 0], poly[n, 1], poly[n, 2] = x, y, z
    return n+1


@njit(cache=False, fastmath=False)
def _clip(source, n, target, plane, keep_above):
    count = 0
    previous = n-1
    for current in range(n):
        old_inside = source[previous, 2] >= plane if keep_above else source[previous, 2] <= plane
        new_inside = source[current, 2] >= plane if keep_above else source[current, 2] <= plane
        if old_inside != new_inside:
            lo, hi = previous, current
            if _before(source[hi], source[lo]):
                lo, hi = hi, lo
            if source[lo, 2] == plane:
                x, y = source[lo, 0], source[lo, 1]
            elif source[hi, 2] == plane:
                x, y = source[hi, 0], source[hi, 1]
            else:
                t = (plane-source[lo, 2])/(source[hi, 2]-source[lo, 2])
                x = source[lo, 0]+t*(source[hi, 0]-source[lo, 0])
                y = source[lo, 1]+t*(source[hi, 1]-source[lo, 1])
            count = _append(target, count, x, y, plane)
        if new_inside:
            count = _append(target, count, source[current, 0], source[current, 1], source[current, 2])
        previous = current
    if count > 1 and (target[0, 0] == target[count-1, 0] and target[0, 1] == target[count-1, 1]
                      and target[0, 2] == target[count-1, 2]):
        count -= 1
    return count


@njit(cache=False, fastmath=False, inline='always')
def _edge(x0, y0, x1, y1):
    return y0-y1, x1-x0, x0*y1-x1*y0


@njit(cache=False, fastmath=False, inline='always')
def _top_left(x0, y0, x1, y1):
    return y1-y0 < 0 or (y1 == y0 and x1-x0 > 0)


@njit(cache=False, fastmath=False)
def _raster_element(vertices, faces, camera, k, near, far, element, labels,
                    depth, normals, element_ids, face_ids, label_ids, max_pixel_tests):
    stats = np.zeros(10, np.int64)
    polygon = np.empty((8, 3), np.float64)
    scratch = np.empty((8, 3), np.float64)
    projected = np.empty((8, 2), np.float64)
    height, width = depth.shape
    for face_index in range(len(faces)):
        # Canonical cyclic rotation keeps the original orientation while
        # removing irrelevant cyclic face-list differences from arithmetic.
        start = 0
        if _before(vertices[faces[face_index, 1]], vertices[faces[face_index, start]]):
            start = 1
        if _before(vertices[faces[face_index, 2]], vertices[faces[face_index, start]]):
            start = 2
        ia, ib, ic = faces[face_index, start], faces[face_index, (start+1) % 3], faces[face_index, (start+2) % 3]
        ux, uy, uz = vertices[ib, 0]-vertices[ia, 0], vertices[ib, 1]-vertices[ia, 1], vertices[ib, 2]-vertices[ia, 2]
        vx, vy, vz = vertices[ic, 0]-vertices[ia, 0], vertices[ic, 1]-vertices[ia, 1], vertices[ic, 2]-vertices[ia, 2]
        nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
        scale = max(abs(nx), abs(ny), abs(nz))
        if not math.isfinite(scale):
            raise ValueError('World triangle normal overflowed.')
        if scale == 0:
            stats[0] += 1
            continue
        z0, z1, z2 = camera[ia, 2], camera[ib, 2], camera[ic, 2]
        low_z, high_z = min(z0, z1, z2), max(z0, z1, z2)
        if high_z < near:
            stats[1] += 1
            continue
        if low_z > far:
            stats[2] += 1
            continue
        for j in range(3):
            index = faces[face_index, (start+j) % 3]
            polygon[j, 0], polygon[j, 1], polygon[j, 2] = camera[index, 0], camera[index, 1], camera[index, 2]
        n = 3
        if low_z < near:
            stats[3] += 1
            n = _clip(polygon, n, scratch, near, True)
            for j in range(n):
                polygon[j, :] = scratch[j, :]
        if n < 3:
            continue
        if high_z > far:
            stats[4] += 1
            n = _clip(polygon, n, scratch, far, False)
            for j in range(n):
                polygon[j, :] = scratch[j, :]
        if n < 3:
            continue
        anchor = 0
        for j in range(n):
            x, y, z = polygon[j, 0], polygon[j, 1], polygon[j, 2]
            projected[j, 0] = (k[0, 0]*x+k[0, 1]*y+k[0, 2]*z)/z
            projected[j, 1] = (k[1, 0]*x+k[1, 1]*y+k[1, 2]*z)/z
            if not math.isfinite(projected[j, 0]) or not math.isfinite(projected[j, 1]):
                raise ValueError('Pinhole projection overflowed.')
            if _before(polygon[j], polygon[anchor]):
                anchor = j
        px_low = px_high = projected[0, 0]
        py_low = py_high = projected[0, 1]
        for j in range(1, n):
            px_low = min(px_low, projected[j, 0])
            px_high = max(px_high, projected[j, 0])
            py_low = min(py_low, projected[j, 1])
            py_high = max(py_high, projected[j, 1])
        if px_high < .5 or px_low > width-.5 or py_high < .5 or py_low > height-.5:
            stats[5] += 1
            continue
        nx, ny, nz = nx/scale, ny/scale, nz/scale
        length = math.sqrt(nx*nx+ny*ny+nz*nz)
        nx, ny, nz = nx/length, ny/length, nz/length
        for j in range(1, n-1):
            p0, p1, p2 = anchor, (anchor+j) % n, (anchor+j+1) % n
            x0, y0, d0 = projected[p0, 0], projected[p0, 1], polygon[p0, 2]
            x1, y1, d1 = projected[p1, 0], projected[p1, 1], polygon[p1, 2]
            x2, y2, d2 = projected[p2, 0], projected[p2, 1], polygon[p2, 2]
            ax2, ay2, c2 = _edge(x0, y0, x1, y1)
            area = ax2*x2+ay2*y2+c2
            if not math.isfinite(area):
                raise ValueError('Projected area overflowed.')
            if area == 0:
                stats[6] += 1
                continue
            if area < 0:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                d1, d2 = d2, d1
            xmin = max(.5, min(x0, x1, x2))
            xmax = min(width-.5, max(x0, x1, x2))
            ymin = max(.5, min(y0, y1, y2))
            ymax = min(height-.5, max(y0, y1, y2))
            if xmin > xmax or ymin > ymax:
                continue
            xlo, xhi, ylo, yhi = int(math.ceil(xmin-.5)), int(math.floor(xmax-.5)), int(math.ceil(ymin-.5)), int(math.floor(ymax-.5))
            if xlo > xhi or ylo > yhi:
                continue
            a0, b0, c0 = _edge(x1, y1, x2, y2)
            a1, b1, c1 = _edge(x2, y2, x0, y0)
            a2, b2, c2 = _edge(x0, y0, x1, y1)
            area = a2*x2+b2*y2+c2
            if area <= 0:
                stats[6] += 1
                continue
            include0, include1, include2 = _top_left(x1, y1, x2, y2), _top_left(x2, y2, x0, y0), _top_left(x0, y0, x1, y1)
            tests = (xhi-xlo+1)*(yhi-ylo+1)
            if tests > max_pixel_tests-stats[9]:
                raise ValueError('Raster pixel-work budget exceeded; no partial image is published.')
            stats[9] += tests
            stats[7] += 1
            for y in range(ylo, yhi+1):
                yy = y+.5
                for x in range(xlo, xhi+1):
                    xx = x+.5
                    e0, e1, e2 = a0*xx+b0*yy+c0, a1*xx+b1*yy+c1, a2*xx+b2*yy+c2
                    if not ((e0 > 0 or (e0 == 0 and include0)) and (e1 > 0 or (e1 == 0 and include1)) and (e2 > 0 or (e2 == 0 and include2))):
                        continue
                    denominator = e0/d0+e1/d1+e2/d2
                    if denominator <= 0:
                        continue
                    candidate = area/denominator
                    if math.isfinite(candidate) and candidate < depth[y, x]:
                        depth[y, x] = candidate
                        normals[y, x, 0], normals[y, x, 1], normals[y, x, 2] = nx, ny, nz
                        element_ids[y, x], face_ids[y, x] = element, face_index
                        label_ids[y, x] = labels[face_index] if len(labels) else -1
                        stats[8] += 1
    return stats


def render_scene(meshes, camera_doc, frame_index, event_face_ids=None, *, max_pixel_tests=2_000_000_000):
    """Rasterize actual element meshes in stable (element, original-face) order.

    meshes: sequence of (vertices, faces) or (vertices, faces, tags). Tags are
    deliberately ignored. event_face_ids: optional dict {element: int64[F]};
    callers can encode components rather than events. -1 means no label. A
    scalar label cannot express overlapping events; retain a separate CSR
    map or assign their already-defined component ID, never overwrite one.

    Returns NaN-background camera-Z depth/world flat normals, mask, element_id,
    original per-element face_id, label_id, and work/count stats. Strict depth
    comparison makes first face win exact ties. Not a Cycles/RGB renderer.
    """
    started = time.perf_counter()
    require(type(max_pixel_tests) is int and max_pixel_tests > 0, 'Positive integer raster-work budget required.')
    pose, inverse, k, width, height, near, far = _camera(camera_doc, frame_index)
    meshes = list(meshes)
    require(0 < len(meshes) <= 64, 'Expected a bounded nonempty element mesh list.')
    labels = {} if event_face_ids is None else event_face_ids
    require(isinstance(labels, dict) and all(type(i) is int and 0 <= i < len(meshes) for i in labels), 'Face labels must be keyed by existing element index.')
    depth = np.full((height, width), np.inf, np.float64)
    normals = np.full((height, width, 3), np.nan, np.float64)
    element_ids = np.full((height, width), -1, np.int32)
    face_ids = np.full((height, width), -1, np.int64)
    label_ids = np.full((height, width), -1, np.int64)
    totals = np.zeros(10, np.int64)
    counts, input_bytes = [], 0
    for element, mesh in enumerate(meshes):
        require(len(mesh) in (2, 3), 'Expected each mesh to contain vertices/faces and optional ignored tags.')
        vertices, faces = np.asarray(mesh[0]), np.asarray(mesh[1])
        require(vertices.ndim == 2 and vertices.shape[1] == 3 and vertices.dtype.kind in 'fiu'
                and np.isfinite(vertices).all(), 'Vertices must be finite numeric (n,3).')
        require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in 'iu', 'Faces must be integer (m,3).')
        require(not faces.size or (int(faces.min()) >= 0 and int(faces.max()) < len(vertices)), 'Face index is outside its own element.')
        points = np.ascontiguousarray(vertices, dtype=np.float64)
        indices = np.ascontiguousarray(faces, dtype=np.int64)
        camera = np.ascontiguousarray(points @ inverse[:3, :3].T+inverse[:3, 3])
        require(np.isfinite(camera).all(), 'Camera transform overflowed.')
        element_labels = np.empty(0, np.int64)
        if element in labels:
            values = np.asarray(labels[element])
            require(values.shape == (len(faces),) and values.dtype.kind in 'iu'
                    and (not values.size or (int(values.min()) >= -1 and int(values.max()) <= np.iinfo(np.int64).max)),
                    'Face labels must contain one integer ID >= -1 per actual face.')
            element_labels = np.ascontiguousarray(values, dtype=np.int64)
        counts.append({'element': element, 'vertices': len(vertices), 'faces': len(faces)})
        input_bytes += vertices.nbytes+faces.nbytes
        totals += _raster_element(points, indices, camera, k, near, far, element, element_labels,
            depth, normals, element_ids, face_ids, label_ids, max_pixel_tests-int(totals[9]))
    mask = face_ids >= 0
    depth[~mask] = np.nan
    stats = {name: int(value) for name, value in zip(STAT_NAMES, totals)}
    stats.update(elements=counts, input_geometry_bytes=input_bytes, covered_pixels=int(mask.sum()),
        output_buffer_bytes=sum(x.nbytes for x in (depth, normals, element_ids, face_ids, label_ids, mask)),
        frame_index_zero_based=frame_index, width=width, height=height, near=near, far=far,
        max_pixel_tests=max_pixel_tests, wall_seconds=time.perf_counter()-started,
        backend='NUMBA_CPU_SERIAL_FLOAT64_FASTMATH_FALSE_CACHE_FALSE',
        raster_order='element index then original face index; exact depth tie keeps first',
        scope='PRE_DISPLACEMENT_OPAQUE_GEOMETRY_PIXEL_CENTER_SAMPLES_NOT_RGB_OR_ALL_RAY_VISIBILITY')
    return {'depth': depth, 'normals': normals, 'mask': mask, 'element_id': element_ids,
            'face_id': face_ids, 'label_id': label_ids, 'stats': stats}
