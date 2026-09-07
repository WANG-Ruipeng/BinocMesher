"""Independent saved-JSON projection audit; no native queries or raster calls."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

PLANES = ('near', 'far', 'left', 'right', 'top', 'bottom')


def require(value, message):
    if not value:
        raise ValueError(message)


def fraction(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), 'Expected finite numeric coordinate.')
    return F(value)


def inverse(matrix):
    n = len(matrix)
    require(n == 4 and all(len(row) == n for row in matrix), 'Expected 4x4 pose.')
    a = [[fraction(x) for x in row]+[F(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next((i for i in range(column, n) if a[i][column]), None)
        require(pivot is not None, 'Singular pose.')
        a[column], a[pivot] = a[pivot], a[column]
        denominator = a[column][column]
        a[column] = [x/denominator for x in a[column]]
        for i in range(n):
            if i != column:
                scale = a[i][column]
                a[i] = [x-scale*y for x, y in zip(a[i], a[column])]
    return [row[n:] for row in a]


def projector(camera, index):
    require(camera['intrinsics_role'] == 'UNRELAXED_IMAGE_CAMERA', 'Not original image calibration.')
    pose = camera['poses'][index]
    require(pose[3] == [0, 0, 0, 1], 'Non-affine camera pose.')
    k = camera['image_intrinsics'][index]
    require(len(k) == 3 and all(len(r) == 3 for r in k) and k[2] == [0, 0, 1], 'Invalid K.')
    return {'inverse': inverse(pose), 'k': [[fraction(x) for x in row] for row in k],
        'width': fraction(camera['widths'][index]), 'height': fraction(camera['heights'][index]),
        'near': fraction(camera['near']), 'far': fraction(camera['far'])}


def source_geometry(event):
    ids, coordinates, triangles = event['boundary_actual_ids'], event['boundary_coordinates'], event['source_triangles']
    require(len(ids) == len(coordinates) and len(set(ids)) == len(ids), 'Ambiguous source ID coordinate map.')
    require(all(type(i) is int and i >= 0 for i in ids), 'Invalid source ID.')
    lookup = {i: tuple(fraction(x) for x in p) for i, p in zip(ids, coordinates)}
    require(all(len(p) == 3 for p in lookup.values()) and triangles, 'Empty/malformed source geometry.')
    require(all(len(t) == 3 and all(type(i) is int and i in lookup for i in t) for t in triangles),
            'Source triangle has an unbound coordinate ID.')
    used = sorted({i for t in triangles for i in t})
    return [lookup[i] for i in used], [[used.index(i) for i in t] for t in triangles]


def project_support(event, camera):
    points, triangles = source_geometry(event)
    transformed, inequalities = [], []
    for point in points:
        p = (*point, F(1))
        x, y, z, w = [sum(a*b for a, b in zip(row, p)) for row in camera['inverse']]
        require(w == 1, 'Unexpected homogeneous world transform.')
        qx, qy = [sum(a*b for a, b in zip(row, (x, y, z))) for row in camera['k'][:2]]
        transformed.append((qx, qy, z))
        inequalities.append((z-camera['near'], camera['far']-z, qx,
            camera['width']*z-qx, qy, camera['height']*z-qy))
    maxima = [max(p[i] for p in inequalities) for i in range(6)]
    excluded = [PLANES[i] for i, value in enumerate(maxima) if value < 0]
    per_triangle_outside = sum(any(max(inequalities[v][j] for v in tri) < 0 for j in range(6))
                               for tri in triangles)
    bbox = None
    if all(p[2] > 0 for p in transformed):
        bbox = [[float(min(p[d]/p[2] for p in transformed)) for d in range(2)],
                [float(max(p[d]/p[2] for p in transformed)) for d in range(2)]]
    return {'same_plane_exclusion': excluded,
        'strict_primary_witness': {'plane': excluded[0], 'maximum_signed_value_exact': str(maxima[PLANES.index(excluded[0])])} if excluded else None,
        'triangle_count': len(triangles), 'individually_same_plane_excluded_triangles': per_triangle_outside,
        'camera_z_range': [float(min(p[2] for p in transformed)), float(max(p[2] for p in transformed))],
        'positive_z_vertex_pixel_bounds': bbox}


def classify(projection, saved):
    visible, isolated = saved['visible_baseline_source_pixels'], saved['unoccluded_source_pixels']
    require(type(visible) is int and type(isolated) is int and visible >= 0 and isolated >= 0, 'Invalid saved pixel count.')
    require(not visible or isolated, 'Visible source has no isolated support samples.')
    if projection['same_plane_exclusion']:
        require(visible == isolated == 0, 'Exact frustum exclusion contradicts saved raster samples.')
        return 'STRICT_SAME_PLANE_FRUSTUM_EXCLUSION'
    if visible:
        return 'SAVED_VISIBLE_SOURCE_SUPPORT'
    if isolated:
        return 'SAVED_OCCLUDED_OR_DEPTH_TIE_NOT_SELECTED'
    return 'REMAINING_NO_PIXEL_CENTER_COVERAGE'


def aggregate(rows):
    ids = {r['event_id'] for r in rows}
    visible = {r['event_id'] for r in rows if r['saved_visible_pixels'] > 0}
    isolated = {r['event_id'] for r in rows if r['saved_unoccluded_pixels'] > 0}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['event_id']].append(row)
    return {'events': len(ids), 'event_frames': len(rows),
        'classification_counts': dict(Counter(r['classification'] for r in rows)),
        'same_plane_primary_counts': dict(Counter(r['projection']['same_plane_exclusion'][0] for r in rows if r['projection']['same_plane_exclusion'])),
        'events_excluded_at_every_requested_natural_frame': sum(all(r['projection']['same_plane_exclusion'] for r in rr) for rr in grouped.values()),
        'events_with_any_isolated_pixel_coverage': len(isolated), 'events_with_any_saved_visible_support': len(visible),
        'visible_admission_rate': sum(r['jointly_admitted'] for r in [next(x for x in rows if x['event_id'] == eid) for eid in visible])/len(visible) if visible else None,
        'rate_null_reason': None if visible else 'No visible support: denominator is zero, not a measured zero-percent admission rate.'}


def audit(visibility, components):
    visibility, components = Path(visibility).resolve(), Path(components).resolve()
    hashes = {}
    def read(path, expected=None):
        path = Path(path).resolve(); data = path.read_bytes()
        require(len(data) < 10*1024**2, 'Saved JSON exceeds per-file audit budget.')
        sha = hashlib.sha256(data).hexdigest()
        require(expected is None or sha == expected, 'Input reference hash mismatch: '+str(path))
        hashes[str(path)] = sha
        return json.loads(data)
    def reference(root, ref):
        path = (root/ref['path']).resolve()
        require(path.is_relative_to(root), 'Reference leaves its artifact directory.')
        return read(path, ref['sha256'])
    saved = read(visibility/'summary.json'); certificate = read(components/'summary.json')
    require(saved['status'] == 'PASS_COMPLETE_NATURAL_SUPPORT_VISIBILITY_TRIAGE', 'Visibility attempt incomplete.')
    require(saved['certification_sha256'] == hashes[str(components/'summary.json')], 'Wrong component certificate.')
    camera = reference(visibility, saved['original_render_camera'])
    require(camera['pixel_centers'] == 'x+0.5,y+0.5' and camera['depth_kind'] == 'CAMERA_Z_NOT_RAY_LENGTH', 'Camera sampling mismatch.')
    graph = reference(components, certificate['support_graph'])
    memberships = {}
    for c in graph['components']:
        for eid in c['events']:
            require(eid not in memberships, 'Duplicate graph event.')
            memberships[eid] = c
    require(len(memberships) == 131, 'Expected fixed 131-event population.')
    qrefs = {Path(r['path']).stem: r for r in certificate['query_artifacts']}
    frefs = {Path(r['path']).stem: r for r in saved['frame_artifacts']}
    expected_frames = {f'frame_{i:04d}' for i in [*range(21, 29), *range(37, 45)]}
    require(set(frefs) == expected_frames and expected_frames <= set(qrefs), 'Natural frame schedule incomplete.')
    rows = []; by_event = defaultdict(set); buffers_equal = []
    for key in sorted(frefs):
        query = reference(components, qrefs[key]); frame = reference(visibility, frefs[key])
        require(query['query'].get('active') is True and
                {k: v for k, v in query['query'].items() if k != 'active'} == frame['query'],
                'Frame query identity differs (apart from component-only active flag).')
        info = query['query']; index = info['frame_index_zero_based']
        require(info['frame_number'] == index+1 and key == f'frame_{index+1:04d}', 'Frame index is mislabeled.')
        require(camera['times_seconds'][index].hex() == info['global_camera_time_hex'], 'Natural camera time bits differ.')
        projection_camera = projector(camera, index)
        events = {e['event_id']: e for e in query['events']}
        rendered = {e['event_id']: e for e in frame['events']}
        require(len(events) == len(query['events']) and len(rendered) == len(frame['events']) and set(events) == set(rendered), 'Frame population mismatch.')
        expected_ids = {eid for eid, c in memberships.items() if c['root'] == query['root']}
        require(set(events) == expected_ids, 'Root population differs from full graph.')
        identical = all(frame['buffer_sha256']['baseline'][k] == frame['buffer_sha256']['combined'][k]
                        for k in ('depth', 'normals', 'mask', 'element_id', 'face_id', 'label_id'))
        buffers_equal.append(identical)
        for eid, event in sorted(events.items()):
            seen = rendered[eid]; component = memberships[eid]
            require(event['status'] == 'COMPLETE_ACTUAL_REQUESTED_SUPPORT' and seen['domain_kind'] == event['kind'], 'Unresolved/mismatched source support.')
            admitted = component['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE'
            require(seen['jointly_admitted'] == admitted and seen['component_id'] == component['component_id'], 'Admission label differs from component graph.')
            projection = project_support(event, projection_camera)
            rows.append({'event_id': eid, 'query': key, 'element': event['element'],
                'fixed_source': event['kind'] == 'FIXED_SOURCE_AND_REPLACEMENT', 'jointly_admitted': admitted,
                'classification': classify(projection, seen), 'projection': projection,
                'saved_visible_pixels': seen['visible_baseline_source_pixels'], 'saved_unoccluded_pixels': seen['unoccluded_source_pixels']})
            by_event[eid].add(key)
    require(len(rows) == 1048 and len(by_event) == 131 and all(len(v) == 8 for v in by_event.values()), 'Expected 131 events times eight natural samples each.')
    groups = {'all_131_including_one_proxy': aggregate(rows),
        'compiled_130_fixed_source': aggregate([r for r in rows if r['fixed_source']]),
        'jointly_admitted_25': aggregate([r for r in rows if r['jointly_admitted']]),
        'one_broad_support_proxy': aggregate([r for r in rows if not r['fixed_source']])}
    require(groups['compiled_130_fixed_source']['events'] == 130 and groups['jointly_admitted_25']['events'] == 25, 'Fixed stratified denominators changed.')
    require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()), 'Input changed during audit.')
    return {'schema': 'forest-independent-saved-projection-v1', 'status': 'PASS_PROJECTION_ONLY_AUDIT',
        'arithmetic': 'Exact Fraction inverse of recorded binary64 affine c2w, exact source coordinates and six affine homogeneous clip inequalities; strict same-plane negative maximum is sufficient.',
        'clip_domain': 'Continuous original image rectangle [0,W] x [0,H], near <= camera Z <= far. Not widened LOD FOV, and not the smaller pixel-center rectangle.',
        'independent_scope': 'Recomputed projection/frustum exclusion only. Isolated raster samples, full-scene occlusion/tie and final buffer equality are inherited from hash-bound saved artifacts, not rerendered.',
        'camera_transform_scope': 'World-space actual slicing_output coordinates; reader/core pass XYZ through. Terrain wrapper attaches attributes then concatenates, not another axis or object transform. Later displacement is excluded.',
        'coordinate_code_references': ['binocmesher/source/slicing.cpp:1313', 'experiments/c1_lite/forest_native_reader.py:225', 'binocmesher/core.py:595',
            '/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/core.py:81',
            '/home/warpwang/src/BinocMesher/infinigen_binocmesher/infinigen/terrain/utils/mesh.py:338'],
        'camera_provenance': camera['provenance'], 'groups': groups, 'frame_count': len(frefs),
        'all_six_saved_buffers_identical_each_frame': all(buffers_equal),
        'zero_visibility_does_not_mean_zero_events_or_zero_admission_rate': True,
        'no_native_no_raster_no_scene_load': True, 'old_artifacts_unchanged': True,
        'input_sha256': hashes, 'executed_auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'event_frames': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visibility', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.parent.is_dir(), 'Fresh output file in existing artifact directory required.')
    report = audit(args.visibility, args.components)
    data = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n'
    require(len(data.encode()) <= 2*1024**2, 'Compact audit exceeds 2 MiB.')
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(data)
    print(json.dumps({'status': report['status'], 'groups': report['groups'], 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
