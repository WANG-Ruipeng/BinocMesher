"""Connectivity-only attribution using frozen source data and metric code."""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PRIOR = Path(__file__).resolve().parent.parent/'graph_lift_probe'
sys.path.insert(0, str(PRIOR))
from model import GraphModel, area_centroid, base_faces, validate_mesh, require
from model import quadrature, reference_triangles
from metrics import TriangleSurface
from run_probe import measure
from connectivity import canonical_faces, insert_point, retriangulate_delaunay, enumerate_triangulations


LEVELS = ((16, 12), (32, 12), (32, 24))
RULES = ('xyz_mean', 'vertex_mean', 'area_centroid', 'face_root0', 'face_root1',
         'edge_root0', 'edge_root1')


def oriented_canonical(vertices, faces):
    result = []
    for a, b, c in canonical_faces(faces):
        cross = np.cross(vertices[b, :2]-vertices[a, :2], vertices[c, :2]-vertices[a, :2])
        require(abs(cross) > 1e-13, 'degenerate normalized face')
        result.append((a, b, c) if cross > 0 else (a, c, b))
    return np.asarray(result, dtype=int)


def fan_faces(m):
    return np.asarray([[i, (i+1) % m, m] for i in range(m)], dtype=int)


def choose_uv(polygon, rule):
    if rule == 'vertex_mean':
        return polygon.mean(axis=0), {'kind': 'parameter_vertex_mean'}
    if rule == 'area_centroid':
        return area_centroid(polygon), {'kind': 'parameter_area_centroid'}
    root = int(rule[-1])
    faces = base_faces(len(polygon), root)
    if rule.startswith('face'):
        xy = polygon[faces]
        areas = np.cross(xy[:, 1]-xy[:, 0], xy[:, 2]-xy[:, 0])
        chosen = int(np.argmax(areas))
        return polygon[faces[chosen]].mean(axis=0), {
            'kind': 'largest_parameter_face_barycenter', 'root': root,
            'face_vertex_ids': faces[chosen].tolist(), 'xy_candidates': len(faces)}
    counts = {}
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0)+1
    interior = sorted(e for e, count in counts.items() if count == 2)
    edge = min(interior, key=lambda e: (-float(np.sum((polygon[e[0]]-polygon[e[1]])**2)), e))
    return polygon[list(edge)].mean(axis=0), {
        'kind': 'longest_parameter_interior_edge_midpoint', 'root': root,
        'edge_vertex_ids': list(edge), 'xy_candidates': len(interior)}


def construct_points(model, polygon):
    source_before, polygon_before = model.receipt(), polygon.copy()
    start = perf_counter()
    boundary = model.evaluate(polygon)[0]
    boundary_time = perf_counter()-start
    m = len(polygon)
    sets = {}
    for rule in RULES:
        start = perf_counter()
        if rule == 'xyz_mean':
            point = boundary.mean(axis=0)
            queries, selection = 0, {'kind': 'XYZ_boundary_mean'}
        else:
            q, selection = choose_uv(polygon, rule)
            point = model.evaluate(q)[0][0]
            queries = 1
        if rule.startswith(('face', 'edge')):
            initial = insert_point(boundary, base_faces(m, int(rule[-1])), point)
        else:
            initial = fan_faces(m)
        vertices = np.vstack((boundary, point))
        require(np.array_equal(boundary, vertices[:m]), 'construction changed boundary')
        checks = validate_mesh(vertices, initial, boundary)
        sets[rule] = {'vertices': vertices, 'initial_faces': initial,
            'budget': {'interior_model_queries': queries, 'model_queries_for_connectivity': 0,
                       'original_target_queries': 0, 'reference_queries_for_construction': 0,
                       'construction_seconds': perf_counter()-start},
            'selection': selection, 'checks': checks}
    require(source_before == model.receipt() and np.array_equal(polygon_before, polygon),
            'point construction changed source')
    return sets, {'shared_boundary_model_queries': m, 'shared_boundary_seconds': boundary_time,
                  'actual_all_point_rules_interior_queries': 6}


def build_orbits(sets, polygon):
    m = len(polygon)
    result = {}
    for rule, points in sets.items():
        vertices = points['vertices']
        before = vertices.copy()
        initial = points['initial_faces']
        fan = fan_faces(m)
        begin = perf_counter()
        delaunay, flip_stats = retriangulate_delaunay(vertices, initial, m)
        flip_seconds = perf_counter()-begin
        from_fan, _ = retriangulate_delaunay(vertices, fan, m)
        require(canonical_faces(delaunay) == canonical_faces(from_fan),
                'Delaunay result depends on insertion versus fan start')
        begin = perf_counter()
        orbit, orbit_stats = enumerate_triangulations(vertices, initial, m, max_states=256)
        enum_seconds = perf_counter()-begin
        orbit = [oriented_canonical(vertices, faces) for faces in orbit]
        signatures = [canonical_faces(faces) for faces in orbit]
        index = {sig: i for i, sig in enumerate(signatures)}
        ids = {name: index[canonical_faces(faces)] for name, faces in (
            ('initial', initial), ('fan', fan), ('delaunay', delaunay))}
        for faces in orbit:
            validate_mesh(vertices, faces, vertices[:m])
        require(np.array_equal(vertices, before), 'reconnection mutated XYZ')
        result[rule] = {**points, 'orbit': orbit, 'arm_state_ids': ids,
            'flip_stats': flip_stats, 'reconnection_seconds': flip_seconds,
            'enumeration_stats': orbit_stats, 'enumeration_seconds': enum_seconds}
    require(canonical_faces(result['xyz_mean']['orbit'][result['xyz_mean']['arm_state_ids']['delaunay']]) ==
            canonical_faces(result['vertex_mean']['orbit'][result['vertex_mean']['arm_state_ids']['delaunay']]),
            'same XY led to different Delaunay faces')
    if m == 4:
        reference = result['vertex_mean']
        reference_faces = canonical_faces(reference['orbit'][reference['arm_state_ids']['initial']])
        for rule in ('area_centroid', 'edge_root0', 'edge_root1'):
            require(np.array_equal(result[rule]['vertices'], reference['vertices']),
                    'square center and edge midpoint point mismatch')
            require(canonical_faces(result[rule]['initial_faces']) == reference_faces,
                    'square center and edge split face mismatch')
            require(len(result[rule]['orbit']) == 1, 'square center nonunique triangulation')
    return result


def screen(candidate, baseline, scale, metric='symmetric_mean_distance'):
    c, b = candidate['metrics'], baseline['metrics']
    floor = 1e-8 if 'degrees' in metric else 1e-10*max(1., scale)
    nearzero = 1e-8 if 'degrees' in metric else 1e-9*max(1., scale)
    uncertainty = 2*(abs(c[1][metric]-c[0][metric])+abs(c[2][metric]-c[1][metric])+
                     abs(b[1][metric]-b[0][metric])+abs(b[2][metric]-b[1][metric]))+floor
    reduction = b[-1][metric]-c[-1][metric]
    status = ('NUMERIC_TIE' if abs(reduction) <= floor else
              'RESOLVED_IMPROVEMENT' if reduction > uncertainty else
              'RESOLVED_REGRESSION' if reduction < -uncertainty else 'RESOLUTION_UNRESOLVED')
    return {'status': status, 'candidate_error': c[-1][metric], 'baseline_error': b[-1][metric],
            'absolute_reduction': reduction,
            'relative_reduction_percent': None if b[-1][metric] <= nearzero else 100*reduction/b[-1][metric],
            'empirical_resolution_tolerance': uncertainty, 'certified_bound': False}


PAIRS = (
    ('area_centroid/initial', 'face_root0/initial', 'old_gap_initial'),
    ('face_root0/fan', 'face_root0/initial', 'fixed_face_point_fan_effect'),
    ('face_root0/delaunay', 'face_root0/initial', 'fixed_face_point_delaunay_effect'),
    ('area_centroid/delaunay', 'area_centroid/initial', 'fixed_center_delaunay_effect'),
    ('area_centroid/fan', 'face_root0/fan', 'placement_common_fan_root0'),
    ('area_centroid/delaunay', 'face_root0/delaunay', 'placement_common_delaunay_root0'),
    ('area_centroid/fan', 'face_root1/fan', 'placement_common_fan_root1'),
    ('area_centroid/delaunay', 'face_root1/delaunay', 'placement_common_delaunay_root1'),
    ('area_centroid/fan', 'edge_root0/fan', 'center_vs_edge_common_fan_root0'),
    ('area_centroid/delaunay', 'edge_root0/delaunay', 'center_vs_edge_common_delaunay_root0'),
    ('area_centroid/fan', 'edge_root1/fan', 'center_vs_edge_common_fan_root1'),
    ('area_centroid/delaunay', 'edge_root1/delaunay', 'center_vs_edge_common_delaunay_root1'),
    ('area_centroid/delaunay', 'vertex_mean/delaunay', 'center_definitions_common_delaunay'),
    ('vertex_mean/delaunay', 'xyz_mean/delaunay', 'lift_same_delaunay_connectivity'),
    ('area_centroid/oracle_best', 'face_root0/oracle_best', 'oracle_pointset_envelopes_root0'),
    ('area_centroid/oracle_best', 'face_root1/oracle_best', 'oracle_pointset_envelopes_root1'),
    ('area_centroid/oracle_best', 'edge_root0/oracle_best', 'oracle_center_vs_edge_root0'),
)


def measure_case(case_input, original_result):
    src = case_input['source']
    model = GraphModel(*(np.asarray(src[key]) for key in ('lower','upper','zlower','zupper')), tau=src['tau'])
    model.validate()
    polygon = np.asarray(case_input['polygon_uv'])
    input_before, polygon_before = model.receipt(), polygon.copy()
    start = perf_counter()
    sets, query_budget = construct_points(model, polygon)
    sets = build_orbits(sets, polygon)
    scale = float(np.linalg.norm(np.ptp(sets['vertex_mean']['vertices'][:len(polygon)], axis=0)))
    data = {}
    for rule, points in sets.items():
        data[rule] = {
            'selection': points['selection'], 'interior_xyz': points['vertices'][-1].tolist(),
            'budget': points['budget'], 'checks': points['checks'],
            'arm_state_ids': points['arm_state_ids'], 'flip_stats': points['flip_stats'],
            'reconnection_seconds': points['reconnection_seconds'],
            'enumeration_stats': points['enumeration_stats'],
            'enumeration_seconds': points['enumeration_seconds'],
            'states': [{'id': index, 'faces': faces.tolist(), 'metrics': []}
                       for index, faces in enumerate(points['orbit'])]}
    bridge_max = {'distance': 0., 'normal_degrees': 0.}
    evaluation = {'model_queries': 0, 'unique_surface_measurements': 0,
                  'bridge_measurements': 0, 'cached_alias_measurements': 0}
    for level_index, (ref_n, quad_n) in enumerate(LEVELS):
        uv, weights = quadrature(polygon, quad_n)
        analytic = model.evaluate(uv)
        ref_tri = reference_triangles(model, polygon, ref_n)
        reference = TriangleSurface(ref_tri)
        evaluation['model_queries'] += len(uv)+3*len(ref_tri)
        cache = {}
        for rule, points in sets.items():
            vertices = points['vertices']
            for index, faces in enumerate(points['orbit']):
                key = (vertices.tobytes(), canonical_faces(faces))
                if key not in cache:
                    record = measure({'triangles': vertices[faces]}, model, uv, weights, analytic, reference)
                    record['reference_resolution'], record['quadrature_resolution'] = ref_n, quad_n
                    cache[key] = record
                    evaluation['unique_surface_measurements'] += 1
                else:
                    evaluation['cached_alias_measurements'] += 1
                record = cache[key]
                data[rule]['states'][index]['metrics'].append(record)
                if case_input['family'] == 'affine':
                    require(record['sampled_maximum_distance'] < 1e-10*max(1., scale), 'flat distance control')
                    require(record['normal_correspondence_symmetric_mean_degrees'] < 1e-8, 'flat normal control')
        # Original face ordering preserved here to verify frozen results separately.
        for rule, old_name in (('vertex_mean','rebuilt_lifted_vertex_mean'),
                               ('area_centroid','rebuilt_lifted_area_centroid'),
                               ('face_root0','uniform_model_k1')):
            points = sets[rule]
            bridge = measure({'triangles': points['vertices'][points['initial_faces']]},
                             model, uv, weights, analytic, reference)
            evaluation['bridge_measurements'] += 1
            old = original_result['methods'][old_name]['metrics'][level_index]
            for metric in ('symmetric_mean_distance','maximum_directed_p95', 'sampled_maximum_distance'):
                delta = abs(bridge[metric]-old[metric])
                bridge_max['distance'] = max(bridge_max['distance'], delta)
                require(delta <= 1e-10*max(1., scale), 'frozen distance bridge mismatch')
            metric = 'normal_correspondence_symmetric_mean_degrees'
            delta = abs(bridge[metric]-old[metric])
            bridge_max['normal_degrees'] = max(bridge_max['normal_degrees'], delta)
            require(delta <= 1e-8, 'frozen normal bridge mismatch')
    arms = {}
    for rule, values in data.items():
        for arm, state_id in values['arm_state_ids'].items():
            arms[f'{rule}/{arm}'] = values['states'][state_id]
        best = [min(values['states'], key=lambda state: (state['metrics'][level]['symmetric_mean_distance'], state['id']))
                for level in range(len(LEVELS))]
        worst = [max(values['states'], key=lambda state: (state['metrics'][level]['symmetric_mean_distance'], -state['id']))
                 for level in range(len(LEVELS))]
        envelope = {'evaluation_oracle_not_algorithm': True,
                    'normal_metrics_are_at_distance_selected_state_not_normal_optimal': True,
                    'best_state_ids_by_level': [state['id'] for state in best],
                    'worst_state_ids_by_level': [state['id'] for state in worst],
                    'best_metrics': [state['metrics'][i] for i, state in enumerate(best)],
                    'worst_metrics': [state['metrics'][i] for i, state in enumerate(worst)]}
        values['envelope'] = envelope
        arms[f'{rule}/oracle_best'] = {'metrics': envelope['best_metrics']}
    metrics = ('symmetric_mean_distance','normal_correspondence_symmetric_mean_degrees')
    comparisons = {meaning: {'candidate': candidate, 'baseline': baseline,
                    'evaluation_oracle': '/oracle_best' in candidate,
                    **{metric: screen(arms[candidate], arms[baseline], scale, metric)
                       for metric in (metrics[:1] if '/oracle_best' in candidate else metrics)}}
                   for candidate, baseline, meaning in PAIRS}
    require(input_before == model.receipt() and np.array_equal(polygon_before, polygon), 'source mutated')
    return {'id': case_input['id'], 'domain': case_input['domain'], 'family': case_input['family'],
            'seed': case_input['seed'], 'status': 'MEASURED', 'model_only': True,
            'all_invariants_passed': True, 'square_equivalence_passed': True if len(polygon)==4 else None,
            'scale': scale, 'query_budget': query_budget, 'pointsets': data,
            'bridge_maximum_differences': bridge_max, 'evaluation_cost': evaluation,
            'comparisons': comparisons, 'elapsed_seconds': perf_counter()-start}


def summarize(cases):
    from statistics import median
    summaries = []
    groups = ['all','whole_square','cropped_asymmetric_hexagon','affine','bilinear','rational','strong_rational']
    for candidate, baseline, meaning in PAIRS:
        for group in groups:
            selected = [case for case in cases if group in ('all',case['domain'],case['family'])]
            for metric in ('symmetric_mean_distance','normal_correspondence_symmetric_mean_degrees'):
                if metric not in selected[0]['comparisons'][meaning]:
                    continue
                rows = [case['comparisons'][meaning][metric] for case in selected]
                relative = [row['relative_reduction_percent'] for row in rows if row['relative_reduction_percent'] is not None]
                summaries.append({'comparison': meaning,'candidate':candidate,'baseline':baseline,
                    'metric':metric,'group':group,'case_count':len(rows),
                    **{status:sum(row['status']==status for row in rows) for status in (
                        'RESOLVED_IMPROVEMENT','RESOLVED_REGRESSION','NUMERIC_TIE','RESOLUTION_UNRESOLVED')},
                    'nonzero_baseline_cases':len(relative),
                    'median_relative_reduction_percent':median(relative) if relative else None,
                    'median_absolute_reduction':median(row['absolute_reduction'] for row in rows),
                    'worst_absolute_reduction':min(row['absolute_reduction'] for row in rows),
                    'median_candidate_error':median(row['candidate_error'] for row in rows),
                    'median_baseline_error':median(row['baseline_error'] for row in rows)})
    return summaries
