"""B: independently checked two-layer fan mechanisms, never a native scene."""
from copy import deepcopy
from fractions import Fraction as F

import numpy as np

from forest_native_patch import spec_from_source, audit_query
from forest_component_union import compile_union
from forest_native_campaign import mesh_receipt
from test_window_runtime import fixture, fake_runtime


PROGRESS = {}


def require(condition, message):
    PROGRESS['last_requirement'] = {'failure_message': message, 'passed': bool(condition)}
    if not condition:
        raise RuntimeError('B mechanism mismatch: '+message)


def snapshot(centers):
    meshes = [(np.empty((0, 3), float), np.empty((0, 3), np.int32), np.empty(0, np.int32)) for _ in range(5)]
    vids = [np.empty((0, 4), np.int32) for _ in range(5)]
    owners = [np.empty((0, 11), np.int32) for _ in range(5)]
    documents = []
    for element in (0, 1):
        mesh, vertex_ids, raw_owners, _, document = fixture()
        v, f, tags = mesh
        v = np.ascontiguousarray(v+np.asarray([-1, -1, element]))
        vertex_ids[:, (0, 2)] += 1000*element
        raw_owners[:, 0] = element
        document = deepcopy(document)
        labels = [f'{i+1000*element}:0|{i+100+1000*element}:0' for i in range(4)]
        document['element'] = element
        document['boundary_cycle'] = labels
        for row in document['segments']+document['breakpoint_points']:
            for owner in row['owners']:
                owner[0] = element
            row['boundary_cycle'] = labels
            row['source_faces'] = [[labels[int(i)] for i in face] for face in f[:2]]
        for name in ('lower', 'upper'):
            document['anchors'][name]['position'] = [0, 0, element]
        document['anchors']['root']['position'] = [0, 0, str(centers[element])]
        meshes[element], vids[element], owners[element] = (v, f, tags), vertex_ids, raw_owners
        documents.append(document)
    shifts = np.zeros((5, 4), np.int32)
    for array in [a for mesh in meshes for a in mesh]+vids+owners+[shifts]:
        array.flags.writeable = False
    result = dict(meshes=tuple(meshes), vertex_ledgers=tuple(vids), owner_ledgers=tuple(owners),
                  identity_status=1, source_vid_encoding_version=2,
                  source_vid_encoding='ORIGINAL_EFFECTIVE_SOURCE_VID', source_vid_shifts=shifts)
    return result, [spec_from_source(document) for document in documents]


def exact_point(point):
    return tuple(F.from_float(float(x)) for x in point)


def bind_plan_geometry(state, records, centers, progress):
    """Bind every fan sector to the represented baseline and returned plan."""
    fans = {}
    progress['plan_geometry'] = []
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    for record in records:
        eid, plan = record['event_id'], record['plan']
        element = plan['element']
        v = state['meshes'][element][0]
        cycle = plan['boundary_actual_ids']
        boundary = tuple(exact_point(v[i]) for i in cycle)
        center = exact_point(plan['center'])
        positions = {i: point for i, point in zip(cycle, boundary)}
        positions[plan['new_center_id']] = center
        triangles = tuple(tuple(positions[i] for i in face) for face in plan['fan_faces'])
        evidence = {'event_id': eid, 'element': element,
                    'boundary_exact': [list(map(str, p)) for p in boundary],
                    'center_exact': list(map(str, center)),
                    'fan_triangles_exact': [[list(map(str, p)) for p in tri] for tri in triangles]}
        progress['plan_geometry'].append(evidence)
        require(boundary == tuple((F(x), F(y), F(element)) for x, y in corners),
                eid+' actual boundary differs from the declared square and layer')
        require(center == (F(0), F(0), centers[element]),
                eid+' actual plan center differs from the declared fixture')
        expected = tuple((boundary[i], boundary[(i+1) % 4], center) for i in range(4))
        require(triangles == expected, eid+' actual plan does not contain all four declared fan sectors')
        evidence['all_four_sectors_bound'] = True
        fans[eid] = triangles
    return fans


def analytic_witness(fans, progress):
    weights = (F(1, 6), F(1, 6), F(2, 3))
    target = (F(1, 3), F(0), F(1, 2))
    rows = []
    progress['analytic_witness_attempts'] = rows
    for eid in ('lower', 'upper'):
        triangle = fans[eid][1]
        reconstructed = tuple(sum(weights[i]*triangle[i][axis] for i in range(3)) for axis in range(3))
        rows.append({'event_id': eid, 'sector': 1,
                     'triangle': [[str(x) for x in p] for p in triangle],
                     'reconstructed': list(map(str, reconstructed))})
        require(reconstructed == target and min(weights) > 0 and sum(weights) == 1,
                'independent rational witness is not strictly inside both actual plan triangles')
    return {'point': list(map(str, target)), 'barycentric': list(map(str, weights)),
            'strictly_interior_to_both': True, 'triangles': rows,
            'derived_from_actual_plan_geometry_not_contact_checker': True,
            'all_four_plan_sectors_bound_to_explicit_fixture': True}


def compare_arrays(a, b):
    return all(x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
               for mx, my in zip(a, b) for x, y in zip(mx, my))


def evaluate(name, centers, conflict):
    progress = {'phase': 'snapshot', 'centers_exact': list(map(str, centers)),
                'single_attempts': []}
    PROGRESS.setdefault('B_contact', {})[name] = progress
    state, specs = snapshot(centers)
    before = mesh_receipt(state['meshes'])
    progress['baseline'] = before
    records, singles = [], []
    for element, spec in enumerate(specs):
        progress.update(phase='single_audit', current_element=element)
        plan, certificate = audit_query(state, spec, F(1), full_exterior=True, exact_contact_fallback=True)
        progress['single_attempts'].append({'element': element, 'plan': plan, 'certificate': certificate})
        singles.append(certificate)
        require(certificate['status'] == 'PASS' and plan is not None,
                name+f' individual element {element} did not PASS: '+str(certificate))
        require(certificate['retained_faces_checked'] == 18 and certificate['expected_retained_faces'] == 18,
                name+' single-item certificate did not cover the complete other layer and collar')
        require(mesh_receipt(state['meshes']) == before, name+' single-item audit mutated baseline')
        records.append({'event_id': ('lower', 'upper')[element], 'component_id': 'declared_atomic_pair', 'plan': plan})
        progress['individual_certificates_passed'] = len(records)
    progress['phase'] = 'bind_actual_plan_geometry'
    fans = bind_plan_geometry(state, records, centers, progress)
    independent = {'opposite_baseline_vertical_gap_at_least': '1/4' if conflict else '3/4',
                   'safe_union_vertical_gap_at_least': None,
                   'actual_plan_geometry': progress['plan_geometry']}
    if conflict:
        independent['intersection_witness'] = analytic_witness(fans, progress)
    progress['independent_geometry'] = independent
    progress['phase'] = 'joint_union'
    result, joint = compile_union(state['meshes'], records,
                                  expected_components={'declared_atomic_pair': ['lower', 'upper']})
    progress['joint'] = joint
    progress['joint_output_present'] = result is not None
    require(mesh_receipt(state['meshes']) == before, name+' union mutated baseline')
    if conflict:
        require(result is None and joint['status'] == 'REJECT_POLICY_CONTACT',
                'crossing union did not return expected contact rejection: '+str(joint))
        require(joint['failing_pair']['relation'] == 'Q_Q' and bool(joint['failing_pair']['exact_contact']['witness']),
                'joint rejection lacks a replacement/replacement contact witness')
    else:
        require(result is not None and joint['status'] == 'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED',
                'safe union did not PASS: '+str(joint))
        require(joint['source_faces_consumed_once'] == 4 and joint['raw_owners_consumed_once'] == 6,
                'safe union owner/face consumption mismatch')
        progress['phase'] = 'safe_permutation'
        reverse, reversed_report = compile_union(state['meshes'], list(reversed(records)),
                                      expected_components={'declared_atomic_pair': ['upper', 'lower']})
        progress['permuted_joint'] = reversed_report
        require(reverse is not None and compare_arrays(result, reverse), 'permutation changed safe union')
        progress['phase'] = 'safe_actual_output_geometry'
        progress['actual_output_geometry'] = []
        output_fans = {}
        for eid in ('lower', 'upper'):
            mapping = joint['event_mapping'][eid]
            v, f, _ = result[mapping['element']]
            triangles = tuple(tuple(exact_point(p) for p in v[f[row]])
                              for row in mapping['fan_face_rows_by_sector'])
            evidence = {'event_id': eid,
                        'fan_triangles_exact': [[list(map(str, p)) for p in tri] for tri in triangles]}
            progress['actual_output_geometry'].append(evidence)
            require(triangles == fans[eid], eid+' actual output differs from its four bound plan sectors')
            evidence['all_four_sectors_match_bound_plan'] = True
            output_fans[eid] = triangles
        # The validated square/corner/center projections give phi=1-max(|x|,|y|)
        # on all four sectors. Derive both height functions from emitted values.
        base_a, center_a = output_fans['lower'][0][0][2], output_fans['lower'][0][2][2]
        base_b, center_b = output_fans['upper'][0][0][2], output_fans['upper'][0][2][2]
        offset = base_b-base_a
        coefficient = (center_b-base_b)-(center_a-base_a)
        minimum_gap = min(offset, offset+coefficient)
        independent['global_separation'] = {
            'domain': '[-1,1]^2', 'phi': '1-max(abs(x),abs(y))', 'phi_range': ['0', '1'],
            'lower_height_constant': str(base_a), 'lower_height_phi_coefficient': str(center_a-base_a),
            'upper_height_constant': str(base_b), 'upper_height_phi_coefficient': str(center_b-base_b),
            'gap_constant': str(offset), 'gap_phi_coefficient': str(coefficient),
            'minimum_gap': str(minimum_gap), 'bound_to_all_actual_output_fan_sectors': True}
        require(offset == 1 and coefficient == F(-1, 2) and minimum_gap == F(1, 2),
                'safe actual output does not establish the declared global separation')
        independent['safe_union_vertical_gap_at_least'] = str(minimum_gap)
        independent['actual_output_geometry'] = progress['actual_output_geometry']
    report = {'name': name, 'synthetic_not_beb1_event': True, 'query': '1',
              'baseline': before, 'single_certificates': singles,
              'same_snapshot_source_and_policy_bound_by_caller': True,
              'joint': joint, 'baseline_unchanged': True,
              'independent_geometry': independent}
    progress.update(phase='complete', status='PASS', baseline_unchanged=True)
    return report, state, records


def dependency_controls(state, records):
    progress = {'phase': 'missing_membership',
                'scope': 'MISSING_ATOMIC_MEMBERSHIP_AND_SHARED_BOUNDARY_SUPPORT_ONLY',
                'isolated_raw_owner_reuse_tested': False}
    PROGRESS['B_dependency'] = progress
    before = mesh_receipt(state['meshes'])
    output, missing = compile_union(state['meshes'], records[:1],
                        expected_components={'declared_atomic_pair': ['lower', 'upper']})
    progress['missing_user_atomic_member'] = missing
    require(output is None and missing['status'] == 'UNKNOWN_INPUT_OR_PROOF' and
            'component membership' in missing['reason'], 'missing member was not refused')
    duplicate = deepcopy(records[0])
    duplicate['event_id'] = 'lower_duplicate'
    progress['phase'] = 'shared_boundary_support'
    output, shared = compile_union(state['meshes'], [records[0], duplicate])
    progress['duplicate_boundary_support'] = shared
    require(output is None and shared['status'] == 'UNSUPPORTED_SHARED_SUPPORT' and
            'boundary identity' in shared['reason'], 'shared boundary support was not refused')
    require(mesh_receipt(state['meshes']) == before, 'dependency negative mutated baseline')
    progress.update(phase='complete', status='PASS', not_a_minimal_dependency_graph_claim=True,
                    baseline_unchanged=True)
    return progress


def publication():
    progress = {'phase': 'runtime_schedule', 'scope': 'SYNTHETIC_SLICER_REAL_WINDOW_RUNTIME'}
    PROGRESS['B_publication'] = progress
    runtime, baseline, spec = fake_runtime(bad_at=F(3, 2))
    output, report = runtime.run([F(1, 2), F(3, 2), F(2)], spec)
    progress['report'] = report
    require(report['status'] == 'BASELINE_ENTIRE_SCHEDULE' and report['discarded_proposals'] >= 1,
            'late injected identity failure did not discard prepared candidates')
    require(not report['published_partial_results'] and compare_arrays(output, [baseline]*3),
            'late failure leaked partial query results')
    require(runtime._slice.call_count == 3, 'publication did not acquire all ordinary baselines')
    progress.update(phase='complete', status='PASS',
                    expected_failure_injection='identity unavailable at second query',
                    baseline_batch_preserved=True, not_a_geometric_joint_certificate=True)
    return progress


def run():
    PROGRESS.clear()
    PROGRESS.update(status='RUNNING', B_contact={})
    crossing, _, _ = evaluate('crossing', (F(3, 4), F(1, 4)), True)
    witness = crossing['independent_geometry']['intersection_witness']
    safe, state, records = evaluate('safe', (F(1, 4), F(3, 4)), False)
    controls = dependency_controls(state, records)
    publication_report = publication()
    PROGRESS['status'] = 'PASS'
    return {'status': 'PASS', 'B_contact': {'crossing': crossing, 'safe': safe, 'independent_witness': witness},
            'B_dependency': controls, 'B_publication': publication_report,
            'no_certificate_reuse_optimization': True,
            'performance_warning': 'Safe pair is AABB-excluded; rejected pair stops early. Not comparable timing populations.'}

