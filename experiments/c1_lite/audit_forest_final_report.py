#!/usr/bin/env python3
"""Read-only, experiment-specific audit of the final Forest JSON receipts.

This verifies saved evidence and recomputes small exact witnesses. It does not
reload the 1 GiB cache, rerun native slicing, or prove that an arbitrary report
writer truthfully captured mesh arrays. No production/campaign module is used.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct

REGISTRY_SHA = '0ce1a439bf33b785db9dbb3ea55dbc548933477aca64d2fca64829f660609b52'
POPULATION_SHA = 'f6425b58e74b4f12cb4f8030b5e5be1d5d27ea99513e761b0ec4044ac9e826bc'
SOURCE_SUMMARY_SHA = '74fe3be570ab1119c7c005d23b29505317c66072404274f3689c474cfb9fee67'
SOURCE_INDEX_SHA = '6612c0ceca816517b6e28f6f4ec44b8cdb13ab2f779464a38ea0679a6cb38f1e'
WITHDRAWN_SUMMARY_SHA = '40997b1fa316e8ffecd5ad931c24eb89a887b0e974b8e4c6ad86af2c5e27abff'
FROZEN_SO_SHA = 'f4263a2f47ba5283175a921e49b8867998bdd8124aac810793b34242ec43a3c9'
EFFECTIVE_SHA = '78944c29b77938e035bb1a9c61f4b084cb90144371402e5600d9fb31026daa08'
LOCAL_REASONS = {
    'Actual source boundary is not a strict convex XY graph.': 'NONCONVEX_ACTUAL_XY_BOUNDARY',
    'Actual rounded center is not strictly inside source boundary.': 'ACTUAL_CENTER_NOT_STRICTLY_INSIDE',
}
DEGENERATE_REASON = 'Degenerate actual retained triangle touches source interface.'


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def fr(value):
    return F(value['numerator'], value['denominator']) if isinstance(value, dict) else F(value)


class Reader:
    def __init__(self, max_bytes=100*1024**2):
        self.max_bytes, self.bytes, self.hashes = max_bytes, 0, {}

    def read(self, root, name, expected_sha=None):
        root = Path(root).resolve()
        path = (root/name).resolve()
        require(path.is_relative_to(root), 'Artifact path escapes its attempt directory.')
        size = path.stat().st_size
        require(self.bytes+size <= self.max_bytes, 'JSON audit read budget exceeded.')
        data = path.read_bytes()
        self.bytes += len(data)
        actual = hashlib.sha256(data).hexdigest()
        if expected_sha is not None:
            require(actual == expected_sha, 'Artifact SHA mismatch: '+str(path))
        self.hashes[str(path)] = actual
        return json.loads(data)


def validate_population(summary, index, events, *, expected_count=131, expected_sha=POPULATION_SHA):
    ids = [row['event_id'] for row in index]
    require(len(ids) == expected_count and len(set(ids)) == expected_count,
            'Population has an omitted or duplicated canonical event.')
    require(set(ids) == set(events), 'Index/event identity sets differ.')
    require(digest(sorted(ids)) == expected_sha == summary['expected_event_ids_sha256'],
            'Frozen canonical event set changed.')
    require(summary['canonical_event_denominator'] == expected_count, 'Wrong canonical denominator.')
    require(summary['registry_sha256'] == REGISTRY_SHA, 'Wrong registry provenance.')
    for row in index:
        event = events[row['event_id']]
        require(event['event_id'] == row['event_id'] and event['decision'] == row['decision'],
                'Index/event decision or identity mismatch.')
        if 'runtime_status' in row:
            require(event['runtime']['status'] == row['runtime_status'], 'Index/runtime status mismatch.')


def validate_schedule(schedule, root, *, actual=True):
    root = F(root)
    require(root in (F(3, 2), F(5, 2)), 'Unexpected Forest root.')
    bounds = {k: F(v) for k, v in schedule['bounds'].items()}
    require(bounds == {'lower': root-F(1, 4), 'root': root, 'upper': root+F(1, 4)},
            'The predeclared source window changed.')
    require(schedule['all_camera_frames'] == 64 and schedule['root_excluded_from_natural_rates'] is True,
            'Natural-frame denominator or root exclusion changed.')
    require(schedule['other_events_policy'] == 'UNCHANGED_SAME_ORDINARY_BASELINE', 'Wrong event comparison policy.')
    delta = float.fromhex(schedule['delta_t_hex'])
    origin = float.fromhex(schedule['origin_seconds_hex'])
    require(delta.hex() == '0x1.500053e2d6239p-1' and origin.hex() == float(1/48).hex(),
            'Frozen physical camera phase or native delta-T changed.')
    if actual:
        require(schedule['time_mapping_status'] == 'ACTUAL_NATIVE_INITIALIZATION_VERIFIED',
                'Schedule lacks actual native time mapping.')
    first = 21 if root == F(3, 2) else 37
    hit, selected = list(range(first, first+8)), list(range(first-1, first+9))
    natural, exact = schedule['natural'], schedule['exact_root']
    require(len(natural) == 10 and [q['frame_number'] for q in natural] == selected,
            'Natural schedule is not the fixed eight hits and two neighbours.')
    require(schedule['all_queries'] == natural+[exact], 'all_queries differs from natural plus separate root.')
    require(len({q['key'] for q in schedule['all_queries']}) == 11, 'Duplicate requested query.')
    require(schedule['natural_hit_count'] == 8 and schedule['selected_natural_count'] == 10
            and schedule['hit_frame_numbers'] == hit, 'Natural schedule counts mismatch.')
    for q in natural:
        i = q['frame_number']-1
        require(q['key'] == f'frame_{i+1:04d}' and q['frame_index_zero_based'] == i
                and q['kind'] == 'natural' and q['time_mode'] == 'physical', 'Malformed natural query identity.')
        global_time = float((i+0.5)/24)
        local = float(global_time-origin)
        require(float.fromhex(q['global_camera_time_hex']) == global_time
                and float.fromhex(q['physical_time_hex']) == local
                and F(q['evaluation_tau']) == F.from_float(float(local/delta)),
                'Natural query does not reproduce original camera phase and native division.')
        require(q['active'] is (q['frame_number'] in hit), 'Incorrect natural activity flag.')
    require(exact['key'] == 'root_'+str(root).replace('/', '_') and exact['kind'] == 'exact_root'
            and exact['time_mode'] == 'exact' and exact['active'] is True
            and exact['frame_number'] is None and F(exact['evaluation_tau']) == root,
            'Malformed separate exact-root diagnostic.')
    # This product has at most 55 significant bits; Fraction->float is one RNE
    # rounding, matching the original long-double expression for these roots.
    require(float.fromhex(exact['physical_time_hex']) == float(root*F.from_float(delta)),
            'Exact-root physical binding mismatch.')
    return hit


def validate_identity(receipt):
    require(receipt['version'] == 2 and receipt['encoding'] == 'ORIGINAL_EFFECTIVE_SOURCE_VID',
            'Normalized/old observer identities cannot establish native admission.')
    require(receipt['shifts_order'] == ['node0', 'group0', 'node1', 'group1'], 'Identity shift field order mismatch.')
    shifts = receipt['per_element_shifts']
    require(isinstance(shifts, list) and len(shifts) == 5 and all(isinstance(row, list) and len(row) == 4 for row in shifts),
            'Original identity receipt must include all five by four shifts.')
    require(all(type(x) is int and -(2**31) <= x < 2**31 for row in shifts for x in row), 'Invalid int32 identity shifts.')


def token(query):
    return (query['time_mode'], query['evaluation_tau'] if query['time_mode'] == 'exact' else query['physical_time_hex'])


def mesh_receipts(meshes):
    require(len(meshes) == 5, 'Receipt omitted a native element.')
    for mesh in meshes:
        require(type(mesh['vertices']) is int and mesh['vertices'] >= 0
                and type(mesh['faces']) is int and mesh['faces'] >= 0, 'Invalid native element counts.')
        require(set(mesh['sha256']) == {'vertices', 'faces', 'tags'}
                and all(is_sha(h) for h in mesh['sha256'].values()), 'Missing native array hashes.')


def validate_query_receipt(query, reference, source_summary):
    validate_identity(query['identity_encoding'])
    mesh_receipts(query['baseline'])
    require(tuple(query['query_token']) == token(query['query']), 'Query receipt token mismatch.')
    require(query['observer_enabled_disabled_byte_equal'] is True
            and query['all_events_preserved_shared_baseline'] is True, 'Ordinary baseline changed during query.')
    require(query['frozen_library_baseline_parity'] in ('EXACT_RECEIPT_MATCH', 'SAME_FROZEN_LIBRARY'),
            'Frozen-library parity did not pass.')
    require(reference['status'] == 'FROZEN_ORDINARY_QUERY_RECEIPT'
            and reference['library_sha256'] == FROZEN_SO_SHA
            and tuple(reference['query_token']) == tuple(query['query_token'])
            and reference['meshes'] == query['baseline'], 'Frozen ordinary five-element array hashes do not match.')
    require(reference['registry_sha256'] == REGISTRY_SHA
            and reference['camera_inputs_sha256'] == source_summary['camera_inputs_sha256']
            and reference['effective_inputs_sha256'] == EFFECTIVE_SHA
            and reference['initialization']['actual_delta_t_hex'] == '0x1.500053e2d6239p-1',
            'Frozen reference input binding mismatch.')
    require(reference['identity_enabled'] is False and reference['baseline_only'] is True
            and reference['extra_smooth'] is False and reference['reader_closed'] is True,
            'Frozen reference is not a completed raw ordinary query.')
    require(len(query['counts']) == 5, 'Identity count report omitted an element.')
    for element, (count, mesh) in enumerate(zip(query['counts'], query['baseline'])):
        require(count['element'] == element and count['vertices'] == mesh['vertices']
                and count['faces'] == mesh['faces'] and count['identity_vertices'] == mesh['vertices']
                and count['raw_owners'] >= mesh['faces'], 'Incomplete native identity denominator.')
    rows = query['events']
    require(len({r['event_id'] for r in rows}) == len(rows), 'Duplicate event in one query receipt.')
    for row in rows:
        require(row['status'] == row['case']['audit']['status']
                and token(row['case']['query']) == tuple(query['query_token']), 'Query/event case mismatch.')


def exact_point(values):
    require(len(values) == 3, 'Expected three actual coordinates.')
    result = []
    for value in values:
        require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite actual coordinate.')
        require(struct.unpack('f', struct.pack('f', value))[0] == value, 'Witness coordinate is not represented binary32.')
        result.append(F(value))
    return tuple(result)


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def local_witness(audit):
    saved = audit['local_exact']
    boundary = [exact_point(p) for p in saved['boundary']]
    require(len(boundary) == 4, 'Expected four actual boundary points.')
    center = exact_point(saved['center'])
    turns = [cross2(sub(boundary[(i+1) % 4], boundary[i]),
                    sub(boundary[(i+2) % 4], boundary[(i+1) % 4])) for i in range(4)]
    orientation = 1 if turns[0] > 0 else -1
    inward = [orientation*cross2(sub(boundary[(i+1) % 4], boundary[i]), sub(center, boundary[i])) for i in range(4)]
    require([F(x) for x in saved['turns']] == turns and [F(x) for x in saved['inward']] == inward,
            'Saved exact local values do not match actual coordinate witness.')
    return all(orientation*x > 0 for x in turns), all(x > 0 for x in inward)


def validate_native_rejection(audit, baseline=None):
    require(audit['status'] == 'REJECT' and audit.get('certified_necessary_policy_failure') is True,
            'Native rejection lacks necessary-policy marker.')
    reason = audit.get('reason')
    if audit.get('reason_code') == 'FORBIDDEN_ACTUAL_PATCH_RETAINED_CONTACT':
        return validate_contact_witness(audit, baseline)
    if reason in LOCAL_REASONS:
        require(audit['stage'] == 'actual_local_graph', 'Local witness is assigned to the wrong gate.')
        convex, inside = local_witness(audit)
        require(not convex if reason.startswith('Actual source boundary') else convex and not inside,
                'The claimed local necessary failure is actually satisfied.')
        return LOCAL_REASONS[reason]
    if reason == DEGENERATE_REASON:
        require(audit['stage'] == 'actual_interface', 'Degenerate witness assigned to the wrong gate.')
        require(all(local_witness(audit)), 'Interface failure lacks passing preceding local graph gate.')
        cycle, removed = audit['boundary_actual_ids'], audit['replaced_face_rows']
        require(len(cycle) == len(set(cycle)) == 4 and len(removed) == len(set(removed)) == 2,
                'Malformed source support IDs in rejection witness.')
        ids = audit['failing_face_actual_ids']
        points = [exact_point(p) for p in audit['failing_face_binary32_coordinates']]
        require(len(ids) == len(points) == 3 and audit['failing_face'] not in removed
                and type(audit['failing_face']) is int and audit['failing_face'] >= 0,
                'Claimed retained witness is removed or malformed.')
        shared = sorted(set(ids) & set(cycle))
        require(shared and shared == audit['failing_face_shared_boundary_ids'], 'Degenerate face does not share an actual boundary ID.')
        boundary = [exact_point(p) for p in audit['local_exact']['boundary']]
        for identity, point in zip(ids, points):
            require(type(identity) is int and identity >= 0, 'Malformed actual vertex ID.')
            if identity in cycle:
                require(point == boundary[cycle.index(identity)], 'Shared actual ID has inconsistent coordinates.')
        u, v = sub(points[1], points[0]), sub(points[2], points[0])
        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        require(not any(cross) and [F(x) for x in audit['failing_face_exact_cross']] == list(cross),
                'Claimed degenerate retained triangle has nonzero exact area.')
        return 'PREEXISTING_RETAINED_INTERFACE_DEGENERACY_FIXED_POLICY'
    raise AuditError('UNVERIFIED_REJECTION_WITNESS: '+str(reason))


def validate_source_rejection(event, *, prior_frozen_evidence=False):
    rejection = event['decisive_rejection']
    require(rejection['certified_necessary_policy_failure'] is True, 'Source rejection lacks necessity marker.')
    if rejection['gate'] == 'legacy_selector':
        compiler = event['compiler']
        require(rejection['reason_code'] == 'LEGACY_EXHAUSTIVE_SELECTOR_NO_CANDIDATE'
                and rejection['evidence_binding'] == digest(compiler)
                and compiler['status'] == 'REJECT_FIXED_SOURCE_COMPILER', 'Unbound fixed selector rejection.')
        witness = compiler['decisive_witness']
        require(witness['event_raw_candidates'] > 0 and witness['complete_halo_raw_triangles'] >= witness['event_raw_candidates'],
                'Empty-selector rejection lacks its complete nonempty search domain.')
        require(all(s['event_candidates_complete'] is True and s['halo_complete'] is True for s in event['snapshot_summaries']),
                'Incomplete source domain cannot prove an exhaustive selector rejection.')
        return 'FROZEN_FIXED_SELECTOR_NO_CANDIDATE'
    require(rejection['gate'] == 'requested_source_interface' and rejection['reason_code'] == 'SOURCE_INTERFACE_REJECT',
            'Unsupported source necessary rejection type.')
    require(rejection['compiler_sha256'] == digest(event['compiler'])
            and rejection['schedule_sha256'] == digest(event['schedule']), 'Source interface evidence binding mismatch.')
    require(rejection['native_identity_mismatch_count'] == 0 and rejection['native_ordered_identity_model'] == 'PASS',
            'Source identity mismatch cannot establish native contact failure.')
    witnesses = rejection['interface']['witnesses']
    require(witnesses and rejection['interface']['witness_count'] == len(witnesses), 'Missing source interface witness.')
    # Frozen attempt01 has E != V-1 link-count witnesses. This alone violates
    # one nonempty path, independently of the stored status/compatibility flag.
    if any(w['kind'] == 'RETAINED_LINK_NOT_ONE_PATH' and w['link_edges'] != w['link_vertices']-1 for w in witnesses):
        return 'SOURCE_QUOTIENT_LINK_COUNT_NECESSARY_FAILURE'
    require(prior_frozen_evidence and all(w['kind'] == 'RETAINED_LINK_NOT_ONE_PATH' for w in witnesses),
            'Stored source interface witness needs a stronger independent graph replay.')
    return 'PRIOR_VERIFIED_FROZEN_SOURCE_LINK_FAILURE_NOT_REPLAYED'


def validate_contact_witness(audit, baseline):
    require(baseline is not None and audit['stage'] == 'actual_five_element_exterior'
            and audit['exact_contact_fallback_enabled'] is True and all(local_witness(audit)),
            'Exact contact witness lacks actual query/local gate binding.')
    fallback = audit['exact_contact_fallback']
    proof = fallback['exact_contact']
    require(fallback['status'] == proof['status'] == 'REJECT_POLICY_CONTACT'
            and fallback['baseline_novelty_checked'] is False
            and proof['new_contact_relative_to_baseline_proven'] is False,
            'Exact contact result has wrong status or overclaims baseline novelty.')
    element, other = audit['element'], audit['failing_element']
    require(type(element) is int and type(other) is int and 0 <= element < 5 and 0 <= other < 5,
            'Contact witness has invalid element namespace.')
    index = fallback['fan_triangle_index']
    require(type(index) is int and 0 <= index < 4, 'Invalid fan triangle index.')
    cycle = audit['boundary_actual_ids']
    require(len(cycle) == len(set(cycle)) == 4, 'Malformed contact source cycle.')
    boundary = audit['local_exact']['boundary']
    expected_points = [boundary[index], boundary[(index+1) % 4], audit['local_exact']['center']]
    expected_ids = [[element, cycle[index]], [element, cycle[(index+1) % 4]], [element, baseline[element]['vertices']]]
    require(fallback['fan_triangle_coordinates'] == expected_points and fallback['fan_triangle_ids'] == expected_ids,
            'Forbidden point witness is not a triangle in the current actual fan.')
    require(fallback['retained_triangle_coordinates'] == audit['failing_coordinates']
            and fallback['retained_triangle_ids'] == audit['failing_ids'], 'Retained witness differs from failing actual face.')
    require(0 <= audit['failing_face'] < baseline[other]['faces']
            and (element != other or audit['failing_face'] not in audit['replaced_face_rows']),
            'Forbidden contact witness is not a retained face.')
    points_a = [exact_point(p) for p in expected_points]
    points_b = [exact_point(p) for p in audit['failing_coordinates']]
    ids_a = [tuple(x) for x in expected_ids]
    ids_b = [tuple(x) for x in audit['failing_ids']]
    require(len(points_b) == len(ids_b) == 3 and all(len(x) == 2 and x[0] == other
            and type(x[1]) is int and 0 <= x[1] < baseline[other]['vertices'] for x in ids_b),
            'Retained contact IDs do not name actual vertices of their own element.')
    positions = {}
    for identity, point in zip(ids_a+ids_b, points_a+points_b):
        require(identity not in positions or positions[identity] == point, 'One contact identity has inconsistent coordinates.')
        positions[identity] = point
    input_exact = {'a': [[str(x) for x in p] for p in points_a], 'b': [[str(x) for x in p] for p in points_b],
                   'ids_a': expected_ids, 'ids_b': audit['failing_ids']}
    require(proof['input_exact'] == input_exact and proof['input_sha256'] == digest(input_exact),
            'Exact contact proof input binding mismatch.')
    witness = proof['witness']
    weights_a, weights_b = [F(x) for x in witness['barycentric_a_exact']], [F(x) for x in witness['barycentric_b_exact']]
    point = tuple(F(x) for x in witness['point_exact'])
    require(len(weights_a) == len(weights_b) == len(point) == 3
            and all(x >= 0 for x in weights_a+weights_b)
            and sum(weights_a) == sum(weights_b) == 1, 'Intersection barycentric witness is infeasible.')
    for triangle, weights in ((points_a, weights_a), (points_b, weights_b)):
        require(tuple(sum(weights[i]*triangle[i][j] for i in range(3)) for j in range(3)) == point,
                'Claimed intersection point is not on both actual triangles.')
    shared = sorted(set(ids_a) & set(ids_b))
    require(len(shared) <= 2 and proof['shared_ids'] == [list(x) for x in shared], 'Contact shared-identity policy mismatch.')
    allowed = False
    if len(shared) == 1:
        allowed = point == positions[shared[0]]
    elif len(shared) == 2:
        p, q = (positions[x] for x in shared)
        d, offset = sub(q, p), sub(point, p)
        if not any(d):
            allowed = point == p
        else:
            axis = next(i for i, x in enumerate(d) if x)
            parameter = offset[axis]/d[axis]
            allowed = 0 <= parameter <= 1 and all(x == parameter*y for x, y in zip(offset, d))
    require(not allowed, 'Reported forbidden point lies in the permitted actual shared feature.')
    return 'EXACT_FORBIDDEN_FAN_RETAINED_CONTACT_BASELINE_NOVELTY_NOT_TESTED'


def graph_replay(edges, previous, following):
    edges = [tuple(e) for e in edges]
    degree = Counter(v for e in edges for v in e)
    incoming, outgoing = Counter(b for _, b in edges), Counter(a for a, _ in edges)
    adjacency = {v: set() for v in degree}
    for x, y in edges:
        adjacency[x].add(y)
        adjacency[y].add(x)
    remaining, components = set(degree), []
    while remaining:
        reached, todo = set(), [min(remaining)]
        while todo:
            vertex = todo.pop()
            if vertex not in reached:
                reached.add(vertex)
                todo.extend(adjacency[vertex]-reached)
        components.append(sorted(reached))
        remaining -= reached
    endpoints = {previous, following}
    degrees_ok = ({v for v, n in degree.items() if n == 1} == endpoints
                  and all(n == (1 if v in endpoints else 2) for v, n in degree.items()))
    directed = all(incoming[v] == (0 if v == previous else 1)
                   and outgoing[v] == (0 if v == following else 1) for v in degree)
    connected = len(components) == 1
    return {'degree': dict(sorted(degree.items())),
            'incoming': {v: incoming[v] for v in sorted(degree)},
            'outgoing': {v: outgoing[v] for v in sorted(degree)},
            'components': components, 'link_edges': len(edges), 'link_vertices': len(degree),
            'directed_link_path_compatible': directed, 'degree_and_endpoints_compatible': degrees_ok,
            'connected': connected, 'status': 'PASS' if degrees_ok and directed and connected else 'REJECT'}


def validate_link_document(document, source_events, verification):
    require(document['status'] == 'PASS_INDEPENDENT_SOURCE_LINK_REPLAY'
            and document['full_input_hash_set_equal_frozen'] is True
            and document['input_verification']['input_full_sha256'] == verification['input_full_sha256']
            and document['input_verification']['pass'] is True, 'Source link replay lacks the exact frozen full-input set.')
    expected_ids = {eid for eid, e in source_events.items()
                    if e.get('decisive_rejection', {}).get('reason_code') == 'SOURCE_INTERFACE_REJECT'}
    rows = document['events']
    require(len(rows) == 26 and {e['event_id'] for e in rows} == expected_ids, 'Source graph replay changed its 26-event population.')
    failures, confirmed = 0, set()
    for row in rows:
        original = source_events[row['event_id']]
        require(row['frozen_event'] == original['_audit_artifact_binding']
                and row['source_contract_sha256'] == digest(original['compiler']['source'])
                and F(row['root']) == F(original['root']) and row['element'] == original['element'],
                'Source graph replay event/source/root binding mismatch.')
        require(row['snapshot']['event_candidates_complete'] is True and row['snapshot']['halo_complete'] is True
                and row['snapshot']['event_id'] == row['event_id'] and fr(row['snapshot']['time']) == F(row['root']),
                'Source link graph lacks complete exact-root halo.')
        support = next(p for p in original['compiler']['source']['breakpoint_points'] if fr(p['time']) == F(row['root']))
        require(row['source_support'] == {k: support[k] for k in ('time', 'boundary_cycle', 'source_faces', 'owners')},
                'Source graph support differs from frozen exact-root contract.')
        cycle, element = support['boundary_cycle'], original['element']
        def key(elem, face):
            face = tuple(face)
            return elem, min(face[i:]+face[:i] for i in range(3))
        selected = {key(element, face) for face in support['source_faces']}
        require(len(selected) == 2 and len(set(cycle)) == 4, 'Malformed source graph disk.')
        face_keys, owner_ids, consumed = set(), set(), set()
        links, directions = {v: [] for v in cycle}, Counter()
        diagonal = set(support['source_faces'][0]) & set(support['source_faces'][1])
        require(len(diagonal) == 2 and row['internal_diagonal'] == sorted(diagonal), 'Source graph diagonal mismatch.')
        excluded = []
        for index, face_row in enumerate(row['all_oriented_quotient_faces_in_complete_halo']):
            elem, face = face_row['element'], face_row['source_vertices']
            require(len(face) == 3, 'Malformed source quotient triangle.')
            face_key = key(elem, face)
            require(face_key not in face_keys, 'Duplicated oriented source quotient face.')
            face_keys.add(face_key)
            owners = [tuple(o) for o in face_row['owners']]
            require(owners and all(len(o) == 7 and o[0] == elem for o in owners)
                    and len(set(owners)) == len(owners) and not owner_ids.intersection(owners),
                    'Ghost or repeated source owner class.')
            owner_ids.update(owners)
            suppressed = face_key in selected
            require(face_row['suppressed'] is suppressed, 'Source graph omitted or added a suppressed owner class.')
            if suppressed:
                consumed.update(owners)
            if suppressed or elem != element or not set(cycle).intersection(face):
                continue
            if len(set(face)) != 3 or diagonal <= set(face):
                excluded.append({'face_index': index, 'reason': 'RETAINED_INTERFACE_REPEATED_ID'
                    if len(set(face)) != 3 else 'RETAINED_USES_ORIGINAL_INTERNAL_DIAGONAL'})
                continue
            directions.update((face[i], face[(i+1) % 3]) for i in range(3))
            for vertex in set(cycle).intersection(face):
                i = face.index(vertex)
                links[vertex].append((face[(i+1) % 3], face[(i+2) % 3]))
        require(consumed == {tuple(o) for o in support['owners']} and selected <= face_keys,
                'Source graph suppression did not consume exactly the complete two-face owner classes.')
        require(excluded == row['excluded_from_link_graph'], 'Unexplained omitted retained interface faces.')
        root_query = next(q for q in original['requested_source_interface']['queries'] if q['query']['kind'] == 'exact_root')
        require(row['frozen_root_query'] == root_query['query']
                and row['frozen_root_interface_sha256'] == digest(root_query['interface']), 'Source root graph witness is stale.')
        saved_links = {g['source_vid']: g for g in row['vertex_links']}
        old_links = {g['source_vid']: g for g in root_query['interface']['vertex_link_checks']}
        event_failures = 0
        for i, vertex in enumerate(cycle):
            graph = graph_replay(links[vertex], cycle[i-1], cycle[(i+1) % 4])
            saved = saved_links[vertex]
            require(saved['directed_edges'] == [list(e) for e in links[vertex]]
                    and all(saved[k] == v for k, v in graph.items()), 'Stored source graph differs from its complete quotient faces.')
            require(all(old_links[vertex][k] == graph[k] for k in ('status', 'link_edges', 'link_vertices', 'directed_link_path_compatible')),
                    'Rebuilt graph differs from the original root gate witness.')
            event_failures += graph['status'] == 'REJECT'
        require(event_failures > 0, 'Claimed source graph rejection has no failing link.')
        failures += event_failures
        for edge in row['boundary_edge_checks']:
            x, y = edge['edge']
            require(edge['same_direction'] == directions[x, y] and edge['opposite_direction'] == directions[y, x],
                    'Source graph boundary attachment counts mismatch.')
        confirmed.add(row['event_id'])
    require(document['event_count'] == len(confirmed) and document['failed_link_count'] == failures,
            'Source graph replay summary denominator mismatch.')
    return confirmed


def validate_pass(case, query_receipt, source):
    audit, plan, arrays = case['audit'], case['plan'], case['actual_array_receipt']
    require(audit['status'] == 'PASS' and audit['full_exterior'] is True and all(local_witness(audit)),
            'A source/local-only PASS cannot authorize actual admission.')
    require(audit['continuous_window_admitted'] is False and audit['same_root_group_admitted'] is False,
            'Query result overclaims continuous or group admission.')
    expected = sum(m['faces'] for m in query_receipt['baseline'])-2
    require(audit['expected_retained_faces'] == audit['retained_faces_checked'] == expected
            and sum(audit['certificate_counts'].values()) == expected, 'Retained five-element denominator is incomplete.')
    allowed = {'STRICT_ACTUAL_AABB', 'RELATIVE_EDGE_PLANE', 'RELATIVE_EMPTY_PLANE',
               'RELATIVE_VERTEX_PLANE', 'STRICT_FIXED_PLANE', 'EXACT_FOUR_FAN_TRIANGLE_CONTACT'}
    require(set(audit['certificate_counts']) <= allowed and all(type(n) is int and n >= 0 for n in audit['certificate_counts'].values()),
            'Unrecognized or negative full-exterior certificate count.')
    element = plan['element']
    require(type(element) is int and 0 <= element < 5 and audit['element'] == element, 'Invalid patch element.')
    base = query_receipt['baseline'][element]
    cycle, removed = plan['boundary_actual_ids'], plan['removed_face_rows']
    require(cycle == audit['boundary_actual_ids'] and removed == audit['replaced_face_rows']
            and plan['center'] == audit['local_exact']['center']
            and F(audit['tau']) == F(case['query']['evaluation_tau']),
            'Actual array plan is not the geometry-certified center/cycle/source rows at this query.')
    require(len(cycle) == len(set(cycle)) == 4 and all(0 <= x < base['vertices'] for x in cycle), 'Malformed patch cycle.')
    require(len(removed) == len(set(removed)) == 2 and all(0 <= x < base['faces'] for x in removed), 'Malformed source face rows.')
    require(plan['new_center_id'] == plan['baseline_vertex_count'] == base['vertices']
            and plan['baseline_face_count'] == base['faces'], 'Patch counts differ from actual baseline.')
    require(plan['fan_faces'] == [[cycle[i], cycle[(i+1) % 4], base['vertices']] for i in range(4)], 'Oriented fan rows mismatch.')
    require(plan['source_digest'] == audit['source_digest'] == digest(source), 'Actual plan source binding mismatch.')
    tau = F(case['query']['evaluation_tau'])
    support = next((p for p in source['breakpoint_points'] if fr(p['time']) == tau), None)
    if support is None:
        support = next((p for p in source['segments'] if fr(p['t0']) < tau < fr(p['t1'])), None)
    require(support is not None, 'Actual query has no source owner contract.')
    owners = [tuple(x) for x in plan['consumed_owners']]
    require(len(owners) == len(set(owners)) and all(len(o) == 7 and o[0] == element for o in owners)
            and set(owners) == {tuple(x) for x in support['owners']}
            and plan['consumed_owners'] == audit['consumed_owners'], 'Actual plan did not consume exactly all authorized owners.')
    require(arrays['status'] == 'ACTUAL_ARRAYS_CONSTRUCTED_NOT_PUBLISHED'
            and all(arrays[k] is True for k in ('old_vertices_and_tags_byte_identical',
                 'all_retained_face_rows_byte_identical', 'other_elements_unchanged_by_object_identity')),
            'Actual output lacks exterior-array invariance verification.')
    require(arrays['element'] == element and arrays['vertices_before'] == base['vertices']
            and arrays['faces_before'] == base['faces'] and arrays['vertices_after'] == base['vertices']+1
            and arrays['faces_after'] == base['faces']+2, 'Actual output count change mismatch.')
    require(arrays['output']['vertices'] == arrays['vertices_after'] and arrays['output']['faces'] == arrays['faces_after']
            and set(arrays['output']['sha256']) == {'vertices', 'faces', 'tags'}
            and all(is_sha(h) for h in arrays['output']['sha256'].values()), 'Actual output array receipt missing.')


def validate_committed_schedule(event):
    runtime = event['runtime']
    required = {q['key']: q for q in event['schedule']['all_queries']}
    cases = runtime['cases']
    require(len(required) == len(cases) == 11 and {c['query']['key'] for c in cases} == set(required),
            'ADMIT lacks exactly all eleven unique requested queries.')
    require(all(c['query'] == required[c['query']['key']]
                and c['audit']['status'] == ('PASS' if c['query']['active'] else 'BASELINE_OUTSIDE_WINDOW') for c in cases),
            'ADMIT includes a failed, stale or wrong-activity query.')
    require(runtime['status'] == 'COMMITTED_REQUESTED_SCHEDULE'
            and runtime['published_partial_results'] is False
            and runtime['publication_kind'] == 'COMPACT_VERIFIED_ACTUAL_ARRAY_SCHEDULE_MANIFEST'
            and runtime['source_inputs_unchanged'] is True
            and runtime['whole_mesh_arrays_discarded_after_verification'] is True,
            'Schedule did not atomically commit after input verification.')


def summarize(events):
    counts = Counter(e['decision'] for e in events)
    n, admitted, unknown = len(events), counts['ADMITTED_REQUESTED_SCHEDULE'], counts['UNKNOWN']
    frames = sorted({q['frame_number'] for e in events if e['decision'] == 'ADMITTED_REQUESTED_SCHEDULE'
                     for q in e['schedule']['natural'] if q['active']})
    incidences = sum(sum(q['active'] for q in e['schedule']['natural']) for e in events if e['decision'] == 'ADMITTED_REQUESTED_SCHEDULE')
    return {'admitted': admitted, 'rejected_fixed_policy': counts['REJECTED_FIXED_POLICY'], 'unknown': unknown,
            'canonical_event_denominator': n, 'policy_admission_rate': admitted/n if not unknown else None,
            'policy_admission_rate_bounds': [admitted/n, (admitted+unknown)/n],
            'actual_unique_modified_natural_frames': frames, 'actual_event_frame_modifications': incidences,
            'actual_unique_modified_natural_frame_rate': len(frames)/64,
            'candidate_event_frame_denominator': 131*8,
            'actual_modified_candidate_event_frame_rate': incidences/(131*8),
            'root_diagnostics_excluded': True}


def audit_attempt(attempt, source_attempt, source_link_witness=None):
    attempt, source_attempt = Path(attempt).resolve(), Path(source_attempt).resolve()
    reader = Reader()
    auditor_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = {'schema': 'forest-final-independent-report-audit-v1', 'status': 'INVALID_REPORT',
              'policy_admission_rate': None, 'errors': [], 'rejection_reasons': {},
              'scope': 'Saved-receipt consistency plus exact small rejection witnesses; not native re-execution, all-time admission, same-root union, or image quality.'}
    try:
        if (attempt/'scope_correction.json').exists():
            correction = reader.read(attempt, 'scope_correction.json')
            require(not correction.get('supersedes_admission_interpretation_only')
                    and not str(correction.get('status', '')).startswith('INVALID'), 'Attempt admission interpretation has been withdrawn by scope correction.')
        source_summary = reader.read(source_attempt, 'summary.json', SOURCE_SUMMARY_SHA)
        source_index = reader.read(source_attempt, 'events_index.json', SOURCE_INDEX_SHA)
        source_events = {row['event_id']: reader.read(source_attempt, row['artifact']['path'], row['artifact']['sha256']) for row in source_index}
        validate_population(source_summary, source_index, source_events)
        verification = reader.read(source_attempt, 'input_verification.json')
        require(verification['pass'] is True and verification['executed_sources_unchanged'] is True and not verification['changed'],
                'Frozen source input verification did not pass.')
        require(source_summary['e2_regression'] == 'PASS_FULL_PARSER_AND_FROZEN_E2_PARITY', 'Missing E2 source adapter calibration.')
        summary = reader.read(attempt, 'summary.json')
        require(reader.hashes[str(attempt/'summary.json')] != WITHDRAWN_SUMMARY_SHA, 'Withdrawn native02 summary copied without its scope correction.')
        require(summary['source_campaign_sha256'] == SOURCE_SUMMARY_SHA, 'Native report belongs to a different source campaign.')
        require(summary['status'] in ('COMPLETE_REAL_FIXED_POLICY_RATE', 'UNRESOLVED_ACTUAL_QUERIES'), 'Native attempt did not finish a valid report.')
        index = reader.read(attempt, 'events_index.json')
        events = {row['event_id']: reader.read(attempt, row['artifact']['path'], row['artifact']['sha256']) for row in index}
        validate_population(summary, index, events)
        require(set(events) == set(source_events), 'Cross-stage canonical population differs.')
        refs = {}
        for path in sorted((attempt/'references').glob('*.json')):
            ref = reader.read(attempt, path.relative_to(attempt))
            key = tuple(ref['query_token'])
            require(key not in refs, 'Duplicate frozen reference query.')
            refs[key] = ref
        queries, cases = {}, {}
        for path in sorted((attempt/'queries').glob('*.json')):
            query = reader.read(attempt, path.relative_to(attempt))
            key = tuple(query['query_token'])
            require(key not in queries and key in refs, 'Duplicate query or missing independently saved frozen reference.')
            validate_query_receipt(query, refs[key], source_summary)
            queries[key] = query
            for row in query['events']:
                require(row['event_id'] in events, 'Query evaluated an event outside the frozen population.')
                cases[(row['event_id'], key)] = row['case']
        require(len(queries) == summary['unique_native_queries'], 'Unique native query denominator mismatch.')
        reasons, validated = Counter(), []
        source_rows = {row['event_id']: row for row in source_index}
        graph_replayed = set()
        if source_link_witness is not None:
            path = Path(source_link_witness).resolve()
            graph_document = reader.read(path.parent, path.name)
            graph_sources = {eid: dict(e, _audit_artifact_binding=source_rows[eid]['artifact']) for eid, e in source_events.items()}
            graph_replayed = validate_link_document(graph_document, graph_sources, verification)
        for event_id, event in events.items():
            original = source_events[event_id]
            require(event['source_artifact'] == source_rows[event_id]['artifact'], 'Native event has an unbound source artifact.')
            require(event['compiler'] == original['compiler'], 'Native stage changed the frozen source construction.')
            require(event['schedule_sha256'] == digest(event['schedule']), 'Event schedule hash mismatch.')
            validate_schedule(event['schedule'], event['root'], actual=original['decision'] != 'REJECTED_FIXED_POLICY')
            if original['decision'] == 'REJECTED_FIXED_POLICY':
                require(event['decision'] == original['decision'] and event['decisive_rejection'] == original['decisive_rejection']
                        and not event['runtime_attempted'], 'Native stage changed a frozen source rejection.')
                source_reason = validate_source_rejection(original, prior_frozen_evidence=True)
                reasons['SOURCE_QUOTIENT_COMPLETE_GRAPH_REPLAY' if event_id in graph_replayed else source_reason] += 1
                validated.append(event)
                continue
            require(original['decision'] == 'UNKNOWN' and event['runtime_attempted'] is True, 'Missing actual native attempt for a source survivor.')
            runtime = event['runtime']
            require(runtime['published_partial_results'] is False and runtime['source_sha256'] == digest(event['compiler']['source'])
                    and runtime['schedule_sha256'] == event['schedule_sha256'], 'Runtime source/schedule binding or atomicity mismatch.')
            expected = {q['key']: q for q in event['schedule']['all_queries']}
            seen = set()
            for case in runtime['cases']:
                q, audit = case['query'], case['audit']
                require(q['key'] in expected and q == expected[q['key']] and q['key'] not in seen, 'Duplicate, stale or unrequested native case.')
                seen.add(q['key'])
                key = token(q)
                require(cases.get((event_id, key)) == case, 'Event case differs from shared query artifact.')
                if q['active']:
                    require(F(audit['tau']) == F(q['evaluation_tau']) and audit['element'] == event['element']
                            and audit['source_digest'] == digest(event['compiler']['source']),
                            'Actual geometric audit is stale or belongs to another event/query.')
                    require(audit['source_vid_encoding_version'] == 2
                            and audit['source_vid_encoding'] == 'ORIGINAL_EFFECTIVE_SOURCE_VID'
                            and audit['source_vid_shifts'] == queries[key]['identity_encoding']['per_element_shifts']
                            and audit['source_vid_shifts_applied_by_python'] is False,
                            'Event audit used a different/native-normalized identity model.')
                if audit['status'] == 'PASS':
                    validate_pass(case, queries[key], event['compiler']['source'])
                elif audit['status'] == 'BASELINE_OUTSIDE_WINDOW':
                    require(q['active'] is False and case['output_equals_baseline'] is True
                            and case['baseline'] == queries[key]['baseline'], 'Outside-window output changed.')
                else:
                    require(audit['status'] in ('REJECT', 'UNKNOWN'), 'Unsupported native case status.')
            if event['decision'] == 'ADMITTED_REQUESTED_SCHEDULE':
                validate_committed_schedule(event)
                require(seen == set(expected) and len(runtime['cases']) == 11
                        and all(c['audit']['status'] in ('PASS', 'BASELINE_OUTSIDE_WINDOW') for c in runtime['cases']),
                        'ADMIT lacks exactly all eleven successful requested queries.')
                require(runtime['status'] == 'COMMITTED_REQUESTED_SCHEDULE'
                        and runtime['publication_kind'] == 'COMPACT_VERIFIED_ACTUAL_ARRAY_SCHEDULE_MANIFEST'
                        and runtime['source_inputs_unchanged'] is True
                        and runtime['whole_mesh_arrays_discarded_after_verification'] is True, 'Schedule did not atomically commit after input verification.')
            elif event['decision'] == 'REJECTED_FIXED_POLICY':
                decisive = event['decisive_rejection']
                require(decisive['certified_necessary_policy_failure'] is True and decisive['gate'] == 'actual_requested_query'
                        and decisive['witness'] in runtime['cases'] and decisive['witness']['query']['active'] is True,
                        'Native rejection does not identify an actual required-query case.')
                reasons[validate_native_rejection(decisive['witness']['audit'], queries[token(decisive['witness']['query'])]['baseline'])] += 1
                require(runtime['status'] == 'BASELINE_ENTIRE_REQUESTED_SCHEDULE', 'Rejected schedule did not remain entirely baseline.')
            else:
                require(event['decision'] == 'UNKNOWN' and runtime['status'] == 'UNKNOWN_REQUESTED_SCHEDULE', 'Malformed unresolved event.')
            validated.append(event)
        require(len(cases) == sum(len(e['runtime'].get('cases', [])) for e in events.values()), 'Unreferenced event/query cases remain.')
        for name in ('input_verification', 'final_input_verification'):
            record = summary[name]
            require(record['status'] == 'PASS' and record['private_nonlog_inputs_unchanged'] is True
                    and record['original_files_unchanged'] == 35 and record['original_bytes'] == 1096080433
                    and record['original_input_content_sha256'] == 'aca6fd25083a4fe6c60fec296be3e7b8558731e2e566b08f2c6e25ee45488a71',
                    'Original/private input verification is incomplete or changed.')
        require(summary['private_cache_removed'] is True, 'Private native cache cleanup did not complete.')
        measured = summarize(validated)
        for name in ('admitted', 'rejected_fixed_policy', 'unknown', 'canonical_event_denominator', 'policy_admission_rate',
                     'policy_admission_rate_bounds', 'actual_unique_modified_natural_frames', 'actual_event_frame_modifications'):
            require(summary[name] == measured[name], 'Summary disagrees with independently counted '+name)
        require(summary['all_events_resolved'] is (measured['unknown'] == 0)
                and summary['source_gate_rejections'] == source_summary['rejected_fixed_policy'] == 27
                and summary['actual_query_rejections'] == measured['rejected_fixed_policy']-27,
                'Cross-stage rejection/resolution denominator mismatch.')
        require(summary['runtime_events_attempted'] == 104 and summary['same_root_union_rate'] is None,
                'Runtime attempted/group denominator mismatch.')
        require(summary['candidate_natural_schedule'] == source_summary['candidate_schedule'], 'Candidate exposure population changed.')
        output.update(status='PASS_SAVED_EVIDENCE_AUDIT', **measured, rejection_reasons=dict(sorted(reasons.items())),
                      identity_encoding='VERSION_2_ORIGINAL_EFFECTIVE_SOURCE_VID_ALL_QUERIES',
                      frozen_baseline_queries_verified=len(queries), source_gate_rejections=27,
                      native_events_attempted=104, original_input_verification='SAVED_RECEIPTS_PASS',
                      source_link_events_independently_graph_replayed=len(graph_replayed),
                      private_cache_removed=True,
                      conditional_runtime_commit_rate=(measured['admitted']/104 if not measured['unknown'] else None),
                      observed_committed_fraction_of_attempted_lower_bound=measured['admitted']/104,
                      limitation=('Private removal and full arrays are verified through bound saved receipts, not re-observed here. '
                          + ('All 26 source link graphs were rebuilt from complete saved oriented quotient faces; the one exhaustive-selector rejection inherits the verified frozen compiler evidence.'
                             if len(graph_replayed) == 26 else 'Source link failures without edge lists inherit prior verified frozen attempt01 evidence; they are not graph-replayed here.')))
    except (AuditError, KeyError, TypeError, ValueError, OSError, OverflowError, StopIteration) as error:
        output['errors'].append(type(error).__name__+': '+str(error))
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != auditor_sha:
        output.update(status='INVALID_REPORT', policy_admission_rate=None)
        output['errors'].append('Auditor source changed during this read-only audit.')
    output['auditor_source_sha256'] = auditor_sha
    output['audited_files_sha256'] = dict(sorted(reader.hashes.items()))
    output['json_bytes_read'] = reader.bytes
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--source-attempt', type=Path, required=True)
    parser.add_argument('--source-link-witness', type=Path)
    parser.add_argument('--output', type=Path, help='Fresh JSON file; existing files are never overwritten.')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('--output must be a new file.')
    report = audit_attempt(args.attempt, args.source_attempt, args.source_link_witness)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if args.output is not None:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded)
        print(json.dumps({k: report.get(k) for k in ('status', 'admitted', 'rejected_fixed_policy', 'unknown', 'errors')}, sort_keys=True))
    else:
        print(encoded, end='')
    return 0 if report['status'] == 'PASS_SAVED_EVIDENCE_AUDIT' else 2


if __name__ == '__main__':
    raise SystemExit(main())
