"""Necessary fixed-source gates at requested queries, never native admission.

The adapter partitions every integer time predicate. A complete snapshot at
the corresponding open-cell representative has identical emission, ownership
and SourceVID incidence to the requested time; no geometric interpolation
or floating-point runtime identity claim follows from that statement.
"""
from collections import defaultdict
from fractions import Fraction as F

from interface_topology import audit_interface
from window_source import fr


def support_for_query(source, tau):
    tau = F(tau)
    for point in source['breakpoint_points']:
        if fr(point['time']) == tau:
            return point, tau, 'EXACT_SINGLETON'
    for segment in source['segments']:
        a, b = fr(segment['t0']), fr(segment['t1'])
        if a < tau < b:
            return segment, (a+b)/2, 'CONSTANT_INTEGER_PREDICATE_OPEN_CELL'
    raise ValueError('Requested active query is outside the source partition.')


def face_key(element, face):
    face = tuple(face)
    return element, min(face[i:]+face[:i] for i in range(3))


def audit_snapshot(snapshot, support, *, element):
    """Use complete actual-emission halo; do not silently drop owner replicas."""
    base = {'status': 'UNKNOWN', 'certified_necessary_policy_failure': False,
            'scope': 'Necessary fixed source-quotient contract; not actual binary32 mesh admission.'}
    if not snapshot.event_candidates_complete or not snapshot.halo_complete:
        return {**base, 'reason_code': 'INCOMPLETE_SOURCE_HALO'}
    expected_owners = {tuple(x) for x in support['owners']}
    if len(expected_owners) != len(support['owners']) or any(x[0] != element for x in expected_owners):
        return {**base, 'reason_code': 'MALFORMED_SOURCE_OWNER_CONTRACT'}
    groups = defaultdict(list)
    emitted = {}
    for triangle in snapshot.actual_raw_triangles:
        owner = tuple(triangle.reference.values())
        if owner in emitted:
            return {**base, 'reason_code': 'DUPLICATED_READER_OWNER', 'owner': list(owner)}
        emitted[owner] = triangle
        face = tuple(v.text() for v in triangle.source_vertices)
        groups[face_key(triangle.reference.element, face)].append(owner)
    missing = sorted(expected_owners-set(emitted))
    if missing:
        return {**base, 'status': 'REJECT', 'reason_code': 'SOURCE_OWNER_NOT_RAW_EMITTED',
                'certified_necessary_policy_failure': True,
                'missing_owners': [list(x) for x in missing]}
    selected_keys = {face_key(element, face) for face in support['source_faces']}
    if len(selected_keys) != 2:
        return {**base, 'reason_code': 'MALFORMED_SELECTED_SOURCE_DISK'}
    actual_keys = {face_key(element, tuple(v.text() for v in emitted[o].source_vertices))
                   for o in expected_owners}
    if actual_keys != selected_keys:
        return {**base, 'reason_code': 'RAW_READER_SOURCE_CONTRACT_DISAGREEMENT',
                'actual_keys': [(e, list(f)) for e, f in sorted(actual_keys)]}
    complete_owners = {o for key in selected_keys for o in groups[key]}
    if complete_owners != expected_owners:
        return {**base, 'status': 'REJECT', 'reason_code': 'PARTIAL_RAW_OWNER_CLASS_SUPPRESSION',
                'certified_necessary_policy_failure': True,
                'unconsumed_owners': [list(x) for x in sorted(complete_owners-expected_owners)]}
    # Quotient each oriented face once; raw replicas are not link multiplicity.
    retained = [{'element': e, 'source_vertices': list(face),
                 'owners': [list(x) for x in sorted(owners)]}
                for (e, face), owners in sorted(groups.items()) if (e, face) not in selected_keys]
    interface = audit_interface(support['boundary_cycle'], support['source_faces'], retained, element=element)
    identity_mismatches = []
    boundary = set(support['boundary_cycle'])
    for triangle in snapshot.actual_raw_triangles:
        if triangle.reference.element != element or not boundary.intersection(v.text() for v in triangle.source_vertices):
            continue
        owner = tuple(triangle.reference.values())
        detail = snapshot.details_by_owner.get(owner)
        if detail is None or not detail.get('native_legacy_identity_equal'):
            identity_mismatches.append(list(owner))
    return {**base, 'status': interface['status'],
            'reason_code': 'SOURCE_INTERFACE_' + interface['status'],
            'certified_necessary_policy_failure': interface['status'] == 'REJECT',
            'complete_owner_count': len(complete_owners),
            'quotient_face_count_in_halo': len(groups), 'interface': interface,
            'native_ordered_identity_model': 'PASS' if not identity_mismatches else 'UNKNOWN',
            'native_identity_mismatch_count': len(identity_mismatches),
            'native_identity_mismatch_examples': identity_mismatches[:8],
            'actual_native_array_equivalence': 'NOT_TESTED'}


def audit_requested_source(source, schedule, snapshots, *, event_id, element):
    reports, cache = [], {}
    partition = source.get('partition_proof', {})
    if partition.get('all_thresholds_integer') is not True:
        raise ValueError('Source integer-predicate partition proof is absent.')
    for segment in source['segments']:
        a, b = fr(segment['t0']), fr(segment['t1'])
        if not a < b or any(a < i < b for i in range(a.numerator//a.denominator, b.numerator//b.denominator+1)):
            raise ValueError('Source cell crosses an unsplit integer predicate.')
    from run_window_audit import check_junctions
    check_junctions(source)
    for query in schedule['all_queries']:
        if not query['active']:
            reports.append({'query': query, 'status': 'BASELINE_OUTSIDE_WINDOW'})
            continue
        support, representative, proof = support_for_query(source, F(query['evaluation_tau']))
        if representative not in cache:
            snapshot = snapshots.get((event_id, representative))
            if snapshot is not None and (snapshot.event_id != event_id or F(snapshot.tau) != representative):
                raise ValueError('Snapshot event/time binding differs from requested representative.')
            cache[representative] = ({'status': 'UNKNOWN', 'reason_code': 'MISSING_REQUIRED_SNAPSHOT',
                'certified_necessary_policy_failure': False} if snapshot is None
                else audit_snapshot(snapshot, support, element=element))
        reports.append({'query': query, 'representative_tau': str(representative),
                        'time_predicate_basis': proof, **cache[representative]})
    failures = [row for row in reports if row['status'] == 'REJECT']
    unknown = [row for row in reports if row['status'] == 'UNKNOWN']
    return {'status': 'REJECT' if failures else 'UNKNOWN' if unknown else 'PASS_REQUESTED_SOURCE_INTERFACE',
            'queries': reports, 'distinct_interface_snapshots': len(cache),
            'decisive_rejection': failures[0] if failures else None,
            'not_claimed': 'No actual runtime, full exterior embedding, or same-root union admission.'}
