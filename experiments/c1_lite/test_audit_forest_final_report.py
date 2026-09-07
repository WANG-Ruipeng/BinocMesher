from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

import audit_forest_final_report as a


def schedule(root=F(3, 2)):
    delta = float.fromhex('0x1.500053e2d6239p-1')
    origin = float(1/48)
    first = 21 if root == F(3, 2) else 37
    hit = list(range(first, first+8))
    natural = []
    for frame in range(first-1, first+9):
        i = frame-1
        t = float((i+0.5)/24)
        local = float(t-origin)
        natural.append({'key': f'frame_{frame:04d}', 'kind': 'natural', 'frame_number': frame,
            'frame_index_zero_based': i, 'time_mode': 'physical', 'physical_time_hex': local.hex(),
            'global_camera_time_hex': t.hex(), 'evaluation_tau': str(F.from_float(float(local/delta))),
            'active': frame in hit})
    exact = {'key': 'root_'+str(root).replace('/', '_'), 'kind': 'exact_root', 'frame_number': None,
             'time_mode': 'exact', 'evaluation_tau': str(root),
             'physical_time_hex': float(root*F.from_float(delta)).hex(), 'active': True}
    return {'natural': natural, 'exact_root': exact, 'all_queries': natural+[exact],
            'bounds': {'lower': str(root-F(1, 4)), 'root': str(root), 'upper': str(root+F(1, 4))},
            'all_camera_frames': 64, 'root_excluded_from_natural_rates': True,
            'other_events_policy': 'UNCHANGED_SAME_ORDINARY_BASELINE',
            'delta_t_hex': delta.hex(), 'origin_seconds_hex': origin.hex(),
            'time_mapping_status': 'ACTUAL_NATIVE_INITIALIZATION_VERIFIED',
            'natural_hit_count': 8, 'selected_natural_count': 10, 'hit_frame_numbers': hit}


def local(boundary=None, center=(1, 1, 0)):
    boundary = boundary or [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]
    points = [tuple(map(F, p)) for p in boundary]
    turns = [a.cross2(a.sub(points[(i+1) % 4], points[i]),
                     a.sub(points[(i+2) % 4], points[(i+1) % 4])) for i in range(4)]
    orientation = 1 if turns[0] > 0 else -1
    inward = [orientation*a.cross2(a.sub(points[(i+1) % 4], points[i]), a.sub(tuple(map(F, center)), points[i])) for i in range(4)]
    return {'boundary': boundary, 'center': list(center), 'turns': list(map(str, turns)), 'inward': list(map(str, inward))}


def degeneracy():
    return {'status': 'REJECT', 'certified_necessary_policy_failure': True,
            'reason': a.DEGENERATE_REASON, 'stage': 'actual_interface', 'local_exact': local(),
            'boundary_actual_ids': [0, 1, 2, 3], 'replaced_face_rows': [0, 1],
            'failing_face': 2, 'failing_face_actual_ids': [0, 4, 5],
            'failing_face_binary32_coordinates': [[0, 0, 0], [0, -1, 0], [0, -2, 0]],
            'failing_face_exact_cross': ['0', '0', '0'], 'failing_face_shared_boundary_ids': [0]}


def query_fixture():
    s = schedule()
    key = a.token(s['exact_root'])
    meshes = [{'vertices': 8, 'faces': 10, 'sha256': {k: '1'*64 for k in ('vertices', 'faces', 'tags')}} for _ in range(5)]
    source = {'camera_inputs_sha256': '2'*64}
    query = {'identity_encoding': {'version': 2, 'encoding': 'ORIGINAL_EFFECTIVE_SOURCE_VID',
        'shifts_order': ['node0', 'group0', 'node1', 'group1'], 'per_element_shifts': [[0]*4 for _ in range(5)]},
        'baseline': meshes, 'query_token': list(key), 'query': s['exact_root'],
        'observer_enabled_disabled_byte_equal': True, 'all_events_preserved_shared_baseline': True,
        'frozen_library_baseline_parity': 'EXACT_RECEIPT_MATCH', 'events': [],
        'counts': [{'element': i, 'vertices': 8, 'faces': 10, 'identity_vertices': 8, 'raw_owners': 10} for i in range(5)]}
    reference = {'status': 'FROZEN_ORDINARY_QUERY_RECEIPT', 'library_sha256': a.FROZEN_SO_SHA,
        'query_token': list(key), 'meshes': deepcopy(meshes), 'registry_sha256': a.REGISTRY_SHA,
        'camera_inputs_sha256': '2'*64, 'effective_inputs_sha256': a.EFFECTIVE_SHA,
        'initialization': {'actual_delta_t_hex': s['delta_t_hex']}, 'identity_enabled': False,
        'baseline_only': True, 'extra_smooth': False, 'reader_closed': True}
    return query, reference, source


def pass_fixture():
    query, _, _ = query_fixture()
    owners = [[0, 1, 2, 3, 4, 5, 6], [0, 7, 8, 9, 10, 11, 12]]
    source = {'breakpoint_points': [{'time': '3/2', 'owners': owners}], 'segments': []}
    plan = {'element': 0, 'boundary_actual_ids': [0, 1, 2, 3], 'removed_face_rows': [0, 1],
        'new_center_id': 8, 'baseline_vertex_count': 8, 'baseline_face_count': 10, 'center': [1, 1, 0],
        'fan_faces': [[i, (i+1) % 4, 8] for i in range(4)], 'source_digest': a.digest(source), 'consumed_owners': owners}
    audit = {'status': 'PASS', 'full_exterior': True, 'local_exact': local(), 'tau': '3/2',
        'boundary_actual_ids': [0, 1, 2, 3], 'replaced_face_rows': [0, 1],
        'continuous_window_admitted': False, 'same_root_group_admitted': False,
        'expected_retained_faces': 48, 'retained_faces_checked': 48, 'certificate_counts': {'STRICT_ACTUAL_AABB': 48},
        'element': 0, 'source_digest': a.digest(source), 'consumed_owners': owners}
    arrays = {'status': 'ACTUAL_ARRAYS_CONSTRUCTED_NOT_PUBLISHED', 'element': 0,
        'old_vertices_and_tags_byte_identical': True, 'all_retained_face_rows_byte_identical': True,
        'other_elements_unchanged_by_object_identity': True,
        'vertices_before': 8, 'faces_before': 10, 'vertices_after': 9, 'faces_after': 12,
        'output': {'vertices': 9, 'faces': 12, 'sha256': {'vertices': '3'*64, 'faces': '4'*64, 'tags': '5'*64}}}
    return {'query': query['query'], 'audit': audit, 'plan': plan, 'actual_array_receipt': arrays}, query, source


def contact_fixture():
    query, _, _ = query_fixture()
    d = degeneracy()
    d.update(stage='actual_five_element_exterior', element=0, failing_element=1,
             reason_code='FORBIDDEN_ACTUAL_PATCH_RETAINED_CONTACT', exact_contact_fallback_enabled=True,
             failing_ids=[[1, 4], [1, 5], [1, 6]],
             failing_coordinates=[[1, 0.25, -1], [1, 0.25, 1], [1.5, 0.25, 0]])
    proof = {'status': 'REJECT_POLICY_CONTACT', 'new_contact_relative_to_baseline_proven': False,
             'shared_ids': [], 'witness': {'point_exact': ['1', '1/4', '0'],
             'barycentric_a_exact': ['3/8', '3/8', '1/4'], 'barycentric_b_exact': ['1/2', '1/2', '0']}}
    d['exact_contact_fallback'] = {'status': 'REJECT_POLICY_CONTACT', 'baseline_novelty_checked': False,
        'fan_triangle_index': 0, 'fan_triangle_coordinates': [[0, 0, 0], [2, 0, 0], [1, 1, 0]],
        'fan_triangle_ids': [[0, 0], [0, 1], [0, 8]],
        'retained_triangle_coordinates': d['failing_coordinates'], 'retained_triangle_ids': d['failing_ids'],
        'exact_contact': proof}
    refresh_proof(d)
    return d, query['baseline']


def refresh_proof(d):
    f = d['exact_contact_fallback']
    value = {'a': [[str(F(x)) for x in p] for p in f['fan_triangle_coordinates']],
             'b': [[str(F(x)) for x in p] for p in f['retained_triangle_coordinates']],
             'ids_a': f['fan_triangle_ids'], 'ids_b': f['retained_triangle_ids']}
    f['exact_contact'].update(input_exact=value, input_sha256=a.digest(value))


def committed_fixture():
    s = schedule()
    return {'schedule': s, 'runtime': {'cases': [{'query': q,
            'audit': {'status': 'PASS' if q['active'] else 'BASELINE_OUTSIDE_WINDOW'}} for q in s['all_queries']],
        'status': 'COMMITTED_REQUESTED_SCHEDULE', 'published_partial_results': False,
        'publication_kind': 'COMPACT_VERIFIED_ACTUAL_ARRAY_SCHEDULE_MANIFEST',
        'source_inputs_unchanged': True, 'whole_mesh_arrays_discarded_after_verification': True}}


class FinalAuditTests(unittest.TestCase):
    def test_01_both_fixed_schedules(self):
        for root in (F(3, 2), F(5, 2)):
            self.assertEqual(len(a.validate_schedule(schedule(root), root)), 8)

    def test_02_duplicate_query_rejected(self):
        s = schedule()
        s['all_queries'][-1] = s['all_queries'][0]
        with self.assertRaises(a.AuditError): a.validate_schedule(s, F(3, 2))

    def test_03_phase_and_activity_changes_rejected(self):
        for field, value in (('physical_time_hex', '0x0p+0'), ('active', True)):
            s = schedule(); s['natural'][0][field] = value
            with self.assertRaises(a.AuditError): a.validate_schedule(s, F(3, 2))

    def test_04_all_old_identity_encodings_rejected(self):
        query, _, _ = query_fixture()
        for field, value in (('version', 1), ('encoding', 'NORMALIZED'), ('per_element_shifts', [[0]*4])):
            receipt = deepcopy(query['identity_encoding']); receipt[field] = value
            with self.assertRaises(a.AuditError): a.validate_identity(receipt)

    def test_05_shift_shape_and_integer_range(self):
        query, _, _ = query_fixture()
        for value in (True, 2**31, -2**31-1, 1.5):
            receipt = deepcopy(query['identity_encoding']); receipt['per_element_shifts'][1][2] = value
            with self.assertRaises(a.AuditError): a.validate_identity(receipt)

    def test_06_frozen_query_match(self):
        a.validate_query_receipt(*query_fixture())

    def test_07_parity_label_does_not_replace_hash_match(self):
        q, r, s = query_fixture(); r['meshes'][3]['sha256']['faces'] = '0'*64
        with self.assertRaises(a.AuditError): a.validate_query_receipt(q, r, s)

    def test_08_missing_fifth_element_rejected(self):
        q, r, s = query_fixture(); q['baseline'].pop()
        with self.assertRaises(a.AuditError): a.validate_query_receipt(q, r, s)

    def test_09_incomplete_identity_count_rejected(self):
        q, r, s = query_fixture(); q['counts'][2]['identity_vertices'] -= 1
        with self.assertRaises(a.AuditError): a.validate_query_receipt(q, r, s)

    def test_10_real_degenerate_witness(self):
        self.assertEqual(a.validate_native_rejection(degeneracy()), 'PREEXISTING_RETAINED_INTERFACE_DEGENERACY_FIXED_POLICY')

    def test_11_fake_degeneracy_rejected(self):
        d = degeneracy(); d['failing_face_binary32_coordinates'][2][0] = 1
        with self.assertRaises(a.AuditError): a.validate_native_rejection(d)

    def test_12_removed_or_nonincident_face_not_witness(self):
        for field, value in (('failing_face', 0), ('failing_face_actual_ids', [6, 4, 5])):
            d = degeneracy(); d[field] = value
            with self.assertRaises(a.AuditError): a.validate_native_rejection(d)

    def test_13_same_identity_coordinate_conflict(self):
        d = degeneracy(); d['failing_face_binary32_coordinates'][0][1] = -3
        with self.assertRaises(a.AuditError): a.validate_native_rejection(d)

    def test_14_bare_policy_marker_is_not_a_witness(self):
        d = degeneracy(); d['reason'] = 'Boundary SourceVID missing or ambiguous in complete native ledger.'
        with self.assertRaisesRegex(a.AuditError, 'UNVERIFIED_REJECTION_WITNESS'): a.validate_native_rejection(d)

    def test_15_false_local_rejection(self):
        d = degeneracy(); d.update(stage='actual_local_graph', reason='Actual rounded center is not strictly inside source boundary.')
        with self.assertRaises(a.AuditError): a.validate_native_rejection(d)

    def test_16_exact_outside_center_rejection(self):
        d = degeneracy(); d.update(stage='actual_local_graph', reason='Actual rounded center is not strictly inside source boundary.', local_exact=local(center=(3, 1, 0)))
        self.assertEqual(a.validate_native_rejection(d), 'ACTUAL_CENTER_NOT_STRICTLY_INSIDE')

    def test_17_saved_turn_strings_not_trusted(self):
        d = degeneracy(); d['local_exact']['turns'][0] = '0'
        with self.assertRaises(a.AuditError): a.validate_native_rejection(d)

    def test_18_binary64_only_coordinate_refused(self):
        with self.assertRaises(a.AuditError): a.exact_point([0.1, 0, 0])

    def test_19_complete_array_receipt(self):
        a.validate_pass(*pass_fixture())

    def test_20_retained_denominator_and_outside_rows(self):
        c, q, s = pass_fixture(); c['audit']['retained_faces_checked'] -= 1
        with self.assertRaises(a.AuditError): a.validate_pass(c, q, s)
        c, q, s = pass_fixture(); c['actual_array_receipt']['all_retained_face_rows_byte_identical'] = False
        with self.assertRaises(a.AuditError): a.validate_pass(c, q, s)

    def test_21_partial_owner_consumption(self):
        c, q, s = pass_fixture(); c['plan']['consumed_owners'] = c['plan']['consumed_owners'][:1]
        with self.assertRaises(a.AuditError): a.validate_pass(c, q, s)

    def test_22_unknown_keeps_point_rate_null(self):
        events = [{'decision': d, 'schedule': schedule()} for d in ('ADMITTED_REQUESTED_SCHEDULE', 'REJECTED_FIXED_POLICY', 'UNKNOWN')]
        result = a.summarize(events)
        self.assertIsNone(result['policy_admission_rate'])
        self.assertEqual(result['policy_admission_rate_bounds'], [1/3, 2/3])
        self.assertEqual(result['actual_event_frame_modifications'], 8)
        self.assertEqual(result['actual_unique_modified_natural_frame_rate'], 8/64)

    def test_23_population_duplicate_or_mismatch(self):
        ids = ['a', 'b']; expected = a.digest(ids)
        summary = {'expected_event_ids_sha256': expected, 'canonical_event_denominator': 2, 'registry_sha256': a.REGISTRY_SHA}
        index = [{'event_id': i, 'decision': 'UNKNOWN'} for i in ids]
        events = {i: {'event_id': i, 'decision': 'UNKNOWN'} for i in ids}
        a.validate_population(summary, index, events, expected_count=2, expected_sha=expected)
        index[1] = index[0]
        with self.assertRaises(a.AuditError): a.validate_population(summary, index, events, expected_count=2, expected_sha=expected)

    def test_24_artifact_hash_path_and_read_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'a.json'; path.write_text('{}', encoding='utf-8')
            self.assertEqual(a.Reader().read(tmp, 'a.json', hashlib.sha256(b'{}').hexdigest()), {})
            for name, sha, budget in (('a.json', '0'*64, 100), ('../escape', None, 100), ('a.json', None, 1)):
                with self.assertRaises(a.AuditError): a.Reader(budget).read(tmp, name, sha)

    def test_25_scope_correction_rejected_before_other_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'scope_correction.json').write_text(json.dumps({'supersedes_admission_interpretation_only': True}), encoding='utf-8')
            result = a.audit_attempt(tmp, Path(tmp)/'missing-source')
            self.assertEqual(result['status'], 'INVALID_REPORT')
            self.assertIn('withdrawn', result['errors'][0])
            self.assertIsNone(result['policy_admission_rate'])

    def test_26_source_false_path_flag_alone_not_enough(self):
        e = {'compiler': {}, 'schedule': {}}
        e['decisive_rejection'] = {'certified_necessary_policy_failure': True,
            'gate': 'requested_source_interface', 'reason_code': 'SOURCE_INTERFACE_REJECT',
            'compiler_sha256': a.digest({}), 'schedule_sha256': a.digest({}),
            'native_identity_mismatch_count': 0, 'native_ordered_identity_model': 'PASS',
            'interface': {'witness_count': 1, 'witnesses': [{'kind': 'RETAINED_LINK_NOT_ONE_PATH',
                'link_edges': 5, 'link_vertices': 6, 'directed_link_path_compatible': False}]}}
        with self.assertRaises(a.AuditError): a.validate_source_rejection(e)
        e['decisive_rejection']['interface']['witnesses'][0]['link_edges'] = 6
        self.assertEqual(a.validate_source_rejection(e), 'SOURCE_QUOTIENT_LINK_COUNT_NECESSARY_FAILURE')

    def test_27_exact_contact_independent_witness_replay(self):
        d, baseline = contact_fixture()
        self.assertEqual(a.validate_native_rejection(d, baseline), 'EXACT_FORBIDDEN_FAN_RETAINED_CONTACT_BASELINE_NOVELTY_NOT_TESTED')

    def test_28_infeasible_or_stale_contact_witness(self):
        for change in ('point', 'fan', 'center_id'):
            d, baseline = contact_fixture()
            f = d['exact_contact_fallback']
            if change == 'point': f['exact_contact']['witness']['point_exact'][0] = '2'
            elif change == 'fan': f['fan_triangle_index'] = 1
            else: f['fan_triangle_ids'][2][1] = 7
            with self.assertRaises(a.AuditError): a.validate_native_rejection(d, baseline)

    def test_29_permitted_shared_point_not_forbidden_contact(self):
        d, baseline = contact_fixture()
        d.update(failing_element=0, failing_ids=[[0, 0], [0, 4], [0, 5]],
                 failing_coordinates=[[0, 0, 0], [-1, 0, 0], [0, -1, 0]])
        f = d['exact_contact_fallback']
        f.update(retained_triangle_coordinates=d['failing_coordinates'], retained_triangle_ids=d['failing_ids'])
        f['exact_contact'].update(shared_ids=[[0, 0]], witness={'point_exact': ['0', '0', '0'],
            'barycentric_a_exact': ['1', '0', '0'], 'barycentric_b_exact': ['1', '0', '0']})
        refresh_proof(d)
        with self.assertRaisesRegex(a.AuditError, 'permitted actual shared feature'):
            a.validate_native_rejection(d, baseline)

    def test_30_complete_commit_and_missing_late_query(self):
        event = committed_fixture()
        a.validate_committed_schedule(event)
        event['runtime']['cases'].pop(-2)
        with self.assertRaises(a.AuditError): a.validate_committed_schedule(event)

    def test_31_duplicate_query_and_partial_publish_rejected(self):
        for fault in ('duplicate', 'partial', 'later_unknown'):
            event = committed_fixture()
            if fault == 'duplicate': event['runtime']['cases'][-2] = event['runtime']['cases'][0]
            elif fault == 'partial': event['runtime']['published_partial_results'] = True
            else: event['runtime']['cases'][-2]['audit']['status'] = 'UNKNOWN'
            with self.assertRaises(a.AuditError): a.validate_committed_schedule(event)

    def test_32_arbitrary_certificate_name_cannot_admit(self):
        c, q, s = pass_fixture()
        c['audit']['certificate_counts'] = {'JUST_TRUST_PASS': 48}
        with self.assertRaises(a.AuditError): a.validate_pass(c, q, s)

    def test_33_path_plus_cycle_has_e_v_minus_one_but_is_disconnected(self):
        graph = a.graph_replay([('a', 'p'), ('p', 'b'), ('u', 'v'), ('v', 'u')], 'a', 'b')
        self.assertEqual(graph['link_edges'], graph['link_vertices']-1)
        self.assertTrue(graph['degree_and_endpoints_compatible'])
        self.assertTrue(graph['directed_link_path_compatible'])
        self.assertFalse(graph['connected'])
        self.assertEqual(graph['status'], 'REJECT')
        self.assertEqual(graph['components'], [['a', 'b', 'p'], ['u', 'v']])

    def test_34_graph_connected_wrong_direction_rejected(self):
        graph = a.graph_replay([('a', 'p'), ('b', 'p')], 'a', 'b')
        self.assertTrue(graph['connected'])
        self.assertTrue(graph['degree_and_endpoints_compatible'])
        self.assertFalse(graph['directed_link_path_compatible'])
        self.assertEqual(a.graph_replay([('a', 'p'), ('p', 'b')], 'a', 'b')['status'], 'PASS')

    def test_35_output_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/'saved.json'
            destination.write_text('keep', encoding='utf-8')
            argv = ['audit', '--attempt', tmp, '--source-attempt', tmp, '--output', str(destination)]
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                a.main()
            self.assertEqual(destination.read_text(), 'keep')

    def test_36_plan_cannot_swap_out_certified_geometry(self):
        for field, value in (('center', [1.5, 1, 0]), ('boundary_actual_ids', [1, 2, 3, 0]),
                             ('removed_face_rows', [2, 3])):
            c, q, s = pass_fixture()
            c['plan'][field] = value
            with self.assertRaises(a.AuditError): a.validate_pass(c, q, s)

    def test_37_empty_or_partial_output_hashes_cannot_pass(self):
        for hashes in ({}, {'faces': 'a'*64}):
            c, q, s = pass_fixture()
            c['actual_array_receipt']['output']['sha256'] = hashes
            with self.assertRaises(a.AuditError): a.validate_pass(c, q, s)

    def test_38_stale_time_geometry_cannot_admit(self):
        c, q, s = pass_fixture()
        c['audit']['tau'] = '5/2'
        with self.assertRaises(a.AuditError): a.validate_pass(c, q, s)


if __name__ == '__main__':
    unittest.main()
