"""Tiny in-memory runner contracts; no genuine caches, native or rendering."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_screen as m


class MemoryReports:
    latest = None
    def __init__(self, root=Path('/tmp/not-created-screen-test')):
        self.root = Path(root); self.saved = {}; self.bytes = 0
        self.visibility_frame_artifacts = []
        MemoryReports.latest = self
    def save(self, name, value):
        self.saved[name] = deepcopy(value)
        return {'path': name, 'sha256': 'a'*64}


def empty_scene(n=2):
    meshes = []
    for _ in range(n):
        arrays = (np.empty((0, 3), np.float64), np.empty((0, 3), np.int32), np.empty(0, np.int32))
        for a in arrays: a.flags.writeable = False
        meshes.append(arrays)
    return tuple(meshes)


def query(key='root_1', tau='1', active=True, kind='exact_root'):
    return {'key': key, 'kind': kind, 'time_mode': 'exact' if kind == 'exact_root' else 'physical',
            'physical_time_hex': float(F(tau)).hex(), 'evaluation_tau': tau, 'active': active,
            'frame_index_zero_based': 0, 'absolute_frame_number': 97}


def mini_events():
    qs = [query('f0', '1/4', False, 'natural'), query('f1', '3/4', True, 'natural'),
          query('f2', '5/4', True, 'natural'), query('f3', '7/4', False, 'natural'), query()]
    return [{'event_id': eid, 'root': '1', 'element': 0, 'source_status': 'SOURCE_READY', 'decision': 'UNKNOWN',
             'schedule': {'all_queries': deepcopy(qs), 'natural': deepcopy(qs[:-1]),
                          'bounds': {'lower': '1/2', 'root': '1', 'upper': '3/2'}}} for eid in ('a', 'b', 'c')]


def footprints(events):
    return [{'event_id': e['event_id'], 'element': 0, 'status': 'COMPLETE_ACTUAL_REQUESTED_SUPPORT',
             'boundary_actual_ids': list(range(i*4, i*4+4)), 'source_face_rows': [i*2, i*2+1],
             'consumed_owners': [[0, 0, 0, i, 0, 0, 0]], 'retained_star_face_rows': [],
             'bounds': [[i*3, 0, 0], [i*3+1, 1, 0]]} for i, e in enumerate(events)]


class RunnerTests(unittest.TestCase):
    def test_slice_bound_actual_time_and_baseline_hash(self):
        q = query(); meshes = empty_scene()
        s = {'meshes': meshes, 'query': {'mode': 'exact', 'exact_time': {'numerator': 1, 'denominator': 1},
                                       'physical_time_hex': q['physical_time_hex']}}
        reader = SimpleNamespace(slice_query=lambda *a, **k: s)
        out, baseline = m.slice_bound(reader, q, ledger=False)
        self.assertIs(out, s)
        with self.assertRaisesRegex(ValueError, 'BASELINE_PARITY'):
            m.slice_bound(reader, q, [{'wrong': True}], ledger=False)
        s['query']['physical_time_hex'] = float(2).hex()
        with self.assertRaisesRegex(ValueError, 'query time differs'):
            m.slice_bound(reader, q, baseline, ledger=False)

    def test_identity_is_required_only_when_requested(self):
        q = query(); s = {'meshes': empty_scene(), 'identity_status': 0,
            'query': {'mode': 'exact', 'exact_time': {'numerator': 1, 'denominator': 1}, 'physical_time_hex': q['physical_time_hex']}}
        reader = SimpleNamespace(slice_query=lambda *a, **k: s)
        m.slice_bound(reader, q, ledger=False)
        with self.assertRaisesRegex(ValueError, 'not ready'): m.slice_bound(reader, q, ledger=True)

    def test_pair_matrix_is_complete_and_canonical(self):
        fs = footprints(mini_events()); by_id = {r['event_id']: r for r in fs}
        records = [{'event_id': eid, 'plan': {}} for eid in ('c', 'a', 'b')]
        with patch.object(m, 'check_pair_interactions', side_effect=AssertionError('strict-separated pairs need no exact call')):
            result = m.pair_proofs(None, records, by_id)
        self.assertEqual([r['events'] for r in result], [['a', 'b'], ['a', 'c'], ['b', 'c']])
        self.assertEqual(sum(r['report']['triangle_pairs_excluded'] for r in result), 96)

    def test_shared_identity_never_uses_disjoint_axis_shortcut(self):
        fs = footprints(mini_events()[:2]); fs[1]['consumed_owners'] = fs[0]['consumed_owners']
        with patch.object(m, 'check_pair_interactions', return_value={'status': 'UNSUPPORTED_SHARED_SUPPORT'}) as exact:
            result = m.pair_proofs(None, [{'event_id': 'a'}, {'event_id': 'b'}], {r['event_id']: r for r in fs})
        exact.assert_called_once(); self.assertEqual(result[0]['report']['status'], 'UNSUPPORTED_SHARED_SUPPORT')

    def test_touching_aabb_is_not_strict_separation(self):
        fs = footprints(mini_events()[:2]); fs[1]['bounds'][0][0] = fs[0]['bounds'][1][0]
        with patch.object(m, 'check_pair_interactions', return_value={'status': 'PASS_PAIR_INTERACTIONS'}) as exact:
            m.pair_proofs(None, [{'event_id': 'a'}, {'event_id': 'b'}], {r['event_id']: r for r in fs})
        exact.assert_called_once()

    def test_unexpected_pair_exception_is_attempt_stop(self):
        fs = footprints(mini_events()[:2]); fs[1]['bounds'] = fs[0]['bounds']
        with patch.object(m, 'check_pair_interactions', return_value={'status': 'UNKNOWN_INPUT_OR_PROOF'}):
            with self.assertRaisesRegex(ValueError, 'UNEXPECTED_PAIR'):
                m.pair_proofs(None, [{'event_id': 'a'}, {'event_id': 'b'}], {r['event_id']: r for r in fs})

    def test_no_selector_and_rejected_events_remain_requests(self):
        events = mini_events(); events[1]['decision'] = 'REJECTED_FIXED_POLICY'; events[2]['root'] = '2'
        with patch.object(m, 'candidate_halo_owners', return_value=[[0, 0, 0, 8, 0, 0, 0]]) as halo:
            requests = m.requests_for(events, {'b': {'complete': True}}, {'a': 'spec-a'}, '1', F(1))
        self.assertEqual([r['event_id'] for r in requests], ['a', 'b'])
        self.assertIsNone(requests[1]['spec']); self.assertTrue(requests[1]['candidate_actual_owners']); halo.assert_called_once()

    def certification_context(self, events, audit):
        fs = footprints(events); baseline = m.mesh_receipt(empty_scene())
        refs = {q['key']: baseline for q in events[0]['schedule']['all_queries']}
        s = {'meshes': empty_scene(), 'cost': {}}
        return refs, [patch.object(m, 'slice_bound', return_value=(s, baseline)),
            patch.object(m, 'actual_footprints', return_value=fs), patch.object(m, 'audit_query', side_effect=audit),
            patch.object(m, 'original_identity_receipt', return_value={'version': 2}), patch.object(m, 'message')]

    def test_late_event_reject_filters_pairs_not_graph_nodes(self):
        events = mini_events()
        def audit(s, spec, tau, **kwargs):
            if spec == 'a' and tau == F(5, 4):
                return None, {'status': 'REJECT', 'certified_necessary_policy_failure': True, 'reason': 'synthetic local gate'}
            return {'event': spec, 'tau': str(tau)}, {'status': 'PASS'}
        refs, patches = self.certification_context(events, audit)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            rows, calls = m.certify(None, events, {}, {e['event_id']: e['event_id'] for e in events}, refs,
                MemoryReports(), {'segment_id': 'tiny'}, lambda: None)
        self.assertEqual(calls, 5); self.assertEqual(len(rows), 3)
        self.assertEqual(events[0]['decision'], 'REJECTED_FIXED_POLICY')
        self.assertEqual([e['decision'] for e in events[1:]], ['ADMITTED_REQUESTED_SCHEDULE']*2)
        for row in rows:
            self.assertEqual([e['event_id'] for e in row['events']], ['a', 'b', 'c'])
            self.assertEqual([p['events'] for p in row['pair_proofs']], [['b', 'c']])
        graph = m.make_graph(events, rows, {'symbolic_edges': [{'events': ['a', 'b'], 'reason': 'REJECTED_BRIDGE'}]})
        admitted = [c['events'] for c in graph['components'] if c['decision'] == 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE']
        self.assertEqual(admitted, [['c']]); self.assertEqual(graph['event_count'], 3)

    def test_internal_unknown_does_not_become_silent_method_fallback(self):
        events = mini_events()
        refs, patches = self.certification_context(events, lambda *a, **k: (None,
            {'status': 'UNKNOWN', 'exception_type': 'RuntimeError', 'reason': 'synthetic adapter bug'}))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.assertRaises(ValueError):
                m.certify(None, events, {}, {e['event_id']: e['event_id'] for e in events}, refs,
                    MemoryReports(), {'segment_id': 'tiny'}, lambda: None)

    def test_replay_support_mismatch_stops_before_any_union(self):
        q = query('f', '1', True, 'natural'); reader = SimpleNamespace(delta_t=1.)
        baseline = m.mesh_receipt(empty_scene()); s = {'meshes': empty_scene()}
        with patch.object(m, 'all_queries', return_value=[q]), patch.object(m, 'slice_bound', return_value=(s, baseline)), \
             patch.object(m, 'actual_footprints', return_value=[{'changed': True}]), patch.object(m, 'compile_union') as union:
            with self.assertRaisesRegex(ValueError, 'SUPPORT_REPLAY_CHANGED'):
                m.execute_sequence(reader, {}, {}, [], {}, {}, {'components': []},
                    [{'query': q, 'root': '1', 'events': []}], {'f': baseline}, MemoryReports(), lambda: None)
        union.assert_not_called()

    def test_omp_output_difference_stops_before_frame_publication(self):
        q = query('f', '0', False, 'natural'); reader = SimpleNamespace(delta_t=1.)
        baseline = m.mesh_receipt(empty_scene()); s = {'meshes': empty_scene()}; reports = MemoryReports()
        expected = {'query': q, 'baseline': baseline, 'records': [], 'union': {'wrong': True},
                    'root_support_sha256': None, 'full_root_population_replayed': 0}
        with patch.object(m, 'all_queries', return_value=[q]), patch.object(m, 'slice_bound', return_value=(s, baseline)):
            with self.assertRaisesRegex(ValueError, 'OMP_SEQUENCE_DIFFERENCE'):
                m.execute_sequence(reader, {}, {}, [], {}, {}, {'components': []}, [], {'f': baseline},
                    reports, lambda: None, primary={'f': expected})
        self.assertFalse(reports.saved)

    def test_checkpoint_stops_before_next_native_query(self):
        def expired(): raise ValueError('SCREEN_WALL_BUDGET')
        with patch.object(m, 'all_queries', return_value=[query()]), patch.object(m, 'slice_bound') as native:
            with self.assertRaisesRegex(ValueError, 'WALL_BUDGET'):
                m.execute_sequence(SimpleNamespace(delta_t=1.), {}, {}, [], {}, {}, {'components': []}, [], {},
                    MemoryReports(), expired)
        native.assert_not_called()

    def test_zero_registry_keeps_explicit_empty_graph(self):
        graph = m.make_graph([], [], {'symbolic_edges': []})
        self.assertEqual(graph['event_count'], 0); self.assertEqual(graph['components'], [])
        self.assertEqual(graph['status'], 'COMPLETE_ZERO_REGISTRY_SUPPORT_GRAPH')

    def test_wrong_omp_replay_stops_before_native_initialization(self):
        args = SimpleNamespace(output=Path('/tmp/not-created-screen-test'), segment='tiny', replay_from=Path('/tmp/no-primary'),
            protocol=Path('/tmp/no-protocol'), seal=Path('/tmp/no-seal'))
        values = ({'budgets': {'certification_and_screen_per_scene_wall_seconds': 999,
                             'native_query_peak_rss_bytes': 2**40}}, {'segment_id': 'tiny'}, {}, {}, {},
                  Path('/tmp/no-cache'), {}, [], {'events': []}, {})
        with patch.object(m, 'Reports', MemoryReports), patch.object(m, 'validate_inputs', return_value=values), \
             patch.object(m, 'executed_sources', return_value={}), patch.object(m, 'initialize') as native, \
             patch.object(m, 'PrivateSceneCache') as private, patch.object(m, 'message'), \
             patch.object(m, 'file_sha', return_value='b'*64), \
             patch.dict(m.os.environ, {'OMP_NUM_THREADS': '1'}):
            code = m.run(args)
        self.assertEqual(code, 2); native.assert_not_called(); private.assert_not_called()
        result = MemoryReports.latest.saved['summary.json']
        self.assertEqual(result['status'], 'STOP_SCREEN_ATTEMPT_NOT_COMPLETE')
        self.assertFalse(result['partial_output_published'])


class VisibilityReceiptTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/tmp/not-created-visibility-receipts')
        self.segment = {'segment_id': 'tiny', 'first_frame': 97}
        self.rule = {'minimum_natural_frames': 2, 'minimum_baseline_support_pixels_per_frame': 16,
                     'minimum_actual_replacement_pixels_on_each_qualifying_frame': 1}
        qs = []
        for i in range(2):
            q = query('f'+str(i), str(i+1), True, 'natural')
            q.update(frame_index_zero_based=i, absolute_frame_number=97+i)
            qs.append(q)
        self.events = [{'event_id': 'a', 'schedule': {'natural': qs}}]
        self.components = [{'component_id': 'c', 'events': ['a'],
                            'decision': 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE'}]
        self.docs, self.image_hashes, self.refs = {}, {}, []
        for q in qs:
            image_path = 'images/'+q['key']+'.png'
            self.image_hashes[str((self.root/image_path).resolve())] = 'c'*64
            row = {'query': deepcopy(q), 'events': [{'event_id': 'a', 'component_id': 'c',
                    'jointly_admitted': True, 'domain_kind': 'FIXED_SOURCE_AND_REPLACEMENT',
                    'visible_baseline_source_pixels': 20, 'visible_replacement_pixels': 10}],
                   'components': [{'component_id': 'c', 'decision': 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE',
                    'visible_baseline_union_pixels': 20, 'visible_replacement_union_pixels': 10}],
                   'images': [{'kind': 'depth_pair', 'path': image_path, 'sha256': 'c'*64}]}
            path = 'visibility/'+q['key']+'.json'
            self.docs[str((self.root/path).resolve())] = row
            self.refs.append({'query': deepcopy(q), 'artifact': {'path': path, 'sha256': m.digest(row)}})

    def hashes(self, path):
        key = str(Path(path).resolve())
        return m.digest(self.docs[key]) if key in self.docs else self.image_hashes[key]

    def read_receipts(self, refs=None, events=None):
        bindings = {}
        with patch.object(m, 'file_sha', side_effect=self.hashes), \
             patch.object(m, 'read', side_effect=lambda p: deepcopy(self.docs[str(Path(p).resolve())])):
            rows, summary = m.read_visibility_receipts(self.root, self.refs if refs is None else refs,
                self.events if events is None else events, self.components, self.segment, self.rule, bindings)
        return rows, summary, bindings

    def test_complete_rows_recompute_first_qualifying_and_bind_images(self):
        rows, summary, bindings = self.read_receipts()
        self.assertEqual(summary['natural_event_frame_queries'], 2)
        self.assertEqual(summary['first_qualifying_component']['qualifying_absolute_frames'], [97, 98])
        self.assertEqual(set(bindings), set(self.docs) | set(self.image_hashes))
        self.assertEqual(len(rows), 2)

    def test_omitted_frame_receipt_stops_even_if_remaining_summary_looks_valid(self):
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE_VISIBILITY_FRAME_RECEIPTS'):
            self.read_receipts(self.refs[:1])

    def test_duplicate_frame_receipt_stops(self):
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_OR_UNEXPECTED_VISIBILITY_RECEIPT'):
            self.read_receipts([self.refs[0], self.refs[0], self.refs[1]])

    def test_tampered_frame_bytes_stop_at_hash(self):
        self.docs[next(iter(self.docs))]['events'][0]['visible_baseline_source_pixels'] = 999
        with self.assertRaisesRegex(ValueError, 'BOUND_ARTIFACT_CHANGED'):
            self.read_receipts()

    def test_tampered_image_bytes_stop_at_hash(self):
        self.image_hashes[next(iter(self.image_hashes))] = 'd'*64
        with self.assertRaisesRegex(ValueError, 'VISIBILITY_IMAGE_CHANGED'):
            self.read_receipts()

    def test_ref_query_must_equal_bound_row_query(self):
        self.refs[0]['query']['absolute_frame_number'] = 999
        with self.assertRaisesRegex(ValueError, 'VISIBILITY_RECEIPT_QUERY_CHANGED'):
            self.read_receipts()

    def test_rehashed_row_still_must_match_original_schedule(self):
        key = next(iter(self.docs)); row = self.docs[key]
        row['query']['physical_time_hex'] = float(99).hex()
        self.refs[0]['query'] = deepcopy(row['query'])
        self.refs[0]['artifact']['sha256'] = m.digest(row)
        with self.assertRaisesRegex(ValueError, 'VISIBILITY_RECEIPT_SCHEDULE_CHANGED'):
            self.read_receipts()

    def test_zero_event_receipts_explicitly_empty(self):
        _, summary, bindings = self.read_receipts([], [])
        self.assertEqual(summary['canonical_events'], 0)
        self.assertIsNone(summary['visible_source_admission_rate'])
        self.assertFalse(bindings)

    def test_missing_receipt_list_is_not_empty_success(self):
        with self.assertRaisesRegex(ValueError, 'VISIBILITY_FRAME_RECEIPTS_REQUIRED'):
            m.read_visibility_receipts(self.root, None, [], [], self.segment, self.rule, {})

    def test_input_stop_keeps_supplied_protocol_seal_hashes_without_validation_claim(self):
        args = SimpleNamespace(output=self.root, segment='tiny', replay_from=None,
                               protocol=Path('/tmp/protocol'), seal=Path('/tmp/seal'))
        with patch.object(m, 'Reports', MemoryReports), patch.object(m, 'file_sha', return_value='b'*64), \
             patch.object(m, 'validate_inputs', side_effect=ValueError('INVALID_SEAL')), \
             patch.object(m, 'initialize') as native, patch.object(m, 'message'):
            status = m.run(args)
        result = MemoryReports.latest.saved['summary.json']
        self.assertEqual(status, 2); native.assert_not_called()
        self.assertEqual(result['protocol_sha256'], 'b'*64)
        self.assertEqual(result['preregistration_seal_sha256'], 'b'*64)
        self.assertTrue(result['supplied_input_hashes_do_not_imply_validation'])
        self.assertFalse(result['partial_output_published'])


if __name__ == '__main__':
    unittest.main()
