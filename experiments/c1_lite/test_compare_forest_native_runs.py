"""Synthetic report-only parity regressions; never runs native geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import compare_forest_native_runs as m


def fixture(omp='1'):
    root = {'time_mode': 'exact', 'evaluation_tau': '3/2', 'key': 'root', 'active': True}
    natural = {'time_mode': 'physical', 'physical_time_hex': (0.75).hex(), 'key': 'frame', 'active': True}
    outside = {'time_mode': 'physical', 'physical_time_hex': (1.0).hex(), 'key': 'outside', 'active': False}
    baseline = [{'vertices': 8, 'faces': 10, 'sha256': {k: str(i)*64 for k in ('vertices', 'faces', 'tags')}}
                for i in range(5)]
    receipt = {'status': 'ACTUAL_ARRAYS_CONSTRUCTED_NOT_PUBLISHED',
               'old_vertices_and_tags_byte_identical': True, 'all_retained_face_rows_byte_identical': True,
               'other_elements_unchanged_by_object_identity': True,
               'output': {'vertices': 9, 'faces': 12, 'sha256': {k: 'e'*64 for k in ('vertices', 'faces', 'tags')}}}
    cases = [{'query': q, 'audit': {'status': 'PASS', 'wall_seconds': 1.0},
              'plan': {'element': 1, 'center': [1, 1, 0], 'consumed_owners': [[1, 0, 0, 1, 0, 0, 0]]},
              'actual_array_receipt': deepcopy(receipt), 'check_wall_seconds': 2.0} for q in (root, natural)]
    cases.append({'query': outside, 'audit': {'status': 'BASELINE_OUTSIDE_WINDOW'},
                  'baseline': baseline, 'output_equals_baseline': True})
    admitted = {'event_id': 'event-a', 'root': '3/2', 'decision': 'ADMITTED_REQUESTED_SCHEDULE',
                'schedule': {'all_queries': [root, natural, outside], 'natural': [natural, outside],
                             'origin_seconds': 0.0},
                'runtime': {'status': 'COMMITTED_REQUESTED_SCHEDULE', 'cases': cases,
                            'published_partial_results': False, 'source_inputs_unchanged': True}}
    rejected_case = {'query': root, 'audit': {'status': 'REJECT', 'certified_necessary_policy_failure': True,
                     'witness': {'point_exact': ['1/3', '0', '0']}, 'wall_seconds': 0.2}}
    rejected = {'event_id': 'event-b', 'root': '3/2', 'decision': 'REJECTED_FIXED_POLICY',
                'decisive_rejection': {'gate': 'actual_requested_query', 'certified_necessary_policy_failure': True,
                                      'witness': rejected_case},
                'runtime': {'status': 'BASELINE_ENTIRE_REQUESTED_SCHEDULE', 'cases': [rejected_case]}}
    events = {'event-a': admitted, 'event-b': rejected}
    queries = {}
    for q in (root, natural, outside):
        key = m.token(q)
        rows = []
        for eid, event in events.items():
            for case in event['runtime']['cases']:
                if m.token(case['query']) == key:
                    rows.append({'event_id': eid, 'status': case['audit']['status'], 'case': case})
        queries[key] = {'query': q, 'query_token': list(key), 'baseline': baseline,
                        'observer_enabled_disabled_byte_equal': True, 'all_events_preserved_shared_baseline': True,
                        'events': rows, 'identity_encoding': {'version': 2}, 'native_cost': {'wall_seconds': 1}}
    verification = {'status': 'PASS', 'private_nonlog_inputs_unchanged': True,
                    'original_input_content_sha256': 'cache-hash'}
    summary = {'status': 'COMPLETE_REAL_FIXED_POLICY_RATE', 'all_events_resolved': True,
               'unknown': 0, 'admitted': 1, 'rejected_fixed_policy': 1, 'omp_num_threads': omp,
               'registry_sha256': m.REGISTRY_SHA, 'canonical_event_denominator': 2,
               'expected_event_ids_sha256': m.sha(sorted(events)), 'unique_native_queries': 3,
               'private_cache_removed': True, 'input_verification': verification,
               'final_input_verification': verification, 'executed_sources_sha256': {'fixed.py': 'code'},
               'source_campaign_sha256': 'source', 'native_library_sha256': 'library',
               'actual_delta_t_hex': (0.5).hex(), 'exact_contact_fallback_enabled': True,
               'cost': {'wall_seconds': 2, 'cpu_seconds': 3, 'peak_rss_bytes': 100}}
    return {'summary': summary, 'events': events, 'queries': queries, 'input_receipts': {}}


def compare(a, b):
    return m.compare_loaded(a, b, expected_events=2, expected_queries=3, expected_schedule_queries=3)


def write_attempt(root, run):
    root.mkdir()
    (root/'events').mkdir()
    (root/'queries').mkdir()
    (root/'summary.json').write_text(json.dumps(run['summary']))
    index = []
    for i, (eid, event) in enumerate(run['events'].items()):
        data = json.dumps(event).encode()
        path = 'events/'+str(i)+'.json'
        (root/path).write_bytes(data)
        index.append({'event_id': eid, 'decision': event['decision'],
                      'artifact': {'path': path, 'sha256': hashlib.sha256(data).hexdigest()}})
    (root/'events_index.json').write_text(json.dumps(list(reversed(index))))
    for i, query in enumerate(reversed(list(run['queries'].values()))):
        (root/'queries'/('q'+str(i)+'.json')).write_text(json.dumps(query))


class ForestComparisonTests(unittest.TestCase):
    def test_cost_and_processing_permutation_do_not_change_semantics(self):
        a, b = fixture(), fixture('8')
        b['summary']['cost']['wall_seconds'] = 99
        b['events'] = dict(reversed(list(b['events'].items())))
        b['queries'] = dict(reversed(list(b['queries'].items())))
        for e in b['events'].values():
            e['runtime']['cases'].reverse()
            for c in e['runtime']['cases']:
                c['audit']['wall_seconds'] = 77
                c['check_wall_seconds'] = 99
        b['events']['event-a']['schedule']['all_queries'].reverse()
        b['events']['event-a']['schedule']['natural'].reverse()
        for q in b['queries'].values():
            q['events'].reverse()
        result = compare(a, b)
        self.assertTrue(result['consistent_complete_admission_certified'], result)
        self.assertEqual(result['event_pairs_compared'], 2)
        self.assertEqual(result['query_pairs_compared'], 3)

    def test_plan_difference_is_reported(self):
        a, b = fixture(), fixture('8')
        b['events']['event-a']['runtime']['cases'][0]['plan']['center'][2] = 1
        result = compare(a, b)
        self.assertFalse(result['consistent_complete_admission_certified'])
        self.assertTrue(any(r['kind'] == 'admitted_plan' for r in result['semantic_differences']))

    def test_actual_output_hash_difference_is_reported(self):
        a, b = fixture(), fixture('8')
        b['events']['event-a']['runtime']['cases'][0]['actual_array_receipt']['output']['sha256']['faces'] = 'f'*64
        result = compare(a, b)
        self.assertTrue(any(r['kind'] == 'admitted_actual_output' for r in result['semantic_differences']))

    def test_failure_witness_is_not_ignored(self):
        a, b = fixture(), fixture('8')
        b['events']['event-b']['runtime']['cases'][0]['audit']['witness']['point_exact'][0] = '2/3'
        result = compare(a, b)
        self.assertFalse(result['consistent_complete_admission_certified'])
        self.assertTrue(result['semantic_differences'])

    def test_baseline_hash_difference_is_reported(self):
        a, b = fixture(), fixture('8')
        next(iter(b['queries'].values()))['baseline'][4]['sha256']['vertices'] = '9'*64
        result = compare(a, b)
        self.assertTrue(any(r['kind'] == 'five_element_baseline_hashes' for r in result['semantic_differences']))

    def test_changed_code_or_registry_prevents_parity(self):
        for field, value in (('executed_sources_sha256', {'fixed.py': 'changed'}), ('registry_sha256', 'changed')):
            a, b = fixture(), fixture('8')
            b['summary'][field] = value
            result = compare(a, b)
            self.assertFalse(result['consistent_complete_admission_certified'])

    def test_equal_unknown_runs_are_not_admission_parity(self):
        a, b = fixture(), fixture('8')
        for r in (a, b):
            r['summary'].update(status='UNRESOLVED_ACTUAL_QUERIES', all_events_resolved=False, unknown=1)
        result = compare(a, b)
        self.assertFalse(result['consistent_complete_admission_certified'])
        self.assertEqual(result['status'], 'NOT_CERTIFIED_INCOMPLETE_OR_UNRESOLVED')

    def test_missing_admitted_query_is_not_parity_even_in_both(self):
        a, b = fixture(), fixture('8')
        for r in (a, b):
            r['events']['event-a']['runtime']['cases'].pop()
        self.assertFalse(compare(a, b)['consistent_complete_admission_certified'])

    def test_missing_actual_receipt_not_accepted(self):
        a, b = fixture(), fixture('8')
        for r in (a, b):
            r['events']['event-a']['runtime']['cases'][0].pop('actual_array_receipt')
        self.assertFalse(compare(a, b)['consistent_complete_admission_certified'])

    def test_omp_pair_metadata_is_required(self):
        self.assertFalse(compare(fixture('1'), fixture('1'))['consistent_complete_admission_certified'])

    def test_physical_seconds_are_semantic_not_cost(self):
        self.assertEqual(m.semantic({'origin_seconds': 0.5, 'duration_seconds': 2, 'wall_seconds': 10}),
                         {'origin_seconds': 0.5, 'duration_seconds': 2})
        a, b = fixture(), fixture('8')
        b['events']['event-a']['schedule']['origin_seconds'] = 0.1
        self.assertFalse(compare(a, b)['consistent_complete_admission_certified'])

    def test_artifact_loader_validates_hash_and_order_independently(self):
        with tempfile.TemporaryDirectory(prefix='forest-compare-unit-') as folder:
            left, right = Path(folder)/'left', Path(folder)/'right'
            write_attempt(left, fixture())
            write_attempt(right, fixture('8'))
            self.assertTrue(compare(m.load_attempt(left), m.load_attempt(right))['consistent_complete_admission_certified'])
            (right/'events'/'0.json').write_text('{}')
            result = m.compare_runs(left, right)
            self.assertFalse(result['consistent_complete_admission_certified'])
            self.assertIn('SHA256 mismatch', result['reason'])

    def test_missing_attempt_is_not_certified(self):
        with tempfile.TemporaryDirectory(prefix='forest-compare-missing-') as folder:
            result = m.compare_runs(Path(folder)/'missing', Path(folder)/'also-missing')
            self.assertFalse(result['consistent_complete_admission_certified'])


if __name__ == '__main__':
    unittest.main()
