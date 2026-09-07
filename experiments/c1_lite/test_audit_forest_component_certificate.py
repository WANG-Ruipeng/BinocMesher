import copy
from fractions import Fraction as F
from itertools import combinations
import unittest

from audit_forest_component_certificate import (merge_bindings, recheck_pair,
    validate_footprint, validate_payload, validate_schedule)
from forest_component_graph import build_graph
from run_forest_component_certification import decide_components


def fixture():
    events, queries = [], []
    baseline = [{'vertices': 12, 'faces': 6, 'sha256': {'vertices': 'v', 'faces': 'f', 'tags': 't'}} for _ in range(5)]
    for rindex, root in enumerate(('3/2', '5/2')):
        root_time = F(root)
        times = [root_time+F(offset, 100) for offset in (-20, -15, -10, -5, 5, 10, 15, 20)]
        schedule_queries = [{'key': f'f{rindex}_{i}', 'kind': 'natural', 'active': True,
            'time_mode': 'physical', 'physical_time_hex': float(t).hex(), 'evaluation_tau': str(t)} for i, t in enumerate(times)]
        schedule_queries.append({'key': 'root_'+root, 'kind': 'exact_root', 'active': True,
            'time_mode': 'exact', 'physical_time_hex': float(root_time).hex(), 'evaluation_tau': root})
        levels = {'lower': str(root_time-F(1, 4)), 'root': root, 'upper': str(root_time+F(1, 4))}
        local_events, footprints, plans = [], [], {}
        for ordinal in range(3):
            eid = f'event{rindex}_{ordinal}'
            base, x = 4*ordinal, 3*ordinal
            cycle = list(range(base, base+4))
            owners = [[0, 0, 0, ordinal, 0, 0, j] for j in range(2)]
            row = {'event_id': eid, 'element': 0, 'status': 'COMPLETE_ACTUAL_REQUESTED_SUPPORT',
                'kind': 'FIXED_SOURCE_AND_REPLACEMENT', 'full_interface_star_enumerated': True,
                'boundary_actual_ids': cycle, 'boundary_coordinates': [[x, 0, 0], [x+1, 0, 0], [x+1, 1, 0], [x, 1, 0]],
                'center': [x+0.5, 0.5, 0], 'bounds': [[x, 0, 0], [x+1, 1, 0]],
                'source_face_rows': [2*ordinal, 2*ordinal+1], 'retained_star_face_rows': [],
                'source_triangles': [[base, base+1, base+2], [base, base+2, base+3]], 'consumed_owners': owners}
            plan = {'element': 0, 'baseline_vertex_count': 12, 'baseline_face_count': 6, 'new_center_id': 12,
                'boundary_actual_ids': cycle, 'center': row['center'], 'removed_face_rows': row['source_face_rows'],
                'consumed_owners': owners, 'fan_faces': [[cycle[i], cycle[(i+1) % 4], 12] for i in range(4)]}
            event = {'event_id': eid, 'root': root, 'element': 0,
                'decision': 'ADMITTED_REQUESTED_SCHEDULE' if ordinal < 2 else 'REJECTED_FIXED_POLICY',
                'schedule': {'bounds': levels, 'all_queries': schedule_queries}, 'compiler': {'source': {'levels': levels}},
                'runtime': {'cases': [{'query': q, 'audit': {'status': 'PASS'}, 'plan': plan} for q in schedule_queries]}}
            events.append(event); local_events.append(event); footprints.append(row)
            if ordinal < 2:
                plans[eid] = {'event_id': eid, 'component_id': 'provisional'+eid, 'plan': plan}
        for query in schedule_queries:
            pairs = [{'events': list(pair), 'report': {'status': 'PASS_PAIR_INTERACTIONS',
                      'proof': 'STRICT_ACTUAL_BINARY32_SUPPORT_AABB', 'triangle_pairs_excluded': 32}}
                     for pair in combinations(sorted(plans), 2)]
            queries.append({'root': root, 'query': query, 'baseline': baseline,
                'identity_encoding': {'version': 2, 'encoding': 'ORIGINAL_EFFECTIVE_SOURCE_VID'},
                'events': footprints, 'certified_independent_plans': list(plans.values()), 'pair_proofs': pairs})
    graph = build_graph(events, queries)
    summary = {'query_count': len(queries), **decide_components(graph, queries)}
    frozen = {q['query']['key']: {'baseline': q['baseline']} for q in queries}
    return copy.deepcopy((events, queries, graph, summary, {'symbolic_edges': []}, frozen))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.args = fixture()

    def validate(self):
        return validate_payload(*self.args, event_count=6)

    def test_complete_all_nodes_and_pair_matrix(self):
        result = self.validate()
        self.assertEqual(result['counts']['event_pair_rechecks'], 18)
        self.assertEqual(result['counts']['triangle_pairs_excluded_by_exact_aabb'], 576)
        self.assertEqual(result['counts']['actual_local_rechecks'], 36)
        self.assertEqual(result['jointly_admitted_events'], 4)

    def test_missing_pair_is_not_vacuous_pass(self):
        self.args[1][0]['pair_proofs'].clear()
        with self.assertRaisesRegex(ValueError, 'Incomplete pair'):
            self.validate()

    def test_duplicate_pair_cannot_fill_denominator(self):
        self.args[1][0]['pair_proofs'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate or extraneous pair'):
            self.validate()

    def test_missing_rejected_node(self):
        self.args[1][0]['events'] = self.args[1][0]['events'][:2]
        with self.assertRaisesRegex(ValueError, 'omitted a rejected'):
            self.validate()

    def test_missing_natural_query(self):
        self.args[1].pop(0)
        with self.assertRaisesRegex(ValueError, 'Missing or extraneous actual query'):
            self.validate()

    def test_duplicate_actual_query(self):
        self.args[1].append(self.args[1][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate actual query'):
            self.validate()

    def test_window_overlap_not_ignored_by_root_label(self):
        for event in self.args[0][3:]:
            event['schedule']['bounds']['lower'] = '3/2'
            event['compiler']['source']['levels']['lower'] = '3/2'
        with self.assertRaisesRegex(ValueError, 'not strictly disjoint'):
            self.validate()

    def test_query_phase_binding(self):
        self.args[1][0]['query'] = {**self.args[1][0]['query'], 'physical_time_hex': '0x1p+1'}
        with self.assertRaisesRegex(ValueError, 'frozen schedule'):
            self.validate()

    def test_graph_decision_cannot_be_promoted(self):
        rejected = next(c for c in self.args[2]['components'] if c['independently_rejected_members'])
        rejected['decision'] = 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE'
        with self.assertRaisesRegex(ValueError, 'replay mismatch'):
            self.validate()

    def test_actual_support_bounds_include_center(self):
        row = copy.deepcopy(self.args[1][0]['events'][0])
        row['center'] = [0.5, 0.5, 2.0]
        with self.assertRaisesRegex(ValueError, 'exact extrema'):
            validate_footprint(row, self.args[0][0], self.args[1][0]['baseline'])

    def test_source_triangle_cannot_escape_bound(self):
        row = copy.deepcopy(self.args[1][0]['events'][0])
        row['source_triangles'][0][0] = 7
        with self.assertRaisesRegex(ValueError, 'escape'):
            validate_footprint(row, self.args[0][0], self.args[1][0]['baseline'])

    def test_incomplete_star_rejected(self):
        row = copy.deepcopy(self.args[1][0]['events'][0]); row['full_interface_star_enumerated'] = False
        with self.assertRaisesRegex(ValueError, 'Incomplete interface'):
            validate_footprint(row, self.args[0][0], self.args[1][0]['baseline'])

    def test_conflicting_hash_bindings(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            merge_bindings({'one.py': 'a'}, {'one.py': 'b'})
        self.assertEqual(len(merge_bindings({'one.py': 'a'}, {'one.py': 'a'})), 1)

    def test_shared_support_never_axis_shortcut(self):
        query = self.args[1][0]; a, b = copy.deepcopy(query['events'][:2])
        b['consumed_owners'] = a['consumed_owners']
        self.assertEqual(recheck_pair(a, b, query['baseline'])['status'], 'UNSUPPORTED_SHARED_SUPPORT')

    def test_different_element_ids_do_not_create_shared_feature(self):
        query = self.args[1][0]; a, b = copy.deepcopy(query['events'][:2])
        b['element'] = 1; b['boundary_actual_ids'] = a['boundary_actual_ids']
        b['source_face_rows'] = a['source_face_rows']; b['consumed_owners'] = [[1, *o[1:]] for o in a['consumed_owners']]
        self.assertEqual(recheck_pair(a, b, query['baseline'])['triangle_pairs_excluded'], 32)

    def test_exact_overlap_not_false_pass(self):
        query = self.args[1][0]; a, b = copy.deepcopy(query['events'][:2])
        b['boundary_coordinates'] = a['boundary_coordinates']; b['center'] = a['center']; b['bounds'] = a['bounds']
        result = recheck_pair(a, b, query['baseline'])
        self.assertEqual(result['status'], 'REJECT_POLICY_CONTACT')
        self.assertGreater(result['exact_calls'], 0)


if __name__ == '__main__':
    unittest.main()
