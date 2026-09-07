from copy import deepcopy
import unittest
from screen_contracts import validate_schedule_domain, summarize_visibility


def event(eid='a', root='3/2', lower='5/4', upper='7/4'):
    return {'event_id': eid, 'root': root, 'schedule': {
        'bounds': {'lower': lower, 'root': root, 'upper': upper},
        'natural': [{'key': 'frame_0001', 'kind': 'natural', 'frame_index_zero_based': 0,
                     'evaluation_tau': root, 'time_mode': 'physical', 'physical_time_hex': '0x1.0p+0', 'active': True}],
        'all_queries': [{'key': 'frame_0001', 'kind': 'natural', 'frame_index_zero_based': 0,
                         'evaluation_tau': root, 'time_mode': 'physical', 'physical_time_hex': '0x1.0p+0', 'active': True}]}}


class ContractTests(unittest.TestCase):
    def test_same_root_and_empty(self):
        self.assertEqual(validate_schedule_domain([])['root_count'], 0)
        self.assertEqual(validate_schedule_domain([event(), event('b')])['root_count'], 1)

    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            validate_schedule_domain([event(), event()])

    def test_different_same_root_bounds(self):
        with self.assertRaisesRegex(ValueError, 'HETEROGENEOUS'):
            validate_schedule_domain([event(), event('b', lower='1')])

    def test_overlapping_roots(self):
        with self.assertRaisesRegex(ValueError, 'CROSS_ROOT'):
            validate_schedule_domain([event(), event('b', '8/5', '7/5', '9/5')])

    def test_phase(self):
        e = event(); e['schedule']['all_queries'][0]['evaluation_tau'] = '2'
        with self.assertRaisesRegex(ValueError, 'PHASE'):
            validate_schedule_domain([e])

    def test_disjoint_roots(self):
        self.assertEqual(validate_schedule_domain([event(), event('b', '5/2', '9/4', '11/4')])['root_count'], 2)

    def fixture(self):
        e = event()
        e['schedule']['natural'].append({**e['schedule']['natural'][0], 'key': 'frame_0002', 'frame_index_zero_based': 1})
        comp = {'component_id': 'c', 'events': ['a'], 'decision': 'ADMITTED_COMPONENT_REQUESTED_SCHEDULE'}
        rows = [{'query': {'absolute_frame_number': 97+i},
                 'events': [{'event_id': 'a', 'domain_kind': 'FIXED_SOURCE_AND_REPLACEMENT',
                             'visible_baseline_source_pixels': 16, 'visible_replacement_pixels': 1, 'jointly_admitted': True}],
                 'components': [{'component_id': 'c', 'decision': comp['decision'],
                                 'visible_baseline_union_pixels': 16, 'visible_replacement_union_pixels': 1}]}
                for i in range(2)]
        spec = {'first_frame': 97, 'segment_id': 'forest_b'}
        rule = {'minimum_natural_frames': 2, 'minimum_baseline_support_pixels_per_frame': 16,
                'minimum_actual_replacement_pixels_on_each_qualifying_frame': 1}
        return [e], [comp], rows, spec, rule

    def test_threshold_uses_component_union(self):
        args = self.fixture()
        self.assertEqual(len(summarize_visibility(*args)['qualified_components']), 1)
        args[2][0]['components'][0]['visible_baseline_union_pixels'] = 15
        self.assertEqual(summarize_visibility(*args)['qualified_components'], [])

    def test_zero_and_unknown_denominator(self):
        args = self.fixture()
        for row in args[2]:
            row['events'][0]['visible_baseline_source_pixels'] = 0
            row['events'][0]['visible_replacement_pixels'] = 0
            row['components'][0]['visible_baseline_union_pixels'] = 0
            row['components'][0]['visible_replacement_union_pixels'] = 0
        self.assertIsNone(summarize_visibility(*args)['visible_source_admission_rate'])
        args[2][0]['events'][0]['visible_baseline_source_pixels'] = None
        self.assertEqual(summarize_visibility(*args)['unknown_support_visibility_events'], 1)

    def test_missing_frame_is_not_invisible(self):
        args = self.fixture(); args[2].pop()
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
            summarize_visibility(*args)

    def test_repeated_frame_is_not_two_frames(self):
        args = self.fixture(); args[2][1] = deepcopy(args[2][0])
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            summarize_visibility(*args)


if __name__ == '__main__':
    unittest.main()
