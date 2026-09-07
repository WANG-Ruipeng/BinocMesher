import copy
import unittest

from forest_component_graph import build_graph


def event(name, decision='ADMITTED_REQUESTED_SCHEDULE'):
    return dict(event_id=name, root='3/2', decision=decision,
                schedule={'all_queries': [{'key': 'root', 'active': True}]})


def footprint(name, offset=0, element=0):
    return dict(event_id=name, element=element, status='COMPLETE_ACTUAL_REQUESTED_SUPPORT',
                consumed_owners=[[element, offset, 0, 0, 0, 0, 0]],
                boundary_actual_ids=list(range(offset, offset+4)),
                source_face_rows=[offset, offset+1], retained_star_face_rows=[offset+2],
                bounds=[[offset, 0, 0], [offset+1, 1, 1]])


def graph(events, rows):
    return build_graph(events, [{'root': '3/2', 'query': {'key': 'root'}, 'events': rows}])


class GraphTests(unittest.TestCase):
    def test_independent(self):
        result = graph([event('a'), event('b')], [footprint('a'), footprint('b', 10)])
        self.assertEqual(len(result['components']), 2)
        self.assertEqual(result['edges'], [])

    def test_rejected_shared_boundary_blocks_component(self):
        result = graph([event('a'), event('b', 'REJECTED_FIXED_POLICY')],
                       [footprint('a'), footprint('b', 2)])
        self.assertEqual(result['components'][0]['decision'], 'FAIL_CLOSED_REJECTED_MEMBER')

    def test_cross_element_spatial_overlap(self):
        result = graph([event('a'), event('b')], [footprint('a'), footprint('b', element=1)])
        self.assertEqual(list(result['edges'][0]['reasons']), ['ACTUAL_SUPPORT_AABB_POSSIBLE_INTERACTION'])

    def test_unknown_is_not_empty(self):
        rows = [footprint('a'), footprint('b', 1000)]
        rows[0] = dict(event_id='a', status='UNKNOWN', reason='missing')
        result = graph([event('a'), event('b')], rows)
        self.assertEqual(len(result['components']), 1)
        self.assertEqual(result['unknown_actual_support_rows'], 1)

    def test_missing_population_fails(self):
        with self.assertRaises(ValueError):
            graph([event('a'), event('b')], [footprint('a')])

    def test_missing_query_fails(self):
        with self.assertRaises(ValueError):
            build_graph([event('a')], [])

    def test_retained_face_overlap_and_source_dependency(self):
        rows = [footprint('a'), footprint('b', 100)]
        rows[1]['retained_star_face_rows'] = [0, 2]
        result = graph([event('a'), event('b')], rows)
        reasons = result['edges'][0]['reasons']
        self.assertIn('RETAINED_INTERFACE_FACE_OVERLAP', reasons)
        self.assertIn('SOURCE_RETAINED_INTERFACE_DEPENDENCY', reasons)

    def test_permutation_stable_components(self):
        events, rows = [event('a'), event('b')], [footprint('a'), footprint('b', 2)]
        before = copy.deepcopy(rows)
        self.assertEqual(graph(events, rows)['components'], graph(events[::-1], rows[::-1])['components'])
        self.assertEqual(before, rows)


if __name__ == '__main__':
    unittest.main()
