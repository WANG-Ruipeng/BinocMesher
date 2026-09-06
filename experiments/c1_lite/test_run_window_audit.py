#!/usr/bin/env python3
"""Synthetic integration tests; never read a real source cache."""
import copy
from fractions import Fraction as F
import unittest

from run_window_audit import (audit_source_report, center_at, qualified_vid,
                              aggregate_status, check_junctions)
from window_source import fj


def encode(points):
    return [[fj(F(value)) for value in point] for point in points]


def source_fixture():
    cycle = ['a', 'b', 'c', 'd']
    boundary = encode([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    owner = [0, 0, 0, 0, 0, 0, 0]
    source = {'event_id': 'synthetic', 'status': 'PASS_SOURCE_BRANCH_CONTRACT',
        'coordinate_model': 'ideal rational interpolation of serialized HV binary32',
        'piecewise_source_ownership_certificate': True,
        'half_window_owner_sets_constant': {'left': True, 'right': True},
        'junction_boundary_cycle_and_orientation_equal': True,
        'levels': {'lower': fj(0), 'root': fj(1), 'upper': fj(2)},
        'boundary_cycle': cycle,
        'anchors': {'lower': {'position': encode([[F(1, 2), F(1, 2), 0]])[0]},
                    'root': {'position': encode([[F(1, 2), F(1, 2), 1]])[0]},
                    'upper': {'position': encode([[F(1, 2), F(1, 2), 0]])[0]}},
        'segments': [], 'breakpoint_points': []}
    for t0, t1 in ((0, 1), (1, 2)):
        source['segments'].append({'t0': fj(t0), 't1': fj(t1), 'owners': [owner],
            'boundary_cycle': cycle,
            'boundary': [{'source_vid': name, 'position_t0': position, 'position_t1': position}
                         for name, position in zip(cycle, boundary)]})
    for tau in (0, 1, 2):
        source['breakpoint_points'].append({'time': fj(tau), 'owners': [owner],
            'boundary_cycle': cycle,
            'boundary': [{'source_vid': name, 'position': position}
                         for name, position in zip(cycle, boundary)]})
    return source


def triangle(points, ids=('x', 'y', 'z'), element=0):
    return {'element': element, 'source_vertices': list(ids),
            'positions_t0': encode(points), 'positions_t1': encode(points),
            'owners': [[element, 0, 0, 99, 0, 0, 0]]}


class WindowAuditTests(unittest.TestCase):
    def test_exact_center_and_junctions(self):
        source = source_fixture()
        self.assertEqual(center_at(source, F(1, 2)), encode([[F(1, 2), F(1, 2), F(1, 2)]])[0])
        self.assertEqual(check_junctions(source)['status'], 'PASS')
        bad = copy.deepcopy(source)
        bad['segments'][0]['boundary'][0]['position_t1'] = encode([[0, F(1, 100), 0]])[0]
        with self.assertRaisesRegex(ValueError, 'differs'):
            check_junctions(bad)

    def test_all_branches_and_actual_singletons_are_requested(self):
        calls = []
        exterior = triangle([[2, 0, 0], [3, 0, 0], [2, 1, 0]])
        def provider(t0, t1, owners):
            calls.append((t0, t1))
            yield exterior
        report = audit_source_report(source_fixture(), provider)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(calls, [(F(0), F(1)), (F(1), F(2)),
                                 (F(0), F(0)), (F(1), F(1)), (F(2), F(2))])
        self.assertFalse(report['runtime_admission'])
        self.assertEqual(report['event_isolation_certificate'], 'UNKNOWN')

    def test_same_element_shared_boundary_is_allowed(self):
        shared = triangle([[0, 0, 0], [1, 0, 0], [F(1, 2), -1, 0]], ('a', 'b', 'z'))
        report = audit_source_report(source_fixture(), lambda *args: iter([shared]))
        self.assertEqual(report['status'], 'PASS')

    def test_different_element_same_vid_is_not_a_shared_endpoint(self):
        other = triangle([[0, 0, 0], [1, 0, 0], [F(1, 2), -1, 0]], ('a', 'b', 'z'), 1)
        report = audit_source_report(source_fixture(), lambda *args: iter([other]))
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertNotEqual(qualified_vid(0, 'a'), qualified_vid(1, 'a'))

    def test_repeated_vid_is_unknown_not_new_collision_or_skip(self):
        repeated = triangle([[F(1, 2), F(1, 2), 0], [F(1, 2), F(1, 2), 0], [1, F(1, 2), 0]], ('x', 'x', 'z'))
        report = audit_source_report(source_fixture(), lambda *args: iter([repeated]))
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertTrue(all(unit['retained_counts']['UNKNOWN'] == 1 for unit in report['units']))
        self.assertEqual(report['diagnostics'][0]['kind'], 'SUPERSET_DEGENERATE_UNRESOLVED')

    def test_incomplete_retained_scan_is_unknown(self):
        def broken(*args):
            yield triangle([[2, 0, 0], [3, 0, 0], [2, 1, 0]])
            raise ValueError('synthetic truncated stream')
        report = audit_source_report(source_fixture(), broken)
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertTrue(all(not unit['retained_scan_complete'] for unit in report['units']))

    def test_bad_identity_is_rejected_and_diagnostics_are_bounded(self):
        mismatch = triangle([[2, 0, 0], [3, 0, 0], [2, 1, 0]], ('a', 'y', 'z'))
        report = audit_source_report(source_fixture(), lambda *args: iter([mismatch]*4), max_diagnostics=8)
        self.assertEqual(report['status'], 'REJECT')
        self.assertEqual(len(report['diagnostics']), 8)
        self.assertEqual(report['diagnostics_total'], 20)

    def test_restrictive_local_geometry_does_not_trigger_exterior(self):
        source = source_fixture()
        source['anchors']['root']['position'] = encode([[2, F(1, 2), 0]])[0]
        calls = []
        def provider(*args):
            calls.append(args)
            return iter([])
        report = audit_source_report(source, provider)
        self.assertEqual(report['status'], 'REJECT')
        # Only the two outer singleton graph fans remain admissible.
        self.assertEqual(len(calls), 2)
        self.assertEqual(aggregate_status(['PASS', 'UNKNOWN']), 'UNKNOWN')


    def test_far_degenerate_axis_certificate_keeps_the_denominator(self):
        repeated = triangle([[2, 0, 0], [2, 0, 0], [3, 1, 0]], ('x', 'x', 'z'))
        report = audit_source_report(source_fixture(), lambda *args: iter([repeated]))
        self.assertEqual(report['status'], 'PASS')
        for unit in report['units']:
            self.assertEqual(unit['retained_triangles_checked'], 1)
            self.assertEqual(unit['retained_counts']['PASS'], 1)
            self.assertEqual(unit['retained_certificate_kinds']['SUPERSET_DEGENERATE_STRICT_AXIS'], 1)

    def test_degenerate_axis_certificate_includes_center(self):
        source = source_fixture()
        source['anchors']['root']['position'] = encode([[F(1, 2), F(1, 2), 2]])[0]
        repeated = triangle([[F(1, 2), F(1, 2), 1], [F(1, 2), F(1, 2), 1], [F(3, 4), F(1, 2), 1]], ('x', 'x', 'z'))
        report = audit_source_report(source, lambda *args: iter([repeated]))
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertEqual(report['units'][0]['exterior_status'], 'UNKNOWN')

    def test_degenerate_axis_touch_is_not_strict_separation(self):
        repeated = triangle([[1, F(1, 2), 0], [1, F(1, 2), 0], [2, F(1, 2), 0]], ('x', 'x', 'z'))
        report = audit_source_report(source_fixture(), lambda *args: iter([repeated]))
        self.assertEqual(report['status'], 'UNKNOWN')

    def test_degenerate_same_element_identity_checked_before_far_axis(self):
        repeated = triangle([[2, 0, 0], [2, 0, 0], [3, 1, 0]], ('a', 'a', 'z'))
        report = audit_source_report(source_fixture(), lambda *args: iter([repeated]))
        self.assertEqual(report['status'], 'REJECT')
        self.assertIn('Shared SourceVID', report['diagnostics'][0]['detail']['certificate']['reason'])


if __name__ == '__main__':
    unittest.main()
