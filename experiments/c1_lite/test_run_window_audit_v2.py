import copy
from fractions import Fraction as F
import time
import unittest
from run_window_audit_v2 import audit
from runtime_retained import RetainedUnit
from test_run_window_audit import source_fixture, triangle


def source():
    value = source_fixture()
    for row in value['segments']+value['breakpoint_points']:
        row['source_faces'] = [['a', 'b', 'c'], ['a', 'c', 'd']]
    return value


def provider_for(triangles):
    rows = copy.deepcopy(triangles)
    for row in rows:
        row['raw_occurrence_count'] = len(row['owners'])
    return lambda t0, t1, owners: RetainedUnit(t0, t1, (), tuple(rows), 'synthetic')


class V2IntegrationTests(unittest.TestCase):
    def test_all_cells_and_root_fallback(self):
        report = audit(source(), provider_for([triangle([[2, 0, 0], [3, 0, 0], [2, 1, 0]])]))
        self.assertTrue(report['all_units_scanned'])
        self.assertEqual(len(report['units']), 5)
        self.assertEqual(report['pair_counts']['PASS'], 5)
        self.assertFalse(report['runtime_admitted'])
        self.assertTrue(all(x == 'BASELINE' for x in report['fallback_selection_checks'].values()))

    def test_interface_degenerate_policy_is_not_a_new_collision(self):
        report = audit(source(), provider_for([triangle([[0, 0, 0], [0, 0, 0], [-1, 0, 0]], ('a', 'a', 'x'))]))
        self.assertEqual(report['pair_counts']['REJECT'], 5)
        self.assertIn('interface_degeneracy_policy', report['admission']['rejected_gates'])
        self.assertEqual(report['diagnostics'][0]['kind'], 'BASELINE_INTERFACE_DEGENERACY_POLICY')
        self.assertIn('NOT a new collision', report['diagnostics'][0]['detail']['certificate']['reason'])

    def test_representation_failure_not_promoted_by_ideal_success(self):
        for value in (True, False):
            report = audit(source(), provider_for([]), endpoint_evidence={'prospective_binary32_endpoint_contract_pass': value})
            self.assertEqual(report['admission']['gates']['endpoint_realization']['status'], 'UNKNOWN' if value else 'REJECT')
            self.assertFalse(report['runtime_admitted'])

    def test_timeout_and_provider_failure_fail_closed(self):
        report = audit(source(), provider_for([]), deadline=time.monotonic()-1)
        self.assertFalse(report['all_units_scanned'])
        self.assertEqual(report['admission']['gates']['complete_temporal_partition']['status'], 'UNKNOWN')
        def broken(*args):
            raise ValueError('unpartitioned raw interval')
        report = audit(source(), broken)
        self.assertFalse(report['all_units_scanned'])
        self.assertEqual(report['diagnostics_total'], 5)

    def test_bad_source_does_not_call_provider(self):
        value = source()
        value['coordinate_model'] = 'runtime_float32'
        def forbidden(*args):
            self.fail('provider must not be called')
        report = audit(value, forbidden)
        self.assertEqual(report['status'], 'REJECT')
        self.assertFalse(report['runtime_admitted'])


if __name__ == '__main__':
    unittest.main()
