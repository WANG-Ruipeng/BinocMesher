import struct
import unittest
from fractions import Fraction as F

import window_source as s
from actual_evaluation_envelope import (analytic_bounds, inspect_pair, ordered_vid,
    selected_interval, gamma)


class EnvelopeTests(unittest.TestCase):
    def test_conservative_global_cache_magnitude_fits_target(self):
        report = analytic_bounds(1100, 2200, F(106, 5))
        self.assertTrue(report['exact_error_below_target'])
        self.assertGreater(s.fr(report['exact_rational_entry_coordinate_error']), F(1100, 2**24))

    def test_physical_double_model_omits_rational_conversion(self):
        report = analytic_bounds(10, 20, F(106, 5))
        self.assertLess(s.fr(report['physical_entry_actual_discrete_double_coordinate_error']),
                        s.fr(report['exact_rational_entry_coordinate_error']))

    def test_delta_t_not_silently_assumed(self):
        self.assertFalse(analytic_bounds(1, 1, 22)['delta_t_supplied_and_in_profile'])
        self.assertFalse(analytic_bounds(1, 1, 22, delta_t=F(1, 32))['delta_t_runtime_binding_verified'])
        with self.assertRaises(ValueError):
            analytic_bounds(1, 1, 22, delta_t=F(1, 2**200))

    def test_ordered_parser_does_not_canonicalize(self):
        blob = struct.pack('<ib3xib3x', 19, 3, 2, 1)
        self.assertEqual(ordered_vid(blob).text(), '19:3|2:1')

    def pair(self, ta=20, tb=24, pa=(0., 0., 0.), pb=(1., 1., 1.), da=0, db=0, va=1, vb=1):
        a, b = s.HVID(1, 0), s.HVID(2, 0)
        return s.SourceVID(a, b), {a: s.Hypervertex(pa, ta, da, va), b: s.Hypervertex(pb, tb, db, vb)}

    def test_reverse_time_does_not_pass(self):
        self.assertIn('REVERSED_ORIGINAL', inspect_pair(*self.pair(24, 20))['status'])

    def test_hidden_zero_span_jump_does_not_pass(self):
        self.assertIn('DISCONTINUOUS_ZERO', inspect_pair(*self.pair(20, 24, db=4, va=0))['status'])

    def test_constant_zero_span_is_continuous(self):
        result = inspect_pair(*self.pair(20, 20, pb=(0., 0., 0.)))
        self.assertEqual(result['status'], 'PASS_CONTINUOUS_CLAMPED_SOURCE')
        self.assertEqual(result['maximum_slope'], 0)

    def test_affine_lipschitz_constant(self):
        self.assertEqual(inspect_pair(*self.pair())['maximum_slope'], F(1, 4))

    def test_raw_original_singleton_predicates(self):
        record = s.Record(0, 20, 0, 0, (20, 21), ((),))
        self.assertIsNone(selected_interval(record, F(20)-F(1, 10**30), {0}))
        self.assertEqual(selected_interval(record, F(20), {0}), 0)
        self.assertEqual(selected_interval(record, F(21), {0}), 0)
        self.assertIsNone(selected_interval(record, F(21)+F(1, 10**30), {0}))

    def test_magnitude_gate_and_gamma(self):
        self.assertEqual(gamma(0), 0)
        with self.assertRaises(ValueError):
            analytic_bounds(2**41, 1, 22)
        with self.assertRaises(ValueError):
            gamma(-1)


if __name__ == '__main__':
    unittest.main()
