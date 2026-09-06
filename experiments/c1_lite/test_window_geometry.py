#!/usr/bin/env python3
"""Synthetic exact-certificate tests; no real cache or renderer is invoked."""

from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_geometry import certify_segment


SQUARE = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
CENTER = (F(1, 2), F(1, 2), 0)


def rational(value):
    return F(value['numerator'], value['denominator'])


def transformed(time, shift, off_diagonal):
    """Affine map [[time-shift, off_diagonal], [1, time-shift]]."""
    def transform(point):
        x, y, z = point
        return ((time-shift)*x+off_diagonal*y, x+(time-shift)*y, z)
    return tuple(transform(point) for point in SQUARE), transform(CENTER)


class WindowGeometryTests(unittest.TestCase):
    def test_static_square_exact_margin_and_json(self):
        result = certify_segment(SQUARE, SQUARE, CENTER, CENTER)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['constraints']), 12)
        self.assertEqual(rational(result['minimum_signed_projected_double_fan_area']), F(1, 2))
        self.assertEqual(rational(result['minimum_3d_fan_area_lower_bound']), F(1, 4))
        self.assertTrue(result['local_segment_geometry_certified'])
        json.dumps(result, allow_nan=False)

    def test_clockwise_and_cyclic_shift_are_supported(self):
        reverse = tuple(reversed(SQUARE))
        shifted = SQUARE[2:]+SQUARE[:2]
        for boundary, orientation in ((reverse, -1), (shifted, 1)):
            result = certify_segment(boundary, boundary, CENTER, CENTER)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['orientation_xy'], orientation)

    def test_arbitrary_affine_heights_do_not_change_graph_proof(self):
        start = tuple((x, y, i*i-7) for i, (x, y, _) in enumerate(SQUARE))
        end = tuple((x+2, y-3, 100-i) for i, (x, y, _) in enumerate(SQUARE))
        result = certify_segment(start, end, (F(1, 2), F(1, 2), -100),
                                 (F(5, 2), -F(5, 2), 77))
        self.assertEqual(result['status'], 'PASS')

    def test_mixed_bernstein_coefficients_are_not_a_rejection(self):
        start, c0 = transformed(F(0), F(1, 2), -F(1, 16))
        end, c1 = transformed(F(1), F(1, 2), -F(1, 16))
        result = certify_segment(start, end, c0, c1)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(any(not item['bernstein_all_strictly_positive'] for item in result['constraints']))
        self.assertTrue(all(rational(item['exact_minimum']) > 0 for item in result['constraints']))

    def test_five_normal_probes_miss_interior_orientation_flip(self):
        shift, off_diagonal = F(1, 8), F(1, 64)**2
        for time in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
            boundary, center = transformed(time, shift, off_diagonal)
            self.assertEqual(certify_segment(boundary, boundary, center, center)['status'], 'PASS')
        start, c0 = transformed(F(0), shift, off_diagonal)
        end, c1 = transformed(F(1), shift, off_diagonal)
        result = certify_segment(start, end, c0, c1)
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(rational(result['witness']['normalized_time']), shift)
        self.assertLess(rational(result['witness']['exact_signed_value']), 0)

    def test_five_normal_probes_miss_tangential_projected_collapse(self):
        shift = F(1, 8)
        for time in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
            boundary, center = transformed(time, shift, F(0))
            self.assertEqual(certify_segment(boundary, boundary, center, center)['status'], 'PASS')
        start, c0 = transformed(F(0), shift, F(0))
        end, c1 = transformed(F(1), shift, F(0))
        result = certify_segment(start, end, c0, c1)
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(rational(result['witness']['normalized_time']), shift)
        self.assertEqual(rational(result['witness']['exact_signed_value']), 0)

    def test_outside_or_boundary_center_has_exact_witness(self):
        for center in ((2, F(1, 2), 0), (0, F(1, 2), 100)):
            result = certify_segment(SQUARE, SQUARE, center, center)
            self.assertEqual(result['status'], 'REJECT')
            self.assertEqual(result['witness']['kind'], 'fan_center_strict_halfplane')

    def test_bow_tie_cycle_and_collapsed_footprint_rejected(self):
        for boundary in ((SQUARE[0], SQUARE[2], SQUARE[1], SQUARE[3]),
                         ((0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 2))):
            result = certify_segment(boundary, boundary, CENTER, CENTER)
            self.assertEqual(result['status'], 'REJECT')
            self.assertEqual(result['witness']['kind'], 'boundary_strict_halfplane')

    def test_endpoint_degeneracy_is_not_hidden_by_open_interval(self):
        end = tuple((0, y, z) for _, y, z in SQUARE)
        result = certify_segment(SQUARE, end, CENTER, (0, F(1, 2), 0))
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(rational(result['witness']['normalized_time']), 1)

    def test_fraction_json_and_exact_binary64_input(self):
        center = ({'numerator': 1, 'denominator': 2}, 0.1, 0)
        result = certify_segment(SQUARE, SQUARE, center, center)
        self.assertEqual(result['status'], 'PASS')
        edge_zero = next(item for item in result['constraints']
                         if item['kind'] == 'fan_center_strict_halfplane' and item['edge'] == [0, 1])
        self.assertEqual(rational(edge_zero['exact_minimum']), F.from_float(0.1))
        self.assertNotEqual(rational(edge_zero['exact_minimum']), F(1, 10))

    def test_arithmetic_budget_is_unknown_not_invalid_geometry(self):
        center = (F(1, 2), F(1, 2), F(1, 2**512))
        result = certify_segment(SQUARE, SQUARE, center, center, max_rational_bits=128)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['witness'])
        self.assertFalse(result['local_segment_geometry_certified'])

    def test_invalid_inputs_are_diagnostics_not_geometric_witnesses(self):
        for boundary, center in ((SQUARE[:3], CENTER), (SQUARE, (0, float('nan'), 0)),
                                 (SQUARE, (True, 0, 0))):
            result = certify_segment(boundary, boundary, center, center)
            self.assertEqual(result['status'], 'REJECT')
            self.assertEqual(result['witness']['kind'], 'invalid_input')

    def test_scope_never_claims_exterior_or_runtime_certificate(self):
        result = certify_segment(SQUARE, SQUARE, CENTER, CENTER)
        limitations = ' '.join(result['not_certified'])
        self.assertIn('retained exterior', limitations)
        self.assertIn('Binary32', limitations)
        self.assertIn('Source branch/owner', limitations)


if __name__ == '__main__':
    unittest.main()
