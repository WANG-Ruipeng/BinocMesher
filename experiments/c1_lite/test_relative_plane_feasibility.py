"""Synthetic conditional-plane tests; never read the E2 cache or admit a plan."""

from fractions import Fraction as F
from itertools import permutations
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_relative_plane_feasibility import audit_relative_plane, epsilon_conditions


BOUNDARY = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
IDS = ('a', 'b', 'c', 'd')
CENTER = (F(1, 2), F(1, 2), 0)


def static(triangle, vids=('a', 'p', 'q'), **kwargs):
    return audit_relative_plane(BOUNDARY, BOUNDARY, IDS, CENTER, CENTER,
                                triangle, triangle, vids, projection='xz', **kwargs)


class RelativePlaneFeasibilityTests(unittest.TestCase):
    def test_shared_vertex_strict_plane_and_conditional_status(self):
        result = static((BOUNDARY[0], (F(1, 2), 0, -1), (F(1, 2), -1, 0)))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['allowed_contact'], 'shared_vertex')
        self.assertEqual(result['minimum_signed_margin'], {'numerator': 1, 'denominator': 2})
        self.assertFalse(result['runtime_admitted'])
        self.assertFalse(result['actual_source_vid_mapping_proven'])
        self.assertFalse(result['actual_coordinate_error_bound_proven'])
        self.assertFalse(result['actual_double_time_selector_proven'])
        self.assertTrue(all(not row['actual_bound_proven'] for row in result['epsilon_conditions']))
        json.dumps(result, allow_nan=False)

    def test_shared_edge_structural_zero(self):
        result = static((BOUNDARY[0], BOUNDARY[1], (F(1, 2), -1, -1)), ('a', 'b', 'p'))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['allowed_contact'], 'shared_edge')
        self.assertEqual(len(result['structural_zero_checks']), 2)

    def test_unshared_below_edge_is_disjoint(self):
        triangle = ((F(1, 2), 0, -1), (F(1, 4), -1, 0), (F(3, 4), -1, 0))
        result = static(triangle, ('p', 'q', 'r'))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['allowed_contact'], 'empty')

    def test_wrong_identity_rejected_and_alias_not_admitted(self):
        bad = static(((0, 0, F(1, 10**30)), (F(1, 2), 0, -1), (F(1, 2), -1, 0)))
        self.assertEqual(bad['status'], 'REJECT')
        alias = static((BOUNDARY[0], (F(1, 2), 0, -1), (F(1, 2), -1, 0)), ('fake_a', 'p', 'q'))
        self.assertEqual(alias['status'], 'UNKNOWN')

    def test_nonshared_zero_is_not_strict(self):
        result = static((BOUNDARY[0], (F(1, 2), 0, 0), (F(1, 2), -1, 0)))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['witness']['kind'], 'relative_plane_sign_not_strict')

    def test_center_is_part_of_strict_patch_constraints(self):
        triangle = (BOUNDARY[0], (F(1, 2), 0, -1), (F(1, 2), -1, 0))
        center = (F(1, 2), F(1, 2), -1)
        result = audit_relative_plane(BOUNDARY, BOUNDARY, IDS, center, center,
                                      triangle, triangle, ('a', 'p', 'q'), projection='xz')
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['witness']['role'], 'patch_center')

    def test_exact_error_formula_and_insufficient_budget(self):
        rows = epsilon_conditions(F(1), F(2), F(1), (F(1, 100), F(1)))
        value = rows[0]['functional_error_upper_bound']
        self.assertEqual(F(value['numerator'], value['denominator']), F(26, 625))
        self.assertTrue(rows[0]['sufficient_if_actual_bound_and_common_identity_hold'])
        self.assertFalse(rows[1]['sufficient_if_actual_bound_and_common_identity_hold'])
        with self.assertRaises(ValueError):
            epsilon_conditions(F(1), F(2), F(1), (-F(1),))

    def test_interior_quadratic_failure_not_hidden_by_endpoint_signs(self):
        def scene(time):
            s, eta, delta2 = time-F(1, 8), F(1, 8192), F(1, 64)**2
            boundary = ((F(0), F(0), F(0)), (s, F(1), F(0)),
                        (s-1, F(1), F(0)), (-F(1), F(0), F(0)))
            triangle = ((-delta2, -s, F(0)), (-delta2-eta*s, -s-eta, F(1)),
                        (-delta2-2*eta*s, -s-2*eta, F(0)))
            return boundary, ((s-1)/2, F(1, 2), F(0)), triangle
        b0, c0, p0 = scene(F(0))
        b1, c1, p1 = scene(F(1))
        for b, c, p in ((b0, c0, p0), (b1, c1, p1)):
            self.assertEqual(audit_relative_plane(b, b, IDS, c, c, p, p, ('p', 'q', 'r'), tilt=0)['status'], 'PASS')
        result = audit_relative_plane(b0, b1, IDS, c0, c1, p0, p1, ('p', 'q', 'r'), tilt=0)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['witness']['normalized_time'], {'numerator': 1, 'denominator': 8})

    def test_vertex_permutation_does_not_change_margin(self):
        triangle = (BOUNDARY[0], (F(1, 2), 0, -1), (F(1, 2), -1, 0))
        vids = ('a', 'p', 'q')
        for order in permutations(range(3)):
            result = static(tuple(triangle[i] for i in order), tuple(vids[i] for i in order))
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['minimum_signed_margin'], {'numerator': 1, 'denominator': 2})

    def test_rational_budget_returns_unknown(self):
        triangle = (BOUNDARY[0], (F(1, 2), 0, -F(1, 2**100)), (F(1, 2), -1, 0))
        self.assertEqual(static(triangle, max_rational_bits=32)['status'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
