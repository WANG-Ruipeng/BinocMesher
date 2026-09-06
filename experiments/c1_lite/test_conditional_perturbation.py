"""Synthetic perturbation-envelope tests, not actual runtime assertions."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditional_perturbation import certify_local_perturbation, certify_pair_perturbation

B = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
IDS = ('a', 'b', 'c', 'd')
C = (F(1, 2), F(1, 2), 0)
EPS = F(1, 1024)


def pair(triangle, ids=('p', 'q', 'r'), epsilon=EPS):
    return certify_pair_perturbation(B, B, IDS, C, C, triangle, triangle, ids, epsilon)


class ConditionalPerturbationTests(unittest.TestCase):
    def test_local_fan_whole_coordinate_envelope(self):
        result = certify_local_perturbation(B, B, C, C, EPS)
        self.assertEqual(result['status'], 'PASS_CONDITIONAL')
        self.assertEqual(len(result['proof']['checks']), 12)
        self.assertTrue(all(c['sufficient'] for c in result['proof']['checks']))
        self.assertFalse(result['runtime_admitted'])
        json.dumps(result, allow_nan=False)

    def test_local_large_epsilon_is_unknown(self):
        self.assertEqual(certify_local_perturbation(B, B, C, C, F(1, 4))['status'], 'UNKNOWN')

    def test_strict_axis_gap_has_error_bound(self):
        result = pair(((3, 0, 0), (3, 1, 0), (4, 0, 0)))
        self.assertEqual(result['status'], 'PASS_CONDITIONAL')
        self.assertEqual(result['proof']['kind'], 'ROBUST_FIXED_DIRECTION_CONVEX_HULL')
        self.assertTrue(result['proof']['includes_center'])
        self.assertEqual(result['proof']['error_upper_bound'], {'numerator': 1, 'denominator': 512})

    def test_shared_corner_structural_zero_and_regular_normal(self):
        result = pair((B[0], (-2, 1, 0), (1, -2, 0)), ('a', 'p', 'q'))
        self.assertEqual(result['status'], 'PASS_CONDITIONAL')
        self.assertEqual(result['proof']['kind'], 'ROBUST_SHARED_CORNER_RELATIVE_PLANE')
        self.assertEqual(result['regularity']['status'], 'PASS_CONDITIONAL')

    def test_shared_edge_uses_tilted_relative_plane(self):
        result = pair((B[0], B[1], (F(1, 2), 0, -1)), ('a', 'b', 'p'))
        self.assertEqual(result['status'], 'PASS_CONDITIONAL')
        self.assertEqual(result['proof']['kind'], 'ROBUST_SOURCE_EDGE_RELATIVE_PLANE')
        self.assertEqual(result['proof']['allowed_contact'], 'shared_edge')
        self.assertNotEqual(result['proof']['tilt']['numerator'], 0)

    def test_zero_feature_unshared_contact_becomes_robust_separation(self):
        result = pair(((F(1, 2), 0, -1), (F(1, 4), -1, 0), (F(3, 4), -1, 0)))
        self.assertEqual(result['status'], 'PASS_CONDITIONAL')
        self.assertEqual(result['proof']['allowed_contact'], 'empty')

    def test_smaller_epsilon_is_a_stronger_condition_not_a_measured_bound(self):
        d = F(1, 4096)
        triangle = ((F(1, 2), -d, 0), (F(1, 4), -2*d, 0), (F(3, 4), -2*d, 0))
        self.assertEqual(pair(triangle)['status'], 'UNKNOWN')
        small = pair(triangle, epsilon=F(1, 1 << 20))
        self.assertEqual(small['status'], 'PASS_CONDITIONAL')
        self.assertFalse(small['actual_error_bound_proven'])

    def test_actual_center_is_never_omitted(self):
        triangle = ((F(1, 4), F(1, 4), 1), (F(3, 4), F(1, 4), 1), (F(1, 2), F(3, 4), 1))
        center = (F(1, 2), F(1, 2), 2)
        result = certify_pair_perturbation(B, B, IDS, center, center, triangle, triangle, ('p', 'q', 'r'), EPS)
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_shared_identity_conflict_is_rejected(self):
        result = pair(((0, 0, F(1, 10**30)), (-2, 1, 0), (1, -2, 0)), ('a', 'p', 'q'))
        self.assertEqual(result['status'], 'REJECT')

    def test_remote_degeneracy_can_be_separated_without_cleaning_it(self):
        result = pair(((0, -2, 0), (0, -2, 0), (1, -2, 0)), ('p', 'p', 'q'))
        self.assertEqual(result['status'], 'PASS_CONDITIONAL')
        self.assertIsNone(result['regularity'])

    def test_shared_degenerate_face_not_admitted(self):
        result = pair((B[0], B[0], (0, -1, 0)), ('a', 'a', 'p'))
        self.assertEqual(result['status'], 'REJECT')

    def test_shared_nearly_degenerate_triangle_lacks_robust_normal(self):
        result = pair((B[0], B[1], (F(1, 2), -F(1, 1 << 30), 0)), ('a', 'b', 'p'), F(1, 1 << 20))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['regularity']['kind'], 'SHARED_TRIANGLE_REGULARITY_UNRESOLVED')


if __name__ == '__main__':
    unittest.main()
