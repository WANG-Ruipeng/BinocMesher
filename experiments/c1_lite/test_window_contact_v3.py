#!/usr/bin/env python3
"""Synthetic v3 tests only: no event IDs, real caches, or runtime outputs."""

from fractions import Fraction as F
from itertools import permutations
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_contact_v2 import certify_contact as v2
from window_contact_v3 import certify_contact, oblique_directions
from window_geometry import certify_segment


BOUNDARY = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
VIDS = ('a', 'b', 'c', 'd')
CENTER = (F(1, 2), F(1, 2), 0)


def static(triangle, vids=('p', 'q', 'r'), **kwargs):
    return certify_contact(BOUNDARY, BOUNDARY, VIDS,
                           triangle, triangle, vids, **kwargs)


def hidden_halfplane_zero(time):
    s, eta = time-F(1, 8), F(1, 8192)
    boundary = ((F(0), F(0), F(0)), (s, F(1), F(0)),
                (s-1, F(1), F(0)), (-F(1), F(0), F(0)))
    triangle = ((F(0), -s, -F(1)), (-eta*s, -s-eta, -F(1)),
                (-2*eta*s, -s-2*eta, -F(2)))
    center = ((s-1)/2, F(1, 2), F(0))
    return boundary, triangle, center


class WindowContactV3Tests(unittest.TestCase):
    def assert_pass(self, result, kind=None, shared=None):
        self.assertEqual(result['status'], 'PASS', result.get('reason'))
        self.assertTrue(result['patch_triangle_interior_disjoint_certified'])
        if kind is not None:
            self.assertEqual(result['separation_certificate']['kind'], kind)
        if shared is not None:
            self.assertEqual(result['separation_certificate']['allowed_contact_source_vids'], shared)
        json.dumps(result, allow_nan=False)

    def test_strict_oblique_separator_after_v2_unknown(self):
        triangle = ((F(1, 2), F(5, 4), 0), (-F(1, 2), F(1, 2), 1),
                    (F(3, 2), F(1, 2), 1))
        self.assertEqual(v2(BOUNDARY, BOUNDARY, VIDS, triangle, triangle, ('p', 'q', 'r'),
                            center_start=CENTER, center_end=CENTER)['status'], 'UNKNOWN')
        result = static(triangle, center_start=CENTER, center_end=CENTER)
        self.assert_pass(result, 'STRICT_BOUNDED_OBLIQUE_CONVEX_HULL', [])
        self.assertEqual(result['separation_certificate']['endpoint_pair_count'], 30)
        self.assertTrue(result['separation_certificate']['includes_patch_center'])

    def test_direction_budget_eight_needed_and_seven_is_unknown(self):
        triangle = ((F(1, 2), F(5, 4), 0), (-F(1, 2), -6, 1), (F(3, 2), -6, 1))
        small = static(triangle, center_start=CENTER, center_end=CENTER, max_direction_weight=7)
        self.assertEqual(small['status'], 'UNKNOWN')
        self.assertIsNone(small['witness'])
        result = static(triangle, center_start=CENTER, center_end=CENTER, max_direction_weight=8)
        self.assert_pass(result, 'STRICT_BOUNDED_OBLIQUE_CONVEX_HULL')
        self.assertEqual(result['separation_certificate']['normal'], [0, 1, 8])
        self.assertEqual(result['separation_certificate']['minimum_signed_projection_gap'],
                         {'numerator': 1, 'denominator': 4})

    def test_direction_family_is_finite_unique_and_reproducible(self):
        directions = oblique_directions(8)
        self.assertEqual(directions, oblique_directions(8))
        self.assertEqual(len(directions), 90)
        self.assertEqual(len(set(directions)), len(directions))
        self.assertIn((0, 1, 8), directions)
        self.assertIn((8, 0, -1), directions)
        self.assertTrue(all(len([x for x in n if x]) == 2 for n in directions))
        self.assertTrue(all(min(abs(x) for x in n if x) == 1 for n in directions))

    def test_missing_center_disables_oblique_hull_proof(self):
        triangle = ((F(1, 2), F(5, 4), 0), (-F(1, 2), F(1, 2), 1), (F(3, 2), F(1, 2), 1))
        result = static(triangle)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertFalse(result['oblique_certificate_enabled'])

    def test_elevated_center_cannot_be_omitted_from_bounds(self):
        triangle = ((F(1, 2), F(5, 4), 0), (-F(1, 2), F(1, 2), 1), (F(3, 2), F(1, 2), 1))
        center = (F(1, 2), F(1, 2), 100)
        result = static(triangle, center_start=center, center_end=center)
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_oblique_motion_gap_is_checked_at_both_endpoints(self):
        start = ((F(1, 2), F(5, 4), 0), (-F(1, 2), F(1, 2), 1), (F(3, 2), F(1, 2), 1))
        end = ((F(1, 2), F(3, 4), 0), *start[1:])
        result = certify_contact(BOUNDARY, BOUNDARY, VIDS, start, end, ('p', 'q', 'r'),
                                 center_start=CENTER, center_end=CENTER)
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_zero_feature_unshared_point_below_edge_is_empty(self):
        triangle = ((F(1, 2), 0, -1), (F(1, 4), -1, 0), (F(3, 4), -1, 0))
        result = static(triangle)
        self.assert_pass(result, 'SOURCE_HALFPLANE_ZERO_FEATURE_SPATIAL', [])
        self.assertEqual(result['separation_certificate']['zero_feature_source_vids'], ['p'])

    def test_zero_feature_segment_meets_only_shared_vertex(self):
        triangle = (BOUNDARY[0], (F(1, 2), 0, -1), (F(1, 2), -1, 0))
        result = static(triangle, ('a', 'p', 'q'))
        self.assert_pass(result, 'SOURCE_HALFPLANE_ZERO_FEATURE_SPATIAL', ['a'])
        self.assertEqual(result['retained_regularity_certificate']['status'], 'PASS')
        self.assertEqual(result['v2_status'], 'UNKNOWN')

    def test_zero_feature_at_other_source_endpoint(self):
        triangle = (BOUNDARY[1], (F(1, 2), 0, -1), (F(1, 2), -1, 0))
        self.assert_pass(static(triangle, ('b', 'p', 'q')),
                         'SOURCE_HALFPLANE_ZERO_FEATURE_SPATIAL', ['b'])

    def test_coplanar_nonshared_zero_point_is_not_admitted(self):
        triangle = ((F(1, 2), 0, 0), (F(1, 4), -1, 0), (F(3, 4), -1, 0))
        result = static(triangle)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['witness'])

    def test_opposite_spatial_sides_have_undeclared_edge_contact(self):
        triangle = ((F(1, 4), 0, 1), (F(3, 4), 0, -1), (F(1, 2), -1, 0))
        self.assertEqual(static(triangle)['status'], 'UNKNOWN')

    def test_shared_id_not_a_whitelist_for_extra_edge_contact(self):
        triangle = (BOUNDARY[0], (F(1, 2), 0, 0), (F(1, 2), -1, 1))
        self.assertEqual(static(triangle, ('a', 'p', 'q'))['status'], 'UNKNOWN')

    def test_same_coordinate_different_vid_cannot_be_shared(self):
        triangle = (BOUNDARY[0], (0, -1, 0), (-1, -1, 0))
        self.assertEqual(static(triangle, ('fake_a', 'p', 'q'))['status'], 'UNKNOWN')

    def test_shared_identity_mismatch_rejected_before_new_rules(self):
        start = (BOUNDARY[0], (F(1, 2), 0, -1), (F(1, 2), -1, 0))
        end = ((0, 0, F(1, 10**30)), *start[1:])
        result = certify_contact(BOUNDARY, BOUNDARY, VIDS, start, end, ('a', 'p', 'q'))
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(result['witness']['kind'], 'shared_identity_mismatch')

    def test_shared_degeneracy_guard_preserved(self):
        triangle = (BOUNDARY[0], (0, -1, 0), (0, -2, 0))
        result = static(triangle, ('a', 'p', 'q'))
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(result['witness']['kind'], 'retained_triangle_degenerate')

    def test_five_probes_miss_spatial_functional_zero(self):
        def triangle(time):
            return (BOUNDARY[0], (F(1, 2), 0, 1-8*time), (F(1, 2), -1, 0))
        for time in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
            self.assert_pass(static(triangle(time), ('a', 'p', 'q')),
                             'SOURCE_HALFPLANE_ZERO_FEATURE_SPATIAL')
        result = certify_contact(BOUNDARY, BOUNDARY, VIDS, triangle(F(0)), triangle(F(1)),
                                 ('a', 'p', 'q'))
        self.assertEqual(result['retained_regularity_certificate']['status'], 'PASS')
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['witness'])

    def test_isolated_halfplane_zero_is_not_an_identically_zero_feature(self):
        for time in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
            boundary, triangle, _ = hidden_halfplane_zero(time)
            self.assert_pass(certify_contact(boundary, boundary, VIDS,
                                              triangle, triangle, ('p', 'q', 'r')))
        b0, p0, c0 = hidden_halfplane_zero(F(0))
        b1, p1, c1 = hidden_halfplane_zero(F(1))
        self.assertEqual(certify_segment(b0, b1, c0, c1)['status'], 'PASS')
        result = certify_contact(b0, b1, VIDS, p0, p1, ('p', 'q', 'r'))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertTrue(all(not c['supported_halfplane_pattern'] for c in result['zero_feature_candidates']))

    def test_permutations_rotations_and_clockwise_cycle(self):
        triangle = (BOUNDARY[0], (F(1, 2), 0, -1), (F(1, 2), -1, 0))
        triangle_ids = ('a', 'p', 'q')
        for reverse in (False, True):
            seed = tuple(reversed(BOUNDARY)) if reverse else BOUNDARY
            ids = tuple(reversed(VIDS)) if reverse else VIDS
            for offset in range(4):
                boundary = seed[offset:] + seed[:offset]
                boundary_ids = ids[offset:] + ids[:offset]
                for order in permutations(range(3)):
                    permuted = tuple(triangle[i] for i in order)
                    result = certify_contact(boundary, boundary, boundary_ids, permuted, permuted,
                                             tuple(triangle_ids[i] for i in order))
                    self.assert_pass(result, 'SOURCE_HALFPLANE_ZERO_FEATURE_SPATIAL', ['a'])

    def test_oblique_direction_selection_is_permutation_independent(self):
        triangle = ((F(1, 2), F(5, 4), 0), (-F(1, 2), -6, 1), (F(3, 2), -6, 1))
        for order in permutations(range(3)):
            permuted = tuple(triangle[i] for i in order)
            result = static(permuted, center_start=CENTER, center_end=CENTER)
            self.assert_pass(result, 'STRICT_BOUNDED_OBLIQUE_CONVEX_HULL')
            self.assertEqual(result['separation_certificate']['normal'], [0, 1, 8])

    def test_affine_translation_of_zero_feature(self):
        start = (BOUNDARY[0], (F(1, 2), 0, -1), (F(1, 2), -1, 0))
        def transform(point):
            return (point[0]+4, 2*point[1]-3, point[2]+7)
        end_boundary = tuple(transform(p) for p in BOUNDARY)
        end = tuple(transform(p) for p in start)
        result = certify_contact(BOUNDARY, end_boundary, VIDS, start, end, ('a', 'p', 'q'))
        self.assert_pass(result, 'SOURCE_HALFPLANE_ZERO_FEATURE_SPATIAL', ['a'])

    def test_invalid_budgets_and_exact_arithmetic_limit(self):
        triangle = ((F(1, 2), 0, -1), (F(1, 4), -1, 0), (F(3, 4), -1, 0))
        for limit in (0, 9, True, F(1)):
            self.assertEqual(static(triangle, max_direction_weight=limit)['status'], 'REJECT')
        enormous = ((F(1, 2), 0, -F(1, 2**512)), *triangle[1:])
        self.assertEqual(static(enormous, max_rational_bits=128)['status'], 'UNKNOWN')

    def test_old_pass_is_preserved_without_runtime_claim(self):
        triangle = ((0, -2, 0), (1, -2, 0), (F(1, 2), -1, 0))
        result = static(triangle)
        self.assert_pass(result, 'STRICT_SOURCE_EDGE_EXTERIOR', [])
        self.assertEqual(result['v2_status'], 'PASS')
        self.assertFalse(result['patch_geometry_checked_here'])
        self.assertIn('binary32', ' '.join(result['not_certified']))
        self.assertIn('runtime rounding', ' '.join(result['not_certified']))


if __name__ == '__main__':
    unittest.main()
