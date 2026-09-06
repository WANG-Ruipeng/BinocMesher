#!/usr/bin/env python3
"""Synthetic exact-contact tests; no real source caches or runtime rendering."""

from fractions import Fraction as F
from itertools import permutations
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_contact_v2 import certify_contact
from window_exterior import certify_exterior_separation
from window_geometry import certify_segment


BOUNDARY = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
VIDS = ('a', 'b', 'c', 'd')
CENTER = (F(1, 2), F(1, 2), 0)


def static(triangle, vids=('a', 'p', 'q'), **kwargs):
    return certify_contact(BOUNDARY, BOUNDARY, VIDS,
                           triangle, triangle, vids, **kwargs)


def hidden_contact(time):
    s, delta2, eta = time-F(1, 8), F(1, 64)**2, F(1, 8192)
    boundary = ((F(0), F(0), F(0)), (s, F(1), F(0)),
                (s-1, F(1), F(0)), (-F(1), F(0), F(0)))
    triangle = (boundary[0], (-delta2, -s, F(0)),
                (-delta2-eta*s, -s-eta, F(1)))
    center = ((s-1)/2, F(1, 2), F(0))
    return boundary, triangle, center


class WindowContactV2Tests(unittest.TestCase):
    def assert_pass(self, result, kind=None, shared=None):
        self.assertEqual(result['status'], 'PASS', result.get('reason'))
        self.assertTrue(result['patch_triangle_interior_disjoint_certified'])
        if kind is not None:
            self.assertEqual(result['separation_certificate']['kind'], kind)
        if shared is not None:
            self.assertEqual(result['separation_certificate']['allowed_contact_source_vids'], shared)
        json.dumps(result, allow_nan=False)

    def test_positive_incident_combination_needed(self):
        triangle = (BOUNDARY[0], (-2, 1, 0), (1, -2, 0))
        old = certify_exterior_separation(BOUNDARY, BOUNDARY, VIDS,
                                          triangle, triangle, ('a', 'p', 'q'))
        self.assertEqual(old['status'], 'UNKNOWN')
        result = static(triangle)
        self.assert_pass(result, 'POSITIVE_INCIDENT_HALFPLANE_COMBINATION', ['a'])
        self.assertEqual(result['separation_certificate']['positive_integer_weights'], [1, 1])
        self.assertEqual(result['retained_regularity_certificate']['status'], 'PASS')

    def test_bounded_search_is_unknown_not_invalid(self):
        triangle = (BOUNDARY[0], (-1, 7, 0), (1, -9, 0))
        short = static(triangle, max_weight=4)
        self.assertEqual(short['status'], 'UNKNOWN')
        self.assertIsNone(short['witness'])
        result = static(triangle, max_weight=8)
        self.assert_pass(result, 'POSITIVE_INCIDENT_HALFPLANE_COMBINATION')
        self.assertEqual(result['separation_certificate']['positive_integer_weights'], [8, 1])
        self.assertEqual(result['limits']['max_weight'], 8)

    def test_illegal_triangle_overlap_stays_unknown(self):
        triangle = (BOUNDARY[0], (-1, 2, 0), (2, -1, 0))
        result = static(triangle)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertFalse(result['patch_triangle_interior_disjoint_certified'])
        self.assertIsNone(result['witness'])

    def test_extra_interior_contact_is_not_whitelisted_by_shared_vertex(self):
        result = static((BOUNDARY[0], (-1, 0, 0), (F(1, 2), F(1, 2), 0)))
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_shared_supporting_line_edge(self):
        triangle = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), 0, 1))
        result = static(triangle, ('a', 'b', 'p'))
        self.assert_pass(result, 'SUPPORTING_LINE_SOURCE_EDGE', ['a', 'b'])
        self.assertEqual(result['separation_certificate']['allowed_contact'], 'shared_edge')
        self.assertEqual(result['first_tier_status'], 'UNKNOWN')

    def test_shared_edge_off_support_line_cannot_use_new_rule(self):
        result = static((BOUNDARY[0], BOUNDARY[1], (F(1, 2), F(1, 2), 0)),
                        ('a', 'b', 'p'))
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_shared_vertex_support_line_spatial_separation(self):
        triangle = (BOUNDARY[0], (F(1, 4), 0, 1), (F(3, 4), 0, 2))
        result = static(triangle)
        self.assert_pass(result, 'SUPPORTING_LINE_SPATIAL_SHARED_VERTEX', ['a'])

    def test_support_line_opposite_heights_have_extra_contact(self):
        triangle = (BOUNDARY[0], (F(1, 4), 0, 1), (F(3, 4), 0, -1))
        self.assertEqual(static(triangle)['status'], 'UNKNOWN')

    def test_support_line_extension_has_only_declared_vertex(self):
        triangle = (BOUNDARY[0], (-1, 0, 1), (-2, 0, -1))
        self.assert_pass(static(triangle), shared=['a'])

    def test_different_vid_at_same_coordinate_is_not_shared(self):
        triangle = (BOUNDARY[0], (-2, 1, 0), (1, -2, 0))
        self.assertEqual(static(triangle, ('fake_a', 'p', 'q'))['status'], 'UNKNOWN')
        edge_triangle = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), 0, 1))
        self.assertEqual(static(edge_triangle, ('fake_a', 'b', 'p'))['status'], 'UNKNOWN')

    def test_same_vid_must_have_identical_end_trajectory(self):
        triangle = (BOUNDARY[0], (-2, 1, 0), (1, -2, 0))
        end = ((0, 0, F(1, 10**30)), *triangle[1:])
        result = certify_contact(BOUNDARY, BOUNDARY, VIDS,
                                 triangle, end, ('a', 'p', 'q'))
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(result['witness']['kind'], 'shared_identity_mismatch')

    def test_distinct_ids_geometrically_degenerate_is_rejected(self):
        triangle = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), 0, 0))
        result = static(triangle, ('a', 'b', 'p'))
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(result['witness']['kind'], 'retained_triangle_degenerate')

    def test_repeated_ids_rejected(self):
        result = static((BOUNDARY[0], BOUNDARY[0], (0, -1, 0)), ('a', 'a', 'p'))
        self.assertEqual(result['status'], 'REJECT')

    def test_degenerate_contact_is_not_admitted_by_old_tier(self):
        triangle = (BOUNDARY[0], (-1, -1, 0), (-2, -2, 0))
        old = certify_exterior_separation(BOUNDARY, BOUNDARY, VIDS,
                                          triangle, triangle, ('a', 'p', 'q'))
        self.assertEqual(old['status'], 'PASS')
        self.assertEqual(static(triangle)['status'], 'REJECT')

    def test_endpoint_degeneracy_is_rejected(self):
        start = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), 0, 1))
        end = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), 0, 0))
        result = certify_contact(BOUNDARY, BOUNDARY, VIDS, start, end, ('a', 'b', 'p'))
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(result['witness']['normalized_time'], {'numerator': 1, 'denominator': 1})

    def test_internal_degeneracy_is_rejected_with_exact_time(self):
        start = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), 0, 1))
        end = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), 0, -7))
        result = certify_contact(BOUNDARY, BOUNDARY, VIDS, start, end, ('a', 'b', 'p'))
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(result['witness']['normalized_time'], {'numerator': 1, 'denominator': 8})

    def test_rotating_nonzero_normal_can_conservatively_remain_unknown(self):
        # The y and z normal components vanish at different times. The
        # triangle is never degenerate, but this bounded regularity rule
        # intentionally does not combine several normal components.
        start = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), -F(1, 4), -F(3, 4)))
        end = (BOUNDARY[0], BOUNDARY[1], (F(1, 2), F(3, 4), F(1, 4)))
        result = certify_contact(BOUNDARY, BOUNDARY, VIDS, start, end, ('a', 'b', 'p'))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['witness'])
        self.assertEqual(result['retained_regularity_certificate']['kind'],
                         'RETAINED_REGULARITY_NOT_CERTIFIED')

    def test_five_samples_miss_extra_contact_but_whole_branch_not_admitted(self):
        for time in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
            boundary, triangle, center = hidden_contact(time)
            self.assertEqual(certify_segment(boundary, boundary, center, center)['status'], 'PASS')
            self.assert_pass(certify_contact(boundary, boundary, VIDS,
                                              triangle, triangle, ('a', 'p', 'q')))
        b0, t0, c0 = hidden_contact(F(0))
        b1, t1, c1 = hidden_contact(F(1))
        self.assertEqual(certify_segment(b0, b1, c0, c1)['status'], 'PASS')
        result = certify_contact(b0, b1, VIDS, t0, t1, ('a', 'p', 'q'))
        self.assertEqual(result['retained_regularity_certificate']['status'], 'PASS')
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['witness'])

    def test_input_triangle_permutations_and_cycle_rotations(self):
        triangle = (BOUNDARY[0], (-2, 1, 0), (1, -2, 0))
        triangle_ids = ('a', 'p', 'q')
        for order in permutations(range(3)):
            for offset in range(4):
                boundary = BOUNDARY[offset:] + BOUNDARY[:offset]
                vids = VIDS[offset:] + VIDS[:offset]
                permuted = tuple(triangle[i] for i in order)
                result = certify_contact(boundary, boundary, vids, permuted, permuted,
                                         tuple(triangle_ids[i] for i in order))
                self.assert_pass(result, 'POSITIVE_INCIDENT_HALFPLANE_COMBINATION', ['a'])

    def test_clockwise_and_nonxy_height_profiles(self):
        boundary = ((0, 0, 0), (1, 0, 2), (1, 1, -5), (0, 1, 7))
        triangle = (boundary[0], boundary[1], (F(1, 2), 0, 3))
        reverse, vids = tuple(reversed(boundary)), tuple(reversed(VIDS))
        result = certify_contact(reverse, reverse, vids, triangle, triangle, ('a', 'b', 'p'),
                                 orientation_xy=-1)
        self.assert_pass(result, 'SUPPORTING_LINE_SOURCE_EDGE', ['a', 'b'])

    def test_affine_translations_and_anisotropic_motion(self):
        first = (BOUNDARY[0], (-2, 1, 0), (1, -2, 0))
        def transform(point):
            x, y, z = point
            return (2*x+4, 3*y-9, z+x-y+2)
        last_boundary = tuple(transform(point) for point in BOUNDARY)
        last = tuple(transform(point) for point in first)
        result = certify_contact(BOUNDARY, last_boundary, VIDS,
                                 first, last, ('a', 'p', 'q'))
        self.assert_pass(result, 'POSITIVE_INCIDENT_HALFPLANE_COMBINATION')

    def test_budget_and_invalid_weight_limits(self):
        triangle = (BOUNDARY[0], (-2, 1, F(1, 2**512)), (1, -2, 0))
        self.assertEqual(static(triangle, max_rational_bits=128)['status'], 'UNKNOWN')
        self.assertEqual(static(triangle, max_weight=0)['status'], 'REJECT')
        self.assertEqual(static(triangle, max_weight=33)['status'], 'REJECT')

    def test_disjoint_old_tier_preserved_and_scope_explicit(self):
        triangle = ((0, -2, 0), (1, -2, 0), (F(1, 2), -1, 0))
        result = static(triangle, ('p', 'q', 'r'))
        self.assert_pass(result, 'STRICT_SOURCE_EDGE_EXTERIOR', [])
        self.assertFalse(result['patch_geometry_checked_here'])
        self.assertIn('binary32', ' '.join(result['not_certified']))
        self.assertIn('manifold', ' '.join(result['not_certified']))


if __name__ == '__main__':
    unittest.main()
