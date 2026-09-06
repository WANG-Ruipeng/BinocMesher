#!/usr/bin/env python3
"""Synthetic tests only; no cache inventory or real rendering is performed."""

from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_exterior import certify_exterior_separation
from window_geometry import certify_segment


BOUNDARY = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
VIDS = ('a', 'b', 'c', 'd')
CENTER = (F(1, 2), F(1, 2), 0)


def static(triangle, vids, **kwargs):
    return certify_exterior_separation(BOUNDARY, BOUNDARY, VIDS,
                                       triangle, triangle, vids, **kwargs)


def moving_hidden_contact(time):
    shift, delta2, eta = F(1, 8), F(1, 64)**2, F(1, 8192)
    edge, side = (time-shift, F(1), F(0)), (-F(1), F(0), F(0))
    def add(first, second):
        return tuple(a+b for a, b in zip(first, second))
    def scale(value, vector):
        return tuple(value*x for x in vector)
    boundary = ((F(0), F(0), F(0)), edge, add(edge, side), side)
    point = add((-delta2, -(time-shift), F(0)), scale(F(1, 2), edge))
    triangle = (point, add(point, scale(eta, edge)), add(point, scale(eta, side)))
    center = scale(F(1, 2), add(edge, side))
    return boundary, triangle, center


class WindowExteriorTests(unittest.TestCase):
    def test_strict_disjoint_triangle(self):
        result = static(((0, -2, 0), (1, -2, 0), (F(1, 2), -1, 5)), ('p', 'q', 'r'))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['separation_certificate']['allowed_contact'], 'empty')
        json.dumps(result, allow_nan=False)

    def test_declared_shared_edge_allowed(self):
        result = static((BOUNDARY[1], BOUNDARY[0], (F(1, 2), -1, 3)), ('b', 'a', 'p'))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['separation_certificate']['allowed_contact'], 'shared_edge')
        self.assertEqual(result['separation_certificate']['allowed_contact_source_vids'], ['a', 'b'])

    def test_declared_shared_vertex_allowed(self):
        result = static((BOUNDARY[0], (0, -1, 0), (-1, -1, 0)), ('a', 'p', 'q'))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['separation_certificate']['allowed_contact'], 'shared_vertex')

    def test_coincident_distinct_vid_is_not_declared_contact(self):
        result = static((BOUNDARY[0], (0, -1, 0), (-1, -1, 0)), ('not_a', 'p', 'q'))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertFalse(result['patch_triangle_interior_disjoint_certified'])

    def test_shared_vid_requires_the_same_exact_trajectory(self):
        start = (BOUNDARY[0], (0, -1, 0), (-1, -1, 0))
        end = ((0, 0, F(1, 10**30)), (0, -1, 0), (-1, -1, 0))
        result = certify_exterior_separation(BOUNDARY, BOUNDARY, VIDS,
                                             start, end, ('a', 'p', 'q'))
        self.assertEqual(result['status'], 'REJECT')
        self.assertEqual(result['witness']['kind'], 'shared_identity_mismatch')
        self.assertEqual(result['witness']['endpoint'], 'end')

    def test_edge_contact_without_matching_identity_is_unknown(self):
        triangle = ((F(1, 4), 0, 0), (F(3, 4), 0, 0), (F(1, 2), -1, 0))
        self.assertEqual(static(triangle, ('p', 'q', 'r'))['status'], 'UNKNOWN')

    def test_five_samples_miss_crossing_but_exact_separator_does_not_pass(self):
        for time in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
            boundary, triangle, center = moving_hidden_contact(time)
            self.assertEqual(certify_segment(boundary, boundary, center, center)['status'], 'PASS')
            sample = certify_exterior_separation(boundary, boundary, VIDS,
                                                 triangle, triangle, ('p', 'q', 'r'))
            self.assertEqual(sample['status'], 'PASS')
        b0, t0, c0 = moving_hidden_contact(F(0))
        b1, t1, c1 = moving_hidden_contact(F(1))
        self.assertEqual(certify_segment(b0, b1, c0, c1)['status'], 'PASS')
        result = certify_exterior_separation(b0, b1, VIDS, t0, t1, ('p', 'q', 'r'),
                                             center_start=c0, center_end=c1)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['witness'])

    def test_vertical_separation_requires_and_includes_center(self):
        triangle = ((F(1, 4), F(1, 4), 1), (F(3, 4), F(1, 4), 1),
                    (F(1, 2), F(3, 4), 1))
        self.assertEqual(static(triangle, ('p', 'q', 'r'))['status'], 'UNKNOWN')
        result = static(triangle, ('p', 'q', 'r'), center_start=CENTER, center_end=CENTER)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['separation_certificate']['axis'], 'z')
        elevated_center = (F(1, 2), F(1, 2), 2)
        unsafe = static(triangle, ('p', 'q', 'r'), center_start=elevated_center, center_end=elevated_center)
        self.assertEqual(unsafe['status'], 'UNKNOWN')

    def test_vertical_motion_crossing_height_is_unknown(self):
        start = ((F(1, 4), F(1, 4), 1), (F(3, 4), F(1, 4), 1), (F(1, 2), F(3, 4), 1))
        end = tuple((x, y, -1) for x, y, _ in start)
        result = certify_exterior_separation(BOUNDARY, BOUNDARY, VIDS, start, end, ('p', 'q', 'r'),
                                             center_start=CENTER, center_end=CENTER)
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_clockwise_cycle_changes_halfplane_sign(self):
        boundary, vids = tuple(reversed(BOUNDARY)), tuple(reversed(VIDS))
        triangle = ((0, -2, 0), (1, -2, 0), (F(1, 2), -1, 0))
        result = certify_exterior_separation(boundary, boundary, vids, triangle, triangle,
                                             ('p', 'q', 'r'), orientation_xy=-1)
        self.assertEqual(result['status'], 'PASS')

    def test_budget_exhaustion_is_unknown(self):
        triangle = ((0, -1, F(1, 2**512)), (1, -1, 0), (0, -2, 0))
        result = static(triangle, ('p', 'q', 'r'), max_rational_bits=128)
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_no_claim_of_exterior_manifoldness(self):
        result = static(((0, -2, 0), (1, -2, 0), (F(1, 2), -1, 0)), ('p', 'q', 'r'))
        self.assertFalse(result['patch_geometry_checked_here'])
        self.assertIn('retained mesh', ' '.join(result['not_certified']))


if __name__ == '__main__':
    unittest.main()
