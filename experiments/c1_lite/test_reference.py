#!/usr/bin/env python3
"""Synthetic tests only; no real cache, renderer, or production writes."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import (GateStop, center_at, endpoint_contract, fan_triangles,
                       compare_heights, point_on_segment_exact, require_convex_graph,
                       rational, stable_cycle)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.boundary = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])

    def test_center_anchors_and_naive(self):
        levels = tuple(map(Fraction, (0, 1, 2)))
        anchors = np.asarray([[0., 0., 0.], [1., 1., 1.], [2., 0., 0.]])
        for index, tau in enumerate(levels):
            np.testing.assert_array_equal(center_at(tau, levels, anchors), anchors[index])
        np.testing.assert_array_equal(center_at(Fraction(1), levels, anchors, False), [1, 0, 0])
        np.testing.assert_array_equal(center_at(Fraction(1, 2), levels, anchors), [.5, .5, .5])
        with self.assertRaises(GateStop):
            center_at(Fraction(3), levels, anchors)

    def test_exact_segment_rejects_tiny_offset(self):
        self.assertTrue(point_on_segment_exact([.5, .5, 0], [0, 0, 0], [1, 1, 0]))
        self.assertFalse(point_on_segment_exact([.5, .5, 1e-30], [0, 0, 0], [1, 1, 0]))

    def test_binary32_endpoint_failure_is_not_hidden(self):
        # Endpoints are themselves binary32, but differently rounded midpoint
        # coordinates do not in general have a common segment parameter.
        first = np.asarray([1, 1, 0], dtype=np.float32).astype(float)
        second = np.asarray([1+2**-23, 1+2**-22, 1], dtype=np.float32).astype(float)
        audit = endpoint_contract(first, second)
        self.assertTrue(audit['binary64_center_on_shared_diagonal_exact'])
        self.assertFalse(audit['binary32_center_on_shared_diagonal_exact'])
        self.assertGreater(audit['binary32_distance_to_shared_diagonal'], 0)

    def test_binary32_exact_endpoint(self):
        audit = endpoint_contract([0, 0, 0], [1, 1, 0])
        self.assertTrue(audit['binary32_center_on_shared_diagonal_exact'])

    def test_first_event_round_source_before_center_regression(self):
        first = [.8056640625, -.6591796875, .2685546875]
        second = [.37841796875, .9480794270833333, .37841796875]
        audit = endpoint_contract(first, second)
        legacy_center = np.asarray(audit['center_binary64']).astype(np.float32).astype(float)
        source = audit['source_diagonal_binary32']
        self.assertFalse(point_on_segment_exact(legacy_center, *source))
        self.assertTrue(audit['binary32_center_on_shared_diagonal_exact'])
        self.assertTrue(audit['binary64_center_on_shared_diagonal_exact'])
        np.testing.assert_array_equal(audit['center_binary32'],
                                      [.592041015625, .1444498598575592, .323486328125])

    def test_binary64_midpoint_rounding_is_separate(self):
        first = [1., 1., 0.]
        second = [1.+2**-52, 1.+2**-51, 1.]
        audit = endpoint_contract(first, second)
        self.assertFalse(audit['binary64_center_on_shared_diagonal_exact'])
        self.assertTrue(audit['binary32_center_on_shared_diagonal_exact'])

    def test_do_not_blame_source_quantization_on_new_center(self):
        audit = endpoint_contract([0, 0, 0], [1+2**-25, 1, 0])
        self.assertFalse(audit['binary32_center_on_ideal_diagonal_exact'])
        self.assertTrue(audit['binary32_center_on_shared_diagonal_exact'])
        self.assertEqual(audit['binary32_distance_to_shared_diagonal'], 0)

    def test_graph_rejects_outside_center_and_degeneracy(self):
        for center in ([2, .5, 0], [0, 0, 0]):
            with self.assertRaises(GateStop):
                require_convex_graph(self.boundary, fan_triangles(self.boundary, center))

    def test_equal_reference_and_coverage(self):
        baseline = self.boundary[[[0, 1, 2], [0, 2, 3]]]
        fan = fan_triangles(self.boundary, [.5, .5, 0])
        result = compare_heights(self.boundary, baseline, fan, fan, lambda p: p[:, 2], 32)
        self.assertEqual(result['samples'], 1024)
        self.assertEqual(result['c1']['mean_absolute_height_error'], 0)
        shifted = fan.copy()
        shifted[:, :, 0] += .1
        with self.assertRaisesRegex(GateStop, 'coverage differs'):
            compare_heights(self.boundary, baseline, shifted, fan, lambda p: p[:, 2], 32)

    def test_cycle_orientation_and_exact_time(self):
        self.assertTrue(stable_cycle(['a', 'b', 'c', 'd'], ['c', 'd', 'a', 'b']))
        self.assertFalse(stable_cycle(['a', 'b', 'c', 'd'], ['a', 'd', 'c', 'b']))
        self.assertEqual(rational('120/11'), Fraction(120, 11))
        with self.assertRaises(GateStop):
            rational(.5)


if __name__ == '__main__':
    unittest.main()
