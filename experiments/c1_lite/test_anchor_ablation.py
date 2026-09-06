#!/usr/bin/env python3
"""Synthetic-only tests for fixed-window anchor ablation."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_ablation import (GRIDS, GateStop, baseline_grid, candidate_centers,
                             evaluate_candidate, fan_triangles, fixed_times,
                             fraction_json, normalized_integral, pairwise_loss,
                             summarize_method)


class AnchorAblationTests(unittest.TestCase):
    def setUp(self):
        self.boundary = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        self.baseline = self.boundary[[[0, 1, 2], [0, 2, 3]]]
        self.levels = tuple(map(Fraction, (0, 1, 3)))
        self.endpoints = (np.asarray([.5, .5, 0]), np.asarray([.5, .5, 0]))

    def test_fixed_33_times_and_unequal_slabs(self):
        times = fixed_times(self.levels)
        self.assertEqual(len(times), 33)
        self.assertEqual(times[0], 0)
        self.assertEqual(times[16], 1)
        self.assertEqual(times[-1], 3)
        self.assertEqual(times.count(Fraction(1)), 1)
        self.assertEqual(times[1]-times[0], Fraction(1, 16))
        self.assertEqual(times[-1]-times[-2], Fraction(1, 8))

    def test_time_integral_not_sample_average(self):
        times = fixed_times(self.levels)
        values = [float(value) for value in times]
        self.assertAlmostEqual(normalized_integral(times, values), 1.5)
        self.assertNotAlmostEqual(float(np.mean(values)), 1.5)
        self.assertAlmostEqual(normalized_integral(times, [7]*33), 7)

    def test_centroid_does_not_inherit_beb1_root(self):
        center = np.asarray([.4, .6, 1.])
        values = candidate_centers(Fraction(1), self.levels, self.endpoints, self.boundary, center)
        np.testing.assert_array_equal(values['beb1'], center)
        np.testing.assert_array_equal(values['centroid'], [.5, .5, 0])
        np.testing.assert_array_equal(values['naive'], [.5, .5, 0])
        for tau, target in zip((self.levels[0], self.levels[-1]), self.endpoints):
            result = candidate_centers(tau, self.levels, self.endpoints, self.boundary, center)
            for value in result.values():
                np.testing.assert_array_equal(value, target)

    def test_negative_quality_is_retained_not_raised(self):
        baseline, shared = baseline_grid(self.boundary, self.baseline, lambda p: p[:, 2], 32)
        worse = evaluate_candidate(self.boundary, fan_triangles(self.boundary, [.5, .5, 1]), shared, 32)
        difference = pairwise_loss(worse, baseline)
        self.assertGreater(difference['mean_absolute_height_error_difference'], 0)
        self.assertIsNone(difference['mean_absolute_height_error_relative_change'])

    def test_reference_is_shared_and_coverage_mismatch_rejected(self):
        batches = []

        def terrain(points):
            batches.append(len(points))
            return points[:, 2]

        _, shared = baseline_grid(self.boundary, self.baseline, terrain, 32)
        fan = fan_triangles(self.boundary, [.5, .5, 0])
        evaluate_candidate(self.boundary, fan, shared, 32)
        evaluate_candidate(self.boundary, fan, shared, 32)
        self.assertEqual(batches, [1024])
        shifted = fan.copy()
        shifted[:, :, 0] += .1
        with self.assertRaisesRegex(GateStop, 'coverage differs'):
            evaluate_candidate(self.boundary, shifted, shared, 32)

    def test_invalid_candidate_is_not_quality_loss(self):
        _, shared = baseline_grid(self.boundary, self.baseline, lambda p: p[:, 2], 32)
        with self.assertRaises(GateStop):
            evaluate_candidate(self.boundary, fan_triangles(self.boundary, [2, 2, 0]), shared, 32)

    def records(self, times, root_error):
        return [{'time': fraction_json(tau), 'grids': [
            {'grid': grid, 'mean_absolute_height_error': root_error if tau == self.levels[1] else 1.,
             'maximum_absolute_height_error': root_error if tau == self.levels[1] else 1.,
             'p95_absolute_height_error': root_error if tau == self.levels[1] else 1.}
            for grid in GRIDS]} for tau in times]

    def test_c0_does_not_get_integrated_measure_zero_gain(self):
        times = fixed_times(self.levels)
        baseline = summarize_method(self.records(times, 1.), times, self.levels, .2, 'baseline')
        c0 = summarize_method(self.records(times, 0.), times, self.levels, .2, 'c0_root_only', baseline)
        for row in c0['grids']:
            self.assertEqual(row['normalized_mean_height_error_integral'], 1.)
            self.assertAlmostEqual(row['mean_height_error_integral_seconds'], .2)
            self.assertEqual(row['root_mean_height_error'], 0.)
            self.assertLess(row['diagnostic_naive_trapezoidal_mean'], 1.)

    def test_incomplete_curve_has_no_full_window_integral(self):
        times = fixed_times(self.levels)
        result = summarize_method(self.records(times[:-1], 1.), times, self.levels, .2, 'beb1')
        self.assertEqual(result['status'], 'INCOMPLETE_NO_WINDOW_INTEGRAL')
        self.assertNotIn('grids', result)

    def test_invalid_time_order_rejected(self):
        with self.assertRaises(GateStop):
            normalized_integral([Fraction(0), Fraction(0)], [1, 2])
        with self.assertRaises(GateStop):
            fixed_times(tuple(map(Fraction, (0, 2, 1))))


if __name__ == '__main__':
    unittest.main()
