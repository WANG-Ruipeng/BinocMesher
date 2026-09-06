#!/usr/bin/env python3
"""Pure synthetic tests; no original experiment data read or changed."""
from fractions import Fraction
import unittest

from census import reconstruct_demo_time, sample_hits, effective_interval
from processed_mesh import Hypervertex


class CensusTests(unittest.TestCase):
    def test_demo_physical_mapping_not_tau_over_24(self):
        mapping = reconstruct_demo_time({'profile': 'demo', 'cameras': 24,
                                        'fading_time': 1/24}, 16, 32)
        self.assertAlmostEqual(mapping['origin_seconds_float'], 1/48)
        self.assertAlmostEqual(mapping['delta_seconds_float'], (23/24+1e-5)/32)
        self.assertNotAlmostEqual(mapping['delta_seconds_float'], 1/24)

    def test_cache_shape_disagreement_rejected(self):
        with self.assertRaises(ValueError):
            reconstruct_demo_time({'profile': 'demo', 'cameras': 24,
                                   'fading_time': 1/24}, 32, 64)

    def test_closed_open_and_root_counts(self):
        result = sample_hits(Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 4)
        self.assertEqual(result['closed_window_count'], 3)
        self.assertEqual(result['open_window_count'], 1)
        self.assertEqual(result['hits'][1]['location'], 'exact_root')

    def test_predeclared_phase_can_miss_narrow_window(self):
        zero = sample_hits(Fraction(49, 100), Fraction(1, 2), Fraction(51, 100), 24)
        half = sample_hits(Fraction(49, 100), Fraction(1, 2), Fraction(51, 100), 24, Fraction(1, 2))
        self.assertEqual(zero['closed_window_count'], 1)
        self.assertEqual(half['closed_window_count'], 0)

    def test_effective_clamp_and_reversed_input(self):
        first = Hypervertex((0, 0, 0), 0, 2, 0)
        second = Hypervertex((1, 0, 0), 10, 3, 1)
        self.assertEqual(effective_interval(first, second), (0, 7))
        self.assertEqual(effective_interval(second, first), (0, 7))

    def test_no_visibility_difference_no_clamp(self):
        first = Hypervertex((0, 0, 0), 0, 2, 1)
        second = Hypervertex((1, 0, 0), 10, 3, 1)
        self.assertEqual(effective_interval(first, second), (0, 10))


if __name__ == '__main__':
    unittest.main()
