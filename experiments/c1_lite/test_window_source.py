#!/usr/bin/env python3
"""Synthetic branch-certificate tests. No real campaign/cache is accessed."""
from fractions import Fraction as F
import unittest

from window_source import (HVID, Hypervertex, SourceVID, Record, Inventory,
    SourceStop, source_formula, source_thresholds, effective_source_id,
    evaluate, partition_points, topology_at, unique_support, canonical_face,
    iter_ordinary_triangles)


class WindowSourceTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = HVID(0, 0), HVID(1, 0)
        self.edge = SourceVID.canonical(self.a, self.b)
        self.hv = {self.a: Hypervertex((0., 2., 4.), 0, 2, 0),
                   self.b: Hypervertex((10., 12., 14.), 10, 3, 1)}

    def test_affine_effective_interval_is_exact(self):
        self.assertEqual(source_thresholds(self.edge, self.hv), (0, 10, 0, 7))
        target, coefficients, _, branch = source_formula(self.edge, F(2), self.hv)
        self.assertEqual(target, self.edge)
        self.assertEqual(branch, 'affine')
        self.assertEqual(evaluate(coefficients, F(2)), (F(20, 7), F(34, 7), F(48, 7)))
        self.assertEqual(evaluate(coefficients, F(7)), (F(10), F(12), F(14)))

    def test_clamp_changes_identity_exactly_at_threshold(self):
        self.assertEqual(effective_source_id(self.edge, F(7), self.hv),
                         SourceVID.canonical(self.b, self.b))
        target, coeff, visible, _ = source_formula(self.edge, F(9), self.hv)
        self.assertEqual(target, SourceVID.canonical(self.b, self.b))
        self.assertEqual(coeff[1], (F(0),)*3)
        self.assertTrue(visible)

    def test_time_reversed_hvid_preserves_formula(self):
        reversed_hv = {self.a: self.hv[self.b], self.b: self.hv[self.a]}
        _, coeff1, _, _ = source_formula(self.edge, F(3), self.hv)
        _, coeff2, _, _ = source_formula(self.edge, F(3), reversed_hv)
        self.assertEqual(coeff1, coeff2)

    def test_zero_effective_span_catches_jump(self):
        hv = {self.a: Hypervertex((0, 0, 0), 0, 0, 0),
              self.b: Hypervertex((1, 0, 0), 1, 1, 1)}
        _, left, _, _ = source_formula(self.edge, F(-1), hv)
        _, right, _, _ = source_formula(self.edge, F(0), hv)
        # Raw clamp reaches e0=0, so this precise visibility arrangement
        # always chooses the right point, including before the raw interval.
        self.assertEqual(evaluate(left, F(-1)), (F(1), F(0), F(0)))
        self.assertEqual(left, right)

    def test_source_thresholds_are_added_not_just_probe_times(self):
        record = Record(0, 4, 0, 0, (2, 6, 10), ((), ()))
        inventory = Inventory((record,), self.hv, 8, 16, (self.edge,), '', (), 0)
        points = dict(partition_points(inventory, F(1), F(5), F(9)))
        self.assertIn(F(3), points)  # t_start-1 load threshold
        self.assertIn(F(6), points)  # processed interval and cache group
        self.assertIn(F(7), points)  # effective clamp, even absent from owner times
        self.assertIn('effective_clamp_SourceVID_visibility', points[F(7)])

    def test_constant_effective_id_per_partition_cell(self):
        inventory = Inventory((), self.hv, 8, 16, (self.edge,), '', (), 0)
        points = [t for t, _ in partition_points(inventory, F(0), F(5), F(10))]
        for lower, upper in zip(points, points[1:]):
            states = [effective_source_id(self.edge, lower+(upper-lower)*weight, self.hv)
                      for weight in (F(1, 101), F(1, 2), F(100, 101))]
            self.assertEqual(states, [states[0]]*3)

    def test_singleton_expanded_interval_and_raw_owner_identity(self):
        hv = {HVID(i, 0): Hypervertex((i, i*i, 0), 0, 0, 1) for i in range(3)}
        edges = tuple(SourceVID.canonical(key, key) for key in hv)
        record = Record(0, 1, 27, 2, (0, 2, 4), ((edges,), (edges,)))
        inv = Inventory((record,), hv, 4, 8, edges, '', (), 0)
        before = topology_at(inv, F(3, 2), {record.identity})[0]
        exact = topology_at(inv, F(2), {record.identity})[0]
        before_owner = next(iter(before.values()))[0][0]
        exact_owner = next(iter(exact.values()))[0][0]
        self.assertEqual(before_owner, (2, 0, 1, 27, 0, 0, 0))
        self.assertEqual(exact_owner, (2, 0, 1, 27, 1, 0, 0))

    def test_ambiguous_combinatorial_candidate_is_not_admitted(self):
        vertices = tuple(SourceVID.canonical(HVID(i, 0), HVID(i, 0)) for i in range(4))
        faces = ((vertices[0], vertices[1], vertices[2]),
                 (vertices[0], vertices[2], vertices[3]),
                 (vertices[0], vertices[1], vertices[3]),
                 (vertices[1], vertices[2], vertices[3]))
        groups = {(0, canonical_face(face)): [] for face in faces}
        # Fabricated all-incidence-two input isolates ambiguity rejection;
        # real topology_at would also reject these overlapping diagonals.
        incidence = {(0, *sorted((vertices[a], vertices[b]))): 2
                     for a in range(4) for b in range(a+1, 4)}
        with self.assertRaisesRegex(SourceStop, 'found 2'):
            unique_support(groups, set(groups), incidence, vertices, 0)


    def test_exterior_stream_and_partial_replica_rejection(self):
        hv = {HVID(i, 0): Hypervertex((i, i*i, 0), 0, 0, 1) for i in range(3)}
        edges = tuple(SourceVID.canonical(key, key) for key in hv)
        records = tuple(Record(0, 0, index, 0, (0, 8), ((edges,),)) for index in (5, 6))
        inv = Inventory(records, hv, 4, 8, edges, '', (), 0)
        result = list(iter_ordinary_triangles(inv, F(1, 4), F(3, 4)))
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['owners']), 2)
        self.assertEqual(result[0]['positions_t0'], result[0]['positions_t1'])
        with self.assertRaisesRegex(SourceStop, 'Partial exclusion'):
            list(iter_ordinary_triangles(inv, F(1, 4), F(3, 4), result[0]['owners'][:1]))
        self.assertEqual(list(iter_ordinary_triangles(inv, F(1, 4), F(3, 4),
                                                     result[0]['owners'])), [])

    def test_exterior_rejects_hidden_threshold_but_accepts_singleton(self):
        hv = {HVID(i, 0): Hypervertex((i, i*i, 0), 0, 0, 1) for i in range(3)}
        edges = tuple(SourceVID.canonical(key, key) for key in hv)
        record = Record(0, 0, 1, 0, (0, 8), ((edges,),))
        inv = Inventory((record,), hv, 4, 8, edges, '', (), 0)
        with self.assertRaisesRegex(SourceStop, 'midpoint is a source threshold'):
            list(iter_ordinary_triangles(inv, F(1), F(3)))
        point = list(iter_ordinary_triangles(inv, F(2), F(2)))
        self.assertEqual(len(point), 1)
        self.assertEqual(point[0]['positions_t0'], point[0]['positions_t1'])


if __name__ == '__main__':
    unittest.main()
