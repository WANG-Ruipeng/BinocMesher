import math
import unittest
from fractions import Fraction as F
from itertools import permutations

from forest_exact_contact import check_triangle_contact


class ExactContactTests(unittest.TestCase):
    def setUp(self):
        self.a = ((0, 0, 0), (2, 0, 0), (0, 2, 0))
        self.ia = ((0, 0), (0, 1), (0, 2))

    def check(self, b, ib=None, **kwargs):
        return check_triangle_contact(self.a, b, self.ia,
            ib if ib is not None else ((0, 3), (0, 4), (0, 5)), **kwargs)

    def witness_valid(self, report):
        self.assertEqual(report['status'], 'REJECT_POLICY_CONTACT')
        data, witness = report['input_exact'], report['witness']
        point = tuple(map(F, witness['point_exact']))
        for side in ('a', 'b'):
            weights = tuple(map(F, witness['barycentric_'+side+'_exact']))
            self.assertEqual(sum(weights), 1)
            self.assertTrue(all(x >= 0 for x in weights))
            vertices = [tuple(map(F, p)) for p in data[side]]
            self.assertEqual(point, tuple(sum(w*p[k] for w, p in zip(weights, vertices)) for k in range(3)))
        self.assertFalse(report['new_contact_relative_to_baseline_proven'])

    def test_01_strict_aabb_separation(self):
        r = self.check(((0, 0, 1), (2, 0, 1), (0, 2, 1)))
        self.assertEqual((r['status'], r['proof']), ('PASS', 'STRICT_EXACT_AABB'))

    def test_02_coplanar_disjoint_with_overlapping_aabbs(self):
        r = self.check(((2, 2, 0), (2, 1, 0), (1, 2, 0)))
        self.assertEqual((r['status'], r['relation']), ('PASS', 'DISJOINT'))
        self.assertTrue(r['exhaustive_basis_search'])

    def test_03_noncoplanar_crossing(self):
        self.witness_valid(self.check(((F(1,2), F(1,2), -1), (F(1,2), F(1,2), 1), (F(3,2), F(1,2), 0))))

    def test_04_coplanar_area_overlap(self):
        self.witness_valid(self.check(((F(1,2), F(1,2), 0), (F(3,2), F(1,2), 0), (F(1,2), F(3,2), 0))))

    def test_05_legal_shared_edge_coplanar(self):
        r = self.check(((0, 0, 0), (2, 0, 0), (0, -2, 0)), ((0, 0), (0, 1), (0, 3)))
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(r['allowed_feature'], 'edge')
        self.assertEqual(set(map(tuple, r['intersection_vertices_exact'])), {('0', '0', '0'), ('2', '0', '0')})

    def test_06_legal_shared_edge_noncoplanar(self):
        r = self.check(((0, 0, 0), (2, 0, 0), (0, 0, 2)), ((0, 0), (0, 1), (0, 3)))
        self.assertEqual(r['status'], 'PASS')

    def test_07_legal_shared_vertex(self):
        r = self.check(((0, 0, 0), (-1, 0, 0), (0, -1, 0)), ((0, 0), (0, 3), (0, 4)))
        self.assertEqual((r['status'], r['allowed_feature']), ('PASS', 'vertex'))

    def test_08_shared_vertex_does_not_authorize_overlap(self):
        self.witness_valid(self.check(((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 0), (0, 3), (0, 4))))

    def test_09_shared_edge_does_not_authorize_area_overlap(self):
        self.witness_valid(self.check(((0, 0, 0), (2, 0, 0), (F(1,2), F(1,2), 0)), ((0, 0), (0, 1), (0, 3))))

    def test_10_cross_element_coincident_geometry_not_shared(self):
        r = self.check(self.a, ((1, 0), (1, 1), (1, 2)))
        self.witness_valid(r)
        self.assertEqual(r['shared_ids'], [])

    def test_11_coincident_point_distinct_ids_forbidden(self):
        self.witness_valid(self.check(((0, 0, 0), (-1, 0, 0), (0, -1, 0))))

    def test_12_retained_line_crosses_interior(self):
        self.witness_valid(self.check(((-1, 1, 0), (1, 1, 0), (3, 1, 0))))

    def test_13_retained_point_inside(self):
        self.witness_valid(self.check(((F(1,2), F(1,2), 0),)*3))

    def test_14_degenerate_shared_point(self):
        r = self.check(((0, 0, 0),)*3, ((0, 0),)*3)
        self.assertEqual((r['status'], r['allowed_feature']), ('PASS', 'vertex'))

    def test_15_degenerate_shared_edge_with_repeated_id(self):
        r = self.check(((0, 0, 0), (2, 0, 0), (2, 0, 0)), ((0, 0), (0, 1), (0, 1)))
        self.assertEqual(r['status'], 'PASS')

    def test_16_two_degenerate_skew_lines(self):
        a = ((0, 0, 0), (2, 2, 2), (1, 1, 1))
        b = ((0, 1, 0), (2, 1, 0), (1, 1, 0))
        r = check_triangle_contact(a, b, self.ia, ((0, 3), (0, 4), (0, 5)))
        self.assertEqual((r['status'], r['relation']), ('PASS', 'DISJOINT'))

    def test_17_touching_vs_nextafter_separation(self):
        touching = ((2., 0., 0.), (3., 0., 0.), (2., -1., 0.))
        self.witness_valid(self.check(touching))
        x = math.nextafter(2., math.inf)
        self.assertEqual(self.check(((x, 0., 0.), (3., 0., 0.), (x, -1., 0.)))['status'], 'PASS')

    def test_18_permutation_and_swap_invariance(self):
        b = ((0, 0, 0), (2, 0, 0), (0, -2, 1))
        ib = ((0, 0), (0, 1), (0, 3))
        for p in permutations(range(3)):
            for q in permutations(range(3)):
                a1, b1 = tuple(self.a[i] for i in p), tuple(b[i] for i in q)
                ai, bi = tuple(self.ia[i] for i in p), tuple(ib[i] for i in q)
                self.assertEqual(check_triangle_contact(a1, b1, ai, bi)['status'], 'PASS')
                self.assertEqual(check_triangle_contact(b1, a1, bi, ai)['status'], 'PASS')

    def test_19_tiny_exact_contact_not_rounded_away(self):
        x = F(1, 2**200)
        self.witness_valid(self.check(((x, x, 0), (2*x, x, 0), (x, 2*x, 0))))

    def test_20_work_budget_never_passes_partial_search(self):
        r = self.check(((0, 0, 0), (2, 0, 0), (0, -2, 0)), ((0, 0), (0, 1), (0, 3)), max_work=20)
        self.assertEqual((r['status'], r['reason']), ('UNKNOWN_WITH_BUDGET', 'EXACT_ARITHMETIC_WORK_BUDGET'))
        self.assertFalse(r['exhaustive_basis_search'])

    def test_21_input_bit_budget(self):
        r = self.check(((F(1, 2**300), 0, 0), (3, 0, 0), (2, -1, 0)), max_rational_bits=128)
        self.assertEqual((r['status'], r['reason']), ('UNKNOWN_WITH_BUDGET', 'RATIONAL_BIT_BUDGET'))

    def test_22_inconsistent_same_identity_is_input_error(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            self.check(((1, 0, 0), (3, 0, 0), (0, -1, 0)), ((0, 0), (0, 3), (0, 4)))

    def test_23_invalid_shapes_ids_and_nonfinite(self):
        for b, ids, kw in (([(0, 0, 0)], None, {}),
                (((math.nan, 0, 0),)*3, None, {}),
                (self.a, (0, 1, 2), {}),
                (self.a, self.ia, {}),
                (self.a, None, {'max_work': 0}),
                (self.a, None, {'max_rational_bits': True})):
            with self.subTest(b=b, ids=ids, kw=kw), self.assertRaises(ValueError):
                self.check(b, ids, **kw)

    def test_24_deterministic_and_input_unmodified(self):
        b = [[0, 0, 0], [2, 0, 0], [0, -2, 0]]
        before = [row[:] for row in b]
        ids = ((0, 0), (0, 1), (0, 3))
        self.assertEqual(self.check(b, ids), self.check(b, ids))
        self.assertEqual(b, before)

    def test_25_intermediate_bit_budget_never_claims_disjoint(self):
        b = ((F(1, 127), F(1, 113), -1),
             (F(1, 127), F(1, 113), 1), (1, 1, 0))
        r = self.check(b, max_rational_bits=8)
        self.assertEqual((r['status'], r['reason']),
                         ('UNKNOWN_WITH_BUDGET', 'RATIONAL_BIT_BUDGET'))
        self.assertGreater(r['maximum_observed_rational_bits'], 8)
        self.assertFalse(r['exhaustive_basis_search'])
        self.witness_valid(self.check(b))


if __name__ == '__main__':
    unittest.main()
