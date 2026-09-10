import unittest
import numpy as np

from model import (GraphModel, DOMAINS, make_cases, all_methods, area_centroid,
                   parameter_triangles, quadrature, polygon_area)


class ModelTests(unittest.TestCase):
    def test_all_constructions_are_valid(self):
        cases = make_cases()
        self.assertEqual(len(cases), 24)
        for case in cases:
            methods = all_methods(case['model'], case['polygon'])
            self.assertEqual(len(methods), 14)
            for method in methods:
                self.assertEqual(method['budget']['reference_queries_used_for_construction'], 0)

    def test_analytic_derivatives(self):
        q = np.array([[.23, .38], [.72, .61]])
        epsilon = 1e-6
        for case in make_cases():
            model = case['model']
            xyz, normal, jac = model.evaluate(q)
            self.assertTrue(np.all(jac >= 1))
            for axis in (0, 1):
                delta = np.zeros(2)
                delta[axis] = epsilon
                numeric = (model.evaluate(q+delta)[0][:, 2]-model.evaluate(q-delta)[0][:, 2])/(2*epsilon)
                np.testing.assert_allclose(numeric, -normal[:, axis]/normal[:, 2], rtol=1e-7, atol=1e-8)

    def test_temporal_plane_and_rank_deficient_chart(self):
        lo = np.array([-.2, -.5, -.6, -.1])
        hi = np.array([.7, .8, .2, .3])
        q = np.array([[.2, .3], [.7, .4]])
        temporal_plane = GraphModel(lo, hi, lo, hi)
        np.testing.assert_allclose(temporal_plane.evaluate(q)[0][:, 2], 0, atol=1e-15)
        rank_two = GraphModel(lo, hi, np.zeros(4), np.zeros(4))
        rank_two.validate()
        np.testing.assert_allclose(rank_two.evaluate(q)[2], 1)

    def test_invalid_domain_is_rejected_not_dropped(self):
        model = GraphModel(np.array([.1, -.2, -.3, -.4]), np.ones(4), np.zeros(4), np.ones(4))
        with self.assertRaisesRegex(ValueError, 'active domain'):
            model.validate()

    def test_center_definitions_and_quadrature(self):
        np.testing.assert_allclose(area_centroid(DOMAINS['whole_square']), [.5, .5])
        hexagon = DOMAINS['cropped_asymmetric_hexagon']
        self.assertGreater(np.linalg.norm(area_centroid(hexagon)-hexagon.mean(axis=0)), .001)
        for polygon in DOMAINS.values():
            for n in (1, 4, 12):
                self.assertEqual(len(parameter_triangles(polygon, n)), (len(polygon)-2)*n*n)
                q, weights = quadrature(polygon, n)
                self.assertAlmostEqual(weights.sum(), polygon_area(polygon))
                np.testing.assert_allclose(np.average(q, axis=0, weights=weights), area_centroid(polygon))


if __name__ == '__main__':
    unittest.main()
