"""Development integration tests on analytic fixtures, not campaign results."""
import unittest
import numpy as np

from model import GraphModel, DOMAINS, all_methods, quadrature, reference_triangles
from metrics import TriangleSurface, pl_lift
from run_probe import measure, metric_guard, classify


class ProbeIntegrationTests(unittest.TestCase):
    def test_analytic_fixture_full_metric_and_pure_control(self):
        for polygon in DOMAINS.values():
            for height in (np.array([0., .2, .3, .1]), np.array([0., 0., .8, 0.])):
                model = GraphModel(-np.ones(4), np.ones(4), height, height)
                uv, weights = quadrature(polygon, 4)
                analytic = model.evaluate(uv)
                reference = TriangleSurface(reference_triangles(model, polygon, 4))
                methods = all_methods(model, polygon)
                records = {method['name']: measure(method, model, uv, weights, analytic, reference)
                           for method in methods}
                family = 'affine' if height[1] != 0 else 'bilinear'
                metric_guard(records, family, 2.)
                base = pl_lift(uv, methods[0]['triangles'])
                for method in methods:
                    if method['name'].startswith('pure_pl'):
                        lifted = pl_lift(uv, method['triangles'])
                        for a, b in zip(base, lifted):
                            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_near_zero_relative_and_unresolved_comparison(self):
        def record(values):
            return {'metrics': [{'symmetric_mean_distance': x} for x in values]}
        result = classify(record([0., 0., 0.]), record([1e-15, 1e-15, 1e-15]), 1.)
        self.assertIsNone(result['relative_reduction_percent'])
        self.assertEqual(result['status'], 'NUMERIC_TIE')
        result = classify(record([.1, .11, .12]), record([.2, .18, .17]), 1.)
        self.assertEqual(result['status'], 'RESOLUTION_UNRESOLVED')


if __name__ == '__main__':
    unittest.main()
