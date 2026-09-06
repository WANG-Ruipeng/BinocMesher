import unittest
import numpy as np
from run_geometry_gate import GateStop, error_stats, heights_at_xy, improvement_gate


class GeometryGateTests(unittest.TestCase):
    def test_plane_and_outside(self):
        triangles = np.array([[[0.,0.,0.], [1.,0.,1.], [0.,1.,2.]]])
        heights = heights_at_xy(triangles, np.array([[.2,.3], [2.,2.]]), 1e-9)
        self.assertAlmostEqual(heights[0], .8)
        self.assertTrue(np.isnan(heights[1]))

    def test_fold_rejected(self):
        triangle = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
        with self.assertRaises(GateStop):
            heights_at_xy(np.array([triangle, triangle+[0.,0.,1.]]), np.array([[.2,.3]]), 1e-9)

    def test_degenerate_projection_rejected(self):
        with self.assertRaises(GateStop):
            heights_at_xy(np.array([[[0.,0.,0.],[1.,0.,0.],[1.,0.,1.]]]), np.array([[.2,0.]]), 1e-9)

    def test_no_gain_and_regression_rejected(self):
        baseline = error_stats(np.array([1.,2.]))
        self.assertFalse(improvement_gate(baseline, baseline, 1e-9))
        self.assertFalse(improvement_gate(baseline, error_stats(np.array([.1,2.1])), 1e-9))
        self.assertTrue(improvement_gate(baseline, error_stats(np.array([.5,1.])), 1e-9))


if __name__ == '__main__':
    unittest.main()
