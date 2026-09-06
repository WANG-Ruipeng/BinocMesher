import unittest
from fractions import Fraction
import numpy as np
from check_window_boundary import segment_distance, slice_triangle


class BoundaryTests(unittest.TestCase):
    def test_segment(self):
        a, b = np.array([0.,0.,0.]), np.array([1.,0.,0.])
        self.assertEqual(segment_distance(np.array([.5,0.,0.]), a, b), 0)
        self.assertEqual(segment_distance(np.array([.5,1.,0.]), a, b), 1)
        self.assertEqual(segment_distance(np.array([2.,0.,0.]), a, b), 1)

    def test_midplane(self):
        vertices = np.array([[0.,0.,0.,0.], [1.,0.,0.,0.], [1.,1.,0.,1.]])
        points = slice_triangle(vertices, [0,1,2], [Fraction(0),Fraction(0),Fraction(1)], Fraction(1,2))
        self.assertEqual(len(points), 2)
        self.assertTrue(any(np.allclose(point, [.5,.5,0]) for point in points))
        self.assertTrue(any(np.allclose(point, [1.,.5,0]) for point in points))

    def test_planar_vs_ruled_side_wall(self):
        a0, b0 = np.array([0.,0.,0.]), np.array([1.,0.,0.])
        a1, b1 = np.array([0.,1.,0.]), np.array([1.,1.,1.])
        am, bm = (a0+a1)/2, (b0+b1)/2
        self.assertGreater(segment_distance((a0+b1)/2, am, bm), .1)
        self.assertEqual(segment_distance((am+bm)/2, am, bm), 0.)


if __name__ == '__main__':
    unittest.main()
