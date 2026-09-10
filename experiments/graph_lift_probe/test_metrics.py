"""Development tests only; these do not execute the registered campaign."""

import unittest

import numpy as np

try:
    from .metrics import TriangleSurface, pl_lift, weighted_stats
except ImportError:
    from metrics import TriangleSurface, pl_lift, weighted_stats


def scalar_closest_point(point, triangle):
    """Independent seven-Voronoi-region closest-point implementation."""
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + d1 / (d1 - d3) * ab
    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + fraction * (c - b)
    denominator = 1.0 / (va + vb + vc)
    return a + (vb * denominator) * ab + (vc * denominator) * ac


class TriangleMetricTests(unittest.TestCase):
    def test_analytic_triangle(self):
        tri = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        surface = TriangleSurface(tri)
        points = np.array([[.25, .25, 2.], [-1., 0., 0.],
                           [1., 1., 0.], [.2, .2, 0.]])
        distances, faces = surface.nearest(points)
        np.testing.assert_allclose(distances, [2., 1., 1 / np.sqrt(2), 0.],
                                   rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(faces, 0)
        np.testing.assert_allclose(surface.normals, [[0., 0., 1.]])

    def test_random_matches_independent_brute_force(self):
        rng = np.random.default_rng(731)
        triangles = rng.normal(size=(47, 3, 3))
        points = rng.normal(size=(137, 3)) * 2.3
        distances, faces = TriangleSurface(triangles).nearest(points)
        brute = np.array([[np.linalg.norm(point - scalar_closest_point(point, tri))
                           for tri in triangles] for point in points])
        np.testing.assert_allclose(distances, brute.min(axis=1),
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(brute[np.arange(len(points)), faces],
                                   distances, rtol=2e-12, atol=2e-12)

    def test_long_triangle_not_nearest_centroid_is_not_pruned(self):
        triangles = np.array([
            [[-100., 0., 0.], [100., 0., 0.], [0., .01, 0.]],
            [[-99.1, -.1, .2], [-98.9, -.1, .2], [-99., .1, .2]],
        ])
        distances, faces = TriangleSurface(triangles).nearest(
            np.array([[-99., .00001, 0.]]))
        self.assertLess(distances[0], 1e-13)
        self.assertEqual(faces[0], 0)

    def test_shared_edge_tie_is_deterministic(self):
        triangles = np.array([
            [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]],
            [[0., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        ])
        distances, faces = TriangleSurface(triangles).nearest(
            np.array([[.5, .5, 1.]]))
        np.testing.assert_allclose(distances, [1.])
        np.testing.assert_array_equal(faces, [0])

    def test_pl_subdivision_preserves_physical_samples_and_area(self):
        triangles = np.array([
            [[0., 0., 0.], [1., 0., .2], [1., 1., 1.]],
            [[0., 0., 0.], [1., 1., 1.], [0., 1., -.3]],
        ])
        subdivided = []
        for triangle in triangles:
            center = triangle.mean(axis=0)
            for edge in range(3):
                subdivided.append([triangle[edge], triangle[(edge + 1) % 3], center])
        subdivided = np.array(subdivided)
        rng = np.random.default_rng(829)
        uv = np.concatenate((rng.random((513, 2)),
                             [[0., 0.], [1., 1.], [.5, .5]]))
        original = pl_lift(uv, triangles)
        refined = pl_lift(uv, subdivided)
        for before, after in zip(original, refined):
            np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)
        offset_points = original[0] + np.array([0., 0., .13])
        before = TriangleSurface(triangles).nearest(offset_points)[0]
        after = TriangleSurface(subdivided).nearest(offset_points)[0]
        np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)
        area_before = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                              triangles[:, 2] - triangles[:, 0]), axis=1).sum()
        area_after = np.linalg.norm(np.cross(subdivided[:, 1] - subdivided[:, 0],
                                             subdivided[:, 2] - subdivided[:, 0]), axis=1).sum()
        self.assertAlmostEqual(area_before, area_after, places=12)

    def test_weighted_statistics(self):
        stats = weighted_stats(np.array([0., 2., 1000.]), np.array([.96, .04, 0.]))
        self.assertAlmostEqual(stats["mean"], .08)
        self.assertEqual(stats["p95"], 0.)
        self.assertEqual(stats["sampled_max"], 2.)
        stats_scaled = weighted_stats(np.array([0., 2., 1000.]),
                                      np.array([96., 4., 0.]))
        self.assertEqual(stats, stats_scaled)

    def test_refuse_invalid_inputs(self):
        valid = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        with self.assertRaises(ValueError):
            TriangleSurface(np.zeros((1, 3, 3)))
        bad = valid.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            TriangleSurface(bad)
        with self.assertRaises(ValueError):
            TriangleSurface(valid).nearest([[np.inf, 0., 0.]])
        with self.assertRaises(ValueError):
            pl_lift([[2., 2.]], valid)
        with self.assertRaises(ValueError):
            pl_lift([[.1, .1]], valid[:, ::-1])
        with self.assertRaises(ValueError):
            weighted_stats([1.], [0.])
        with self.assertRaises(ValueError):
            weighted_stats([1.], [-1.])
        with self.assertRaises(ValueError):
            weighted_stats([np.nan], [1.])

    def test_owned_read_only_triangle_copy_and_empty_queries(self):
        triangles = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        surface = TriangleSurface(triangles)
        triangles[:] = 9.
        self.assertEqual(surface.triangles[0, 0, 0], 0.)
        self.assertFalse(surface.triangles.flags.writeable)
        distances, faces = surface.nearest(np.empty((0, 3)))
        self.assertEqual(distances.shape, (0,))
        self.assertEqual(faces.shape, (0,))


if __name__ == "__main__":
    unittest.main()
