import unittest
import numpy as np

from run_forest_visibility_triage import compare_buffers, compact_source, face_counts


def buffers(depth=1., normal=(0., 0., 1.)):
    return {'mask': np.ones((2, 2), bool), 'depth': np.full((2, 2), depth),
            'normals': np.tile(np.asarray(normal), (2, 2, 1)),
            'element_id': np.array([[0, 0], [1, -1]], np.int32),
            'face_id': np.array([[4, 4], [7, -1]], np.int64)}


class VisibilityTriageTests(unittest.TestCase):
    def test_identical_normals_do_not_gain_acos_roundoff_error(self):
        a = buffers(normal=(.5773502691896257,)*3)
        metrics = compare_buffers(a, a)
        self.assertEqual(metrics['normal_angle_degrees_max'], 0.)
        self.assertEqual(metrics['depth_changed_pixels'], 0)

    def test_changed_depth_and_normal(self):
        result = compare_buffers(buffers(), buffers(2., (0., 1., 0.)))
        self.assertEqual(result['depth_changed_pixels'], 4)
        self.assertEqual(result['depth_abs_p95'], 1.)
        self.assertEqual(result['normal_angle_degrees_p95'], 90.)

    def test_background_and_roi(self):
        a, b = buffers(), buffers()
        b['mask'][0, 0] = False; b['depth'][0, 0] = np.nan
        result = compare_buffers(a, b, np.array([[True, False], [False, False]]))
        self.assertEqual(result['common_foreground_pixels'], 0)
        self.assertEqual(result['silhouette_changed_pixels'], 1)
        self.assertIsNone(result['depth_abs_p95'])

    def test_empty_roi_not_invisible_quality_improvement(self):
        result = compare_buffers(buffers(), buffers(2.), np.zeros((2, 2), bool))
        self.assertEqual(result['roi_pixels'], 0)
        self.assertIsNone(result['depth_abs_p95'])

    def test_element_namespaced_face_counts(self):
        result = face_counts(buffers())
        self.assertEqual(result[0], {4: 2})
        self.assertEqual(result[1], {7: 1})
        self.assertEqual(result[4], {})

    def test_compact_source_preserves_oriented_triangles(self):
        v = np.arange(18).reshape(6, 3)
        f = np.array([[2, 5, 3], [2, 3, 1], [0, 4, 1]])
        cv, cf = compact_source((v, f, np.zeros(6)), [0, 1])
        np.testing.assert_array_equal(cv[cf], v[f[[0, 1]]])


if __name__ == '__main__':
    unittest.main()
