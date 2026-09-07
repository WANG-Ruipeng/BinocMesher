import unittest
from unittest.mock import patch
import numpy as np

from e2_image_metrics import terrain_reference, terrain_normals, image_metrics, image_change, reference_check, angles


class ImageMetricTests(unittest.TestCase):
    def buffer(self, mask):
        mask = np.asarray(mask, bool)
        return {'mask': mask, 'depth': np.where(mask, 2., np.nan),
                'normals': np.broadcast_to([0., 0., 1.], (*mask.shape, 3)).copy()}

    def test_reference_upward_winding(self):
        with patch('e2_image_metrics.heights', side_effect=lambda xy: np.zeros(len(xy))):
            v, f = terrain_reference([0, 1, 0, 1], 4)
        self.assertEqual(len(f), 18)
        n = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
        self.assertTrue(np.all(n[:, 2] > 0))

    def test_independent_gradient_of_plane(self):
        with patch('e2_image_metrics.heights', side_effect=lambda xy: 2*xy[:, 0]+3*xy[:, 1]):
            n = terrain_normals(np.asarray([[.1, .2]]))
        np.testing.assert_allclose(n[0], np.asarray([-2, -3, 1])/np.sqrt(14), atol=1e-11)

    def test_missing_coverage_is_not_hidden(self):
        a, b = self.buffer([[1, 0]]), self.buffer([[1, 1]])
        result = image_metrics(a, b, np.ones((1, 2), bool))
        self.assertEqual(result['both_valid_pixels'], 1)
        self.assertEqual(result['missing_reference_coverage_pixels'], 1)
        self.assertEqual(result['depth']['mean'], 0)
        self.assertEqual(reference_check(a, b, np.ones((1, 2), bool))['status'], 'STOP_REFERENCE_UNRESOLVED')

    def test_empty_roi_is_not_zero_quality_error(self):
        b = self.buffer([[1]])
        result = image_metrics(b, b, np.zeros((1, 1), bool))
        self.assertIsNone(result['depth']['mean'])

    def test_identical_normals_no_arccos_noise(self):
        n = np.asarray([[.1, .2, .3]])
        np.testing.assert_array_equal(angles(n, n), [0.])

    def test_depth_change_threshold(self):
        a, b = self.buffer([[1, 1]]), self.buffer([[1, 1]])
        b['depth'][0, 0] += 1e-7
        self.assertEqual(image_change(a, b)['depth_changed_pixels'], 1)


if __name__ == '__main__':
    unittest.main()
