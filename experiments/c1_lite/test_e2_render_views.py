import unittest
import numpy as np

from e2_render_views import camera_at, project, roi_rectangle, world_hits, srgb8, boundary_mask


class RenderViewTests(unittest.TestCase):
    def test_original_camera_forward_and_principal_point(self):
        camera = camera_at(15)
        pixel, depth = project(np.asarray([[0., 9.5, 3.]]), camera)
        np.testing.assert_array_equal(pixel, [[320., 180.]])
        np.testing.assert_array_equal(depth, [5.])

    def test_outside_roi_is_empty(self):
        roi, stats = roi_rectangle(np.asarray([[5., 9., 0.], [6., 10., 0.]]), camera_at(15))
        self.assertEqual(roi.sum(), 0)
        self.assertEqual(stats['roi_pixels'], 0)

    def test_unprojection_uses_pixel_center(self):
        camera = camera_at(0)
        depth = np.ones((360, 640))*5
        mask = np.zeros(depth.shape, bool); mask[180, 320] = True
        _, points = world_hits({'depth': depth, 'mask': mask}, camera, mask)
        pixels, recovered = project(points, camera)
        np.testing.assert_allclose(pixels, [[320.5, 180.5]])
        np.testing.assert_allclose(recovered, [5.])

    def test_background_and_color_transfer(self):
        np.testing.assert_array_equal(srgb8(np.asarray([0., 1.])), [0, 255])
        self.assertTrue(0 < srgb8(np.asarray(.18)) < 255)

    def test_boundary_includes_image_edge(self):
        mask = np.ones((3, 3), bool)
        self.assertEqual(boundary_mask(mask).sum(), 8)


if __name__ == '__main__':
    unittest.main()
