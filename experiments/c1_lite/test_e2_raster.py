import unittest
import numpy as np

from e2_raster import rasterize


class RasterTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.eye(4)
        self.K = np.array([[2., 0, 2], [0, 2, 2], [0, 0, 1]])
        self.v = np.array([[-1., -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]])
        self.f = np.array([[0, 1, 2], [0, 2, 3]], np.int32)

    def draw(self, v=None, f=None, **kwargs):
        return rasterize(self.v if v is None else v, self.f if f is None else f,
                         kwargs.pop('c2w', self.pose), kwargs.pop('K', self.K),
                         width=kwargs.pop('width', 4), height=kwargs.pop('height', 4), **kwargs)

    def test_analytic_constant_depth_plane(self):
        result = self.draw()
        expected = np.zeros((4, 4), bool); expected[1:3, 1:3] = True
        self.assertTrue(np.array_equal(result['mask'], expected))
        np.testing.assert_array_equal(result['depth'][expected], 2.)
        np.testing.assert_array_equal(result['normals'][expected], np.tile([0., 0., 1.], (4, 1)))
        self.assertEqual(result['stats']['covered_pixels'], 4)

    def test_background_contract(self):
        r = self.draw(); outside = ~r['mask']
        self.assertTrue(np.isnan(r['depth'][outside]).all())
        self.assertTrue(np.isnan(r['normals'][outside]).all())
        self.assertTrue((r['face_id'][outside] == -1).all())
        self.assertEqual(r['depth'].dtype, np.float64)
        self.assertEqual(r['normals'].shape, (4, 4, 3))

    def test_perspective_depth_not_linear_z(self):
        v = np.array([[0., 0, 1], [8, 0, 2], [0, 16, 4]])
        r = self.draw(v, np.array([[0, 1, 2]]), K=np.eye(3))
        self.assertTrue(r['mask'][0, 0])
        self.assertAlmostEqual(r['depth'][0, 0], 1/(.75/1+.125/2+.125/4), places=14)
        self.assertNotAlmostEqual(r['depth'][0, 0], .75*1+.125*2+.125*4)

    def test_no_backface_culling_and_world_winding_preserved(self):
        a, b = self.draw(), self.draw(f=self.f[:, ::-1])
        np.testing.assert_array_equal(a['mask'], b['mask'])
        np.testing.assert_allclose(a['depth'], b['depth'], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(a['normals'][a['mask']], -b['normals'][b['mask']])

    def test_pose_transform_but_normals_remain_world_space(self):
        pose = np.array([[0., 0, 1, 3], [0, 1, 0, 4], [-1, 0, 0, 5], [0, 0, 0, 1]])
        world = self.v @ pose[:3, :3].T+pose[:3, 3]
        r = self.draw(world, c2w=pose)
        np.testing.assert_allclose(r['depth'][r['mask']], 2., atol=0, rtol=0)
        np.testing.assert_array_equal(r['normals'][r['mask']], np.tile([1., 0, 0], (4, 1)))

    def test_near_clipping_matches_explicit_clipped_quad(self):
        v = np.array([[-1., -1, .5], [2, -1, 2], [-1, 2, 2]])
        r = self.draw(v, np.array([[0, 1, 2]]), near=1)
        clipped = np.array([[-1., 0, 1], [0, -1, 1], [2, -1, 2], [-1, 2, 2]])
        expected = self.draw(clipped, np.array([[0, 1, 2], [0, 2, 3]]), near=1)
        np.testing.assert_array_equal(r['mask'], expected['mask'])
        np.testing.assert_allclose(r['depth'], expected['depth'], rtol=1e-14, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(r['normals'], expected['normals'], atol=1e-14, equal_nan=True)
        self.assertEqual(r['stats']['near_clipped_faces'], 1)
        self.assertTrue((r['face_id'][r['mask']] == 0).all())
        self.assertTrue((r['depth'][r['mask']] >= 1-1e-14).all())

    def test_one_front_two_behind_near_clips_to_triangle(self):
        v = np.array([[-1., -1, .5], [2, -1, .5], [-1, 2, 2]])
        r = self.draw(v, np.array([[0, 1, 2]]), near=1)
        self.assertEqual(r['stats']['near_clipped_faces'], 1)
        self.assertTrue(r['mask'].any())
        self.assertEqual(r['stats']['raster_triangles'], 1)

    def test_on_near_is_included(self):
        v = self.v.copy(); v[:, 2] = 1
        r = self.draw(v, near=1)
        self.assertTrue(r['mask'].all())
        np.testing.assert_allclose(r['depth'], 1, rtol=0, atol=0)

    def test_wholly_behind_near_is_empty(self):
        v = self.v.copy(); v[:, 2] = .5
        r = self.draw(v, near=1)
        self.assertFalse(r['mask'].any())
        self.assertEqual(r['stats']['behind_near_faces'], 2)

    def test_depth_occlusion_both_input_orders(self):
        v = np.vstack((self.v*2, self.v))
        f = np.vstack((self.f, self.f+4))
        a, b = self.draw(v, f), self.draw(v, f[::-1])
        np.testing.assert_allclose(a['depth'][a['mask']], 2., rtol=0, atol=0)
        np.testing.assert_allclose(a['depth'], b['depth'], rtol=0, atol=0, equal_nan=True)
        self.assertTrue((a['face_id'][a['mask']] >= 2).all())

    def test_exact_depth_tie_first_face_wins(self):
        r = self.draw(f=np.array([[0, 1, 2], [2, 1, 0]]))
        self.assertTrue(r['mask'].any())
        self.assertTrue((r['face_id'][r['mask']] == 0).all())
        self.assertTrue((r['normals'][r['mask'], 2] == 1).all())

    def test_cyclic_rotations_and_face_permutation_same_buffers(self):
        a = self.draw(); b = self.draw(f=np.roll(self.f[::-1], 1, axis=1))
        np.testing.assert_array_equal(a['mask'], b['mask'])
        np.testing.assert_allclose(a['depth'], b['depth'], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(a['normals'], b['normals'], rtol=0, atol=0, equal_nan=True)

    def test_repeated_and_collinear_faces_skipped_not_mutated(self):
        v = np.vstack((self.v, [0, -1, 2]))
        f = np.vstack((self.f, [0, 0, 1], [0, 4, 1])); before = f.copy()
        r = self.draw(v, f)
        self.assertEqual(r['stats']['degenerate_faces'], 2)
        np.testing.assert_array_equal(f, before)
        self.assertEqual(r['stats']['input_faces'], 4)

    def test_world_triangle_edge_on_to_camera_is_not_rasterized(self):
        v = np.array([[0., 0, 1], [0, 0, 2], [0, 1, 2]])
        r = self.draw(v, np.array([[0, 1, 2]]))
        self.assertEqual(r['stats']['degenerate_faces'], 0)
        self.assertEqual(r['stats']['projected_degenerate_triangles'], 1)
        self.assertFalse(r['mask'].any())

    def test_empty_mesh(self):
        r = self.draw(np.empty((0, 3)), np.empty((0, 3), np.int32))
        self.assertFalse(r['mask'].any())
        self.assertEqual(r['stats']['input_faces'], 0)

    def test_viewport_culls_remote_faces(self):
        r = self.draw(v=self.v+np.array([100, 0, 0]))
        self.assertFalse(r['mask'].any())
        self.assertEqual(r['stats']['viewport_rejected_faces'], 2)

    def test_invalid_inputs(self):
        for kwargs in ({'near': 0}, {'near': 1e-9}, {'width': 0}, {'width': 1.5},
                       {'c2w': np.zeros((4, 4))}, {'K': np.zeros((3, 3))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.draw(**kwargs)
        for vertices, faces in ((np.zeros((3, 2)), self.f),
                                (self.v, self.f.astype(float)),
                                (self.v, np.array([[0, 1, 100]])),
                                (self.v*np.nan, self.f)):
            with self.assertRaises(ValueError):
                self.draw(vertices, faces)


if __name__ == '__main__':
    unittest.main()
