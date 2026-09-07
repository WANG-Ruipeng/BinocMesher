from copy import deepcopy
import unittest
import numpy as np

from forest_raster import render_scene, forest_image_camera_document


def camera(width=16, height=12, near=1e-6, far=10.):
    return {'intrinsics_role': 'UNRELAXED_IMAGE_CAMERA', 'poses': [np.eye(4).tolist()],
            'image_intrinsics': [[[8., 0., width/2], [0., 8., height/2], [0., 0., 1.]]],
            'widths': [width], 'heights': [height], 'near': near, 'far': far}


def plane(z=2.):
    return (np.asarray([[-10., -10., z], [10., -10., z], [10., 10., z], [-10., 10., z]]),
            np.asarray([[0, 1, 2], [0, 2, 3]], np.int32))


class ForestRasterTests(unittest.TestCase):
    def test_01_analytic_plane(self):
        result = render_scene([plane()], camera(), 0)
        self.assertTrue(result['mask'].all())
        np.testing.assert_allclose(result['depth'], 2., rtol=0, atol=1e-14)
        np.testing.assert_array_equal(result['normals'], np.broadcast_to([0., 0., 1.], (12, 16, 3)))

    def test_02_perspective_correct_camera_z(self):
        doc = camera(10, 10)
        doc['image_intrinsics'] = [np.eye(3).tolist()]
        vertices = np.asarray([[0., 0., 1.], [16., 0., 2.], [0., 32., 4.]])
        result = render_scene([(vertices, np.asarray([[0, 1, 2]]))], doc, 0)
        expected = 1/(.625/1+.1875/2+.1875/4)
        self.assertAlmostEqual(result['depth'][1, 1], expected, places=14)

    def test_03_near_plane_clipping(self):
        vertices = np.asarray([[-2., -2., .5], [3., -2., 2.], [0., 3., 2.]])
        result = render_scene([(vertices, np.asarray([[0, 1, 2]]))], camera(near=1.), 0)
        self.assertTrue(result['mask'].any())
        self.assertEqual(result['stats']['near_clipped_faces'], 1)
        self.assertGreaterEqual(np.nanmin(result['depth']), 1.-1e-14)
        self.assertTrue((result['face_id'][result['mask']] == 0).all())

    def test_04_far_plane_clipping(self):
        vertices = np.asarray([[-2., -2., 2.], [3., -2., 2.], [0., 3., 6.]])
        result = render_scene([(vertices, np.asarray([[0, 1, 2]]))], camera(far=3.), 0)
        self.assertTrue(result['mask'].any())
        self.assertEqual(result['stats']['far_clipped_faces'], 1)
        self.assertLessEqual(np.nanmax(result['depth']), 3.+1e-14)

    def test_05_empty_and_both_clipped_away(self):
        for meshes in ([(np.empty((0, 3)), np.empty((0, 3), np.int32))], [plane(.01)], [plane(20.)]):
            result = render_scene(meshes, camera(near=.1), 0)
            self.assertFalse(result['mask'].any())
            self.assertTrue(np.isnan(result['depth']).all())
            self.assertTrue(np.isnan(result['normals']).all())
            self.assertTrue((result['face_id'] == -1).all())

    def test_06_cross_element_occlusion_and_labels(self):
        result = render_scene([plane(3.), plane(2.)], camera(), 0,
                              event_face_ids={0: np.asarray([7, 7]), 1: np.asarray([9, 9])})
        self.assertTrue((result['element_id'] == 1).all())
        self.assertTrue((result['label_id'] == 9).all())
        np.testing.assert_allclose(result['depth'], 2.)

    def test_07_exact_ties_preserve_first_element(self):
        result = render_scene([plane(), plane()], camera(), 0)
        self.assertTrue((result['element_id'] == 0).all())

    def test_08_no_backface_culling_or_normal_flip(self):
        v, f = plane()
        a = render_scene([(v, f)], camera(), 0)
        b = render_scene([(v, f[:, ::-1])], camera(), 0)
        np.testing.assert_array_equal(a['mask'], b['mask'])
        np.testing.assert_allclose(a['depth'], b['depth'], atol=1e-14)
        np.testing.assert_array_equal(a['normals'], -b['normals'])

    def test_09_degenerate_count_not_mesh_mutation(self):
        v = np.asarray([[0., 0., 2.], [1., 0., 2.], [2., 0., 2.]])
        f = np.asarray([[0, 1, 2]])
        result = render_scene([(v, f)], camera(), 0)
        self.assertFalse(result['mask'].any())
        self.assertEqual(result['stats']['degenerate_faces'], 1)
        np.testing.assert_array_equal(f, [[0, 1, 2]])

    def test_10_original_ids_after_viewport_rejection(self):
        v, f = plane()
        vertices = np.concatenate((v+np.asarray([100., 100., 0.]), v))
        faces = np.concatenate((f, f+4))
        result = render_scene([(vertices, faces)], camera(), 0)
        self.assertEqual(result['stats']['viewport_rejected_faces'], 2)
        self.assertTrue(np.isin(result['face_id'], [2, 3]).all())

    def test_11_tags_are_not_visibility_filters(self):
        v, f = plane()
        result = render_scene([(v, f, np.zeros(len(v), bool))], camera(), 0)
        self.assertTrue(result['mask'].all())

    def test_12_readonly_inputs_and_repeated_determinism(self):
        v, f = plane(); v.flags.writeable = False; f.flags.writeable = False
        saved_v, saved_f = v.copy(), f.copy()
        a = render_scene([(v, f)], camera(), 0)
        b = render_scene([(v, f)], camera(), 0)
        for key in ('depth', 'normals', 'face_id', 'element_id', 'label_id'):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(v, saved_v); np.testing.assert_array_equal(f, saved_f)

    def test_13_world_normals_use_world_not_camera_frame(self):
        doc = camera()
        doc['poses'][0] = [[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        result = render_scene([plane()], doc, 0)
        np.testing.assert_array_equal(result['normals'], np.broadcast_to([0., 0., 1.], (12, 16, 3)))

    def test_14_pixel_work_failure_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'work budget'):
            render_scene([plane()], camera(), 0, max_pixel_tests=5)

    def test_15_camera_role_and_bad_frame_rejected(self):
        doc = camera(); doc['intrinsics_role'] = 'RELAXED_MESHER_CAMERA'
        with self.assertRaises(ValueError): render_scene([plane()], doc, 0)
        with self.assertRaises(ValueError): render_scene([plane()], camera(), 1)
        with self.assertRaises(ValueError): render_scene([plane()], camera(), True)

    def test_16_bad_shapes_indices_and_nonfinite_rejected(self):
        v, f = plane()
        for mesh in ((v, f.astype(float)), (v, np.asarray([[0, 1, 99]])),
                     (np.asarray([[np.nan, 0., 2.]]), np.asarray([[0, 0, 0]]))):
            with self.assertRaises(ValueError): render_scene([mesh], camera(), 0)

    def test_17_label_shape_and_foreign_element_rejected(self):
        for labels in ({0: np.asarray([1])}, {1: np.asarray([1, 2])}, {0: np.asarray([1., 2.])}):
            with self.assertRaises(ValueError): render_scene([plane()], camera(), 0, labels)

    def test_18_existing_numpy_oracle_parity(self):
        from e2_raster import rasterize
        rng = np.random.default_rng(7)
        vertices = rng.uniform(-2, 2, (90, 3)); vertices[:, 2] = rng.uniform(.3, 5., 90)
        faces = np.arange(90).reshape(-1, 3)
        doc = camera()
        actual = render_scene([(vertices, faces)], doc, 0)
        reference = rasterize(vertices, faces, np.eye(4), np.asarray(doc['image_intrinsics'][0]), width=16, height=12)
        np.testing.assert_array_equal(actual['mask'], reference['mask'])
        np.testing.assert_allclose(actual['depth'], reference['depth'], rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(actual['normals'], reference['normals'], atol=1e-12, equal_nan=True)

    def test_19_unverified_forest_camera_document_rejected(self):
        doc = camera(); before = deepcopy(doc)
        with self.assertRaises(ValueError): forest_image_camera_document(doc)
        self.assertEqual(doc, before)

    def test_20_shared_edge_has_no_pixel_crack(self):
        result = render_scene([plane()], camera(), 0)
        self.assertEqual(result['stats']['covered_pixels'], 16*12)
        self.assertEqual(set(np.unique(result['face_id'])), {0, 1})


if __name__ == '__main__':
    unittest.main()
