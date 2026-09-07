import unittest
from audit_forest_visibility_projection import inverse, projector, project_support, source_geometry, classify, aggregate


def camera():
    return {'intrinsics_role': 'UNRELAXED_IMAGE_CAMERA', 'poses': [[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]],
        'image_intrinsics': [[[1.,0.,1.],[0.,1.,1.],[0.,0.,1.]]], 'widths': [2], 'heights': [2], 'near': .1, 'far': 10.}


def event(points):
    return {'boundary_actual_ids': [0,1,2], 'boundary_coordinates': points, 'source_triangles': [[0,1,2]]}


class ProjectionTests(unittest.TestCase):
    def test_all_six_planes(self):
        for plane, center in [('near',(0,0,.01)),('far',(0,0,11)),('left',(-3,0,1)),
                              ('right',(3,0,1)),('top',(0,-3,1)),('bottom',(0,3,1))]:
            x,y,z = center
            p = project_support(event([(x,y,z),(x+.001,y,z),(x,y+.001,z)]), projector(camera(),0))
            self.assertIn(plane,p['same_plane_exclusion'])

    def test_inside_not_excluded(self):
        p = project_support(event([(0,0,1),(.1,0,1),(0,.1,1)]),projector(camera(),0))
        self.assertEqual(p['same_plane_exclusion'],[])

    def test_strict_boundary_touch_not_excluded(self):
        p = project_support(event([(-1,0,1),(-1,.1,1),(-1,-.1,1)]),projector(camera(),0))
        self.assertEqual(p['same_plane_exclusion'],[])

    def test_union_straddles_without_common_plane(self):
        p = project_support(event([(-3,0,1),(3,0,1),(0,3,1)]),projector(camera(),0))
        self.assertEqual(p['same_plane_exclusion'],[])

    def test_behind_does_not_divide_negative_z_to_bbox(self):
        p = project_support(event([(0,0,-1),(.1,0,-1),(0,.1,-1)]),projector(camera(),0))
        self.assertIn('near',p['same_plane_exclusion']); self.assertIsNone(p['positive_z_vertex_pixel_bounds'])

    def test_translated_camera_world_points(self):
        c=camera(); c['poses'][0][0][3]=10.
        p=project_support(event([(10,0,1),(10.1,0,1),(10,.1,1)]),projector(c,0))
        self.assertEqual(p['same_plane_exclusion'],[])

    def test_inverse_singular_rejected(self):
        with self.assertRaises(ValueError): inverse([[0]*4 for _ in range(4)])

    def test_source_ghost_id(self):
        e=event([(0,0,1)]*3);e['source_triangles']=[[0,1,99]]
        with self.assertRaises(ValueError): source_geometry(e)

    def test_source_duplicate_id(self):
        e=event([(0,0,1)]*3);e['boundary_actual_ids']=[0,1,1]
        with self.assertRaises(ValueError): source_geometry(e)

    def test_saved_raster_contradiction_rejected(self):
        with self.assertRaises(ValueError): classify({'same_plane_exclusion':['left']},{'visible_baseline_source_pixels':0,'unoccluded_source_pixels':1})

    def test_occlusion_retains_tie_qualification(self):
        self.assertEqual(classify({'same_plane_exclusion':[]},{'visible_baseline_source_pixels':0,'unoccluded_source_pixels':1}),
                         'SAVED_OCCLUDED_OR_DEPTH_TIE_NOT_SELECTED')

    def test_zero_denominator_is_null(self):
        self.assertIsNone(aggregate([])['visible_admission_rate'])


if __name__ == '__main__': unittest.main()
