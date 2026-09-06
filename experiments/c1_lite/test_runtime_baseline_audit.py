"""Synthetic checks for coordinate diagnostics, never a SourceVID proof."""
import unittest

import numpy as np

from runtime_baseline_audit import oriented_coordinate_face, patch_match


class RuntimeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0, 0, 0], [1, 0, 0],
                                  [1, 1, 0], [0, 1, 0]], dtype=float)
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        self.patch = {
            'boundary_cycle': list('abcd'),
            'boundary_positions': dict(zip('abcd', self.vertices.tolist())),
            'source_faces': [list('abc'), list('acd')],
        }

    def test_oriented_faces_preserve_winding(self):
        points = self.vertices[[0, 1, 2]]
        self.assertEqual(oriented_coordinate_face(points),
                         oriented_coordinate_face(points[[1, 2, 0]]))
        self.assertNotEqual(oriented_coordinate_face(points),
                            oriented_coordinate_face(points[[0, 2, 1]]))

    def test_unique_boundary_and_two_faces(self):
        result = patch_match(self.vertices, self.faces, self.patch)
        self.assertTrue(result['unique_coordinate_match'])
        self.assertEqual(result['source_face_coordinate_match_counts'], [1, 1])

    def test_duplicate_coordinate_is_not_identity_proof(self):
        vertices = np.vstack((self.vertices, self.vertices[0]))
        result = patch_match(vertices, self.faces, self.patch)
        self.assertFalse(result['unique_coordinate_match'])
        self.assertEqual(result['boundary_coordinate_match_counts'], [2, 1, 1, 1])

    def test_duplicate_and_reversed_faces_visible(self):
        faces = np.array([[0, 1, 2], [0, 1, 2], [0, 3, 2]], dtype=int)
        result = patch_match(self.vertices, faces, self.patch)
        self.assertEqual(result['source_face_coordinate_match_counts'], [2, 0])


if __name__ == '__main__':
    unittest.main()
