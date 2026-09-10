import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'graph_lift_probe'))
from model import DOMAINS, area_centroid, base_faces, validate_mesh
from connectivity import canonical_faces, enumerate_triangulations, insert_point, retriangulate_delaunay


def vertices(polygon):
    return np.column_stack((polygon, np.zeros(len(polygon))))


class ConnectivityTests(unittest.TestCase):
    def test_square_diagonal_insertion_equals_fan(self):
        boundary = vertices(DOMAINS['whole_square'])
        point = np.array([.5, .5, .31])
        faces = insert_point(boundary, base_faces(4), point)
        fan = [(i, (i+1) % 4, 4) for i in range(4)]
        self.assertEqual(canonical_faces(faces), canonical_faces(fan))
        validate_mesh(np.vstack((boundary, point)), faces, boundary)
        retriangulated, stats = retriangulate_delaunay(np.vstack((boundary, point)), faces, 4)
        self.assertEqual(canonical_faces(retriangulated), canonical_faces(fan))
        self.assertEqual(stats['oracle_queries'], 0)

    def test_asymmetric_hexagon_same_point_preserves_disk(self):
        polygon = DOMAINS['cropped_asymmetric_hexagon']
        boundary = vertices(polygon)
        point = np.r_[area_centroid(polygon), .6]
        all_vertices = np.vstack((boundary, point))
        outcomes = []
        for root in range(6):
            start = base_faces(6, root)
            saved_vertices, saved_faces = boundary.copy(), start.copy()
            inserted = insert_point(boundary, start, point)
            validate_mesh(all_vertices, inserted, boundary)
            result, stats = retriangulate_delaunay(all_vertices, inserted, 6)
            validate_mesh(all_vertices, result, boundary)
            np.testing.assert_array_equal(boundary, saved_vertices)
            np.testing.assert_array_equal(start, saved_faces)
            outcomes.append(canonical_faces(result))
        self.assertEqual(len(set(outcomes)), 1)

    def test_genuine_flip_changes_edge(self):
        boundary = vertices(np.array([[0., 0.], [2., 0.], [2., 2.], [0., 1.]]))
        before = base_faces(4)
        after, stats = retriangulate_delaunay(boundary, before, 4)
        self.assertNotEqual(canonical_faces(before), canonical_faces(after))
        self.assertEqual(stats['strict_flips'], 1)
        validate_mesh(boundary, after, boundary)

    def test_cocircular_tie_is_stable(self):
        boundary = vertices(DOMAINS['whole_square'])
        results = [retriangulate_delaunay(boundary, base_faces(4, r), 4) for r in (0, 1)]
        self.assertEqual(canonical_faces(results[0][0]), canonical_faces(results[1][0]))
        self.assertEqual(results[1][1]['cocircular_tie_flips'], 1)

    def test_interior_insertion_and_refusals(self):
        boundary = vertices(DOMAINS['whole_square'])
        faces = base_faces(4)
        point = np.array([.7, .2, .4])
        after = insert_point(boundary, faces, point)
        self.assertEqual(len(after), len(faces)+2)
        validate_mesh(np.vstack((boundary, point)), after, boundary)
        for invalid, message in (([.5, 0., 2.], 'boundary'),
                                 ([0., 0., 1.], 'duplicate'),
                                 ([1.1, .5, 1.], 'outside'),
                                 ([.3, .4, np.nan], 'invalid')):
            with self.assertRaisesRegex(ValueError, message):
                insert_point(boundary, faces, invalid)
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            insert_point(boundary, [[0, 1, 1], [0, 2, 3]], point)

    def test_enumeration_and_cap(self):
        boundary = vertices(DOMAINS['cropped_asymmetric_hexagon'])
        states, stats = enumerate_triangulations(boundary, base_faces(6), 6)
        self.assertEqual(stats['states'], 14)  # Catalan C_4.
        for faces in states:
            validate_mesh(boundary, faces, boundary)
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            enumerate_triangulations(boundary, base_faces(6), 6, max_states=2)


if __name__ == '__main__':
    unittest.main()
