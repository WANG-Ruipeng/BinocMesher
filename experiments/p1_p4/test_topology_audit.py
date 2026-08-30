#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from topology_audit import (
    INTERSECTION_ALGORITHM,
    audit_mesh,
    audit_mesh_pair,
    triangles_intersect,
)


EPSILON = 1.0e-9


class TopologyAuditTests(unittest.TestCase):
    def test_empty_mesh_is_a_valid_noop(self) -> None:
        vertices = np.empty((0, 3), dtype=float)
        faces = np.empty((0, 3), dtype=int)
        result = audit_mesh_pair(vertices, faces, vertices, faces, epsilon=EPSILON)
        self.assertTrue(result["pass"])

    def test_closed_tetrahedron_is_manifold(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        faces = np.asarray(
            [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)], dtype=int
        )
        result = audit_mesh_pair(vertices, faces, vertices, faces, epsilon=EPSILON)
        self.assertTrue(result["pass"])
        self.assertEqual(result["nonmanifold_edges"], 0)
        self.assertEqual(result["nonmanifold_vertices"], 0)
        self.assertEqual(result["duplicate_oriented_faces"], 0)
        self.assertEqual(result["relative_boundary_mismatches"], 0)
        self.assertEqual(result["new_nonincident_intersections"], 0)

    def test_three_faces_sharing_edge_is_nonmanifold(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            dtype=float,
        )
        faces = np.asarray([(0, 1, 2), (1, 0, 3), (0, 1, 4)], dtype=int)
        result = audit_mesh(vertices, faces, epsilon=EPSILON)
        self.assertEqual(result["nonmanifold_edges"], 1)
        self.assertGreaterEqual(result["nonmanifold_vertices"], 2)

    def test_bow_tie_vertex_fails_vertex_link(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
            dtype=float,
        )
        faces = np.asarray([(0, 1, 2), (0, 3, 4)], dtype=int)
        result = audit_mesh(vertices, faces, epsilon=EPSILON)
        self.assertEqual(result["nonmanifold_edges"], 0)
        self.assertEqual(result["nonmanifold_vertices"], 1)

    def test_same_orientation_duplicate_is_counted(self) -> None:
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.asarray([(0, 1, 2), (1, 2, 0)], dtype=int)
        result = audit_mesh(vertices, faces, epsilon=EPSILON)
        self.assertEqual(result["duplicate_oriented_faces"], 1)

    def test_retriangulation_preserves_oriented_boundary_chain(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
        )
        diagonal_02 = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=int)
        diagonal_13 = np.asarray([(0, 1, 3), (1, 2, 3)], dtype=int)
        result = audit_mesh_pair(
            vertices, diagonal_02, vertices, diagonal_13, epsilon=EPSILON
        )
        self.assertEqual(result["relative_boundary_mismatches"], 0)

    def test_changed_boundary_is_reported(self) -> None:
        baseline_vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
        )
        treatment_vertices = baseline_vertices.copy()
        treatment_vertices[2] = [1.25, 1, 0]
        faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=int)
        result = audit_mesh_pair(
            baseline_vertices, faces, treatment_vertices, faces, epsilon=EPSILON
        )
        self.assertGreater(result["relative_boundary_mismatches"], 0)

    def test_noncoplanar_crossing_is_detected(self) -> None:
        vertices = np.asarray(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [0.5, 0.25, -1],
                [0.5, 0.25, 1],
                [0.5, 1.25, 0],
            ],
            dtype=float,
        )
        baseline_faces = np.asarray([(0, 1, 2)], dtype=int)
        treatment_faces = np.asarray([(0, 1, 2), (3, 4, 5)], dtype=int)
        result = audit_mesh_pair(
            vertices, baseline_faces, vertices, treatment_faces, epsilon=EPSILON
        )
        self.assertEqual(result["new_nonincident_intersections"], 1)
        self.assertEqual(result["broadphase_candidates"], 1)
        self.assertEqual(result["narrowphase_tests"], 1)
        self.assertEqual(result["intersection_algorithm"], INTERSECTION_ALGORITHM)

    def test_coplanar_overlap_is_detected(self) -> None:
        vertices = np.asarray(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [0.25, 0.25, 0],
                [1.25, 0.25, 0],
                [0.25, 1.25, 0],
            ],
            dtype=float,
        )
        baseline_faces = np.asarray([(0, 1, 2)], dtype=int)
        treatment_faces = np.asarray([(0, 1, 2), (3, 4, 5)], dtype=int)
        result = audit_mesh_pair(
            vertices, baseline_faces, vertices, treatment_faces, epsilon=EPSILON
        )
        self.assertEqual(result["new_nonincident_intersections"], 1)

    def test_aabb_overlap_without_triangle_intersection_is_rejected_by_narrow_phase(self) -> None:
        vertices = np.asarray(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [1.1, 1.1, 0],
                [2.1, 1.1, 0],
                [1.1, 2.1, 0],
            ],
            dtype=float,
        )
        faces = np.asarray([(0, 1, 2), (3, 4, 5)], dtype=int)
        result = audit_mesh(vertices, faces, epsilon=EPSILON)
        self.assertEqual(result["broadphase_candidates"], 1)
        self.assertEqual(result["narrowphase_tests"], 1)
        self.assertEqual(result["nonincident_intersections"], 0)

    def test_incident_faces_are_excluded_from_intersection_candidates(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        faces = np.asarray([(0, 1, 2), (0, 3, 1)], dtype=int)
        result = audit_mesh(vertices, faces, epsilon=EPSILON)
        self.assertEqual(result["broadphase_candidates"], 0)
        self.assertEqual(result["narrowphase_tests"], 0)
        self.assertEqual(result["nonincident_intersections"], 0)

    def test_coplanar_degenerate_segment_crossing_triangle_is_detected(self) -> None:
        triangle = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float
        )
        # All vertices of the collinear triangle lie outside the regular
        # triangle, but its longest segment crosses the interior at y=1/4.
        degenerate = np.asarray(
            [[-1, 0.25, 0], [2, 0.25, 0], [3, 0.25, 0]], dtype=float
        )
        self.assertTrue(triangles_intersect(triangle, degenerate, EPSILON))
        self.assertTrue(triangles_intersect(degenerate, triangle, EPSILON))

    def test_coplanar_degenerate_segment_outside_triangle_does_not_intersect(self) -> None:
        triangle = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float
        )
        degenerate = np.asarray(
            [[-1, 1.25, 0], [2, 1.25, 0], [3, 1.25, 0]], dtype=float
        )
        self.assertFalse(triangles_intersect(triangle, degenerate, EPSILON))
        self.assertFalse(triangles_intersect(degenerate, triangle, EPSILON))

    def test_degenerate_crossing_reaches_existing_release_gate(self) -> None:
        vertices = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0.25, 0],
                [2, 0.25, 0],
                [3, 0.25, 0],
            ],
            dtype=float,
        )
        baseline_faces = np.asarray([(0, 1, 2)], dtype=int)
        treatment_faces = np.asarray([(0, 1, 2), (3, 4, 5)], dtype=int)
        result = audit_mesh_pair(
            vertices, baseline_faces, vertices, treatment_faces, epsilon=EPSILON
        )
        self.assertEqual(result["intersection_delta"]["new_geometric_intersections"], 1)
        self.assertEqual(result["degenerate_faces"], 1)
        self.assertEqual(result["new_nonincident_intersections"], 2)
        self.assertFalse(result["pass"])

    def test_isolated_degenerate_face_fails_closed(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float
        )
        faces = np.asarray([(0, 1, 2)], dtype=int)
        result = audit_mesh_pair(
            vertices, faces, vertices, faces, epsilon=EPSILON
        )
        self.assertEqual(result["intersection_delta"]["new_geometric_intersections"], 0)
        self.assertEqual(result["degenerate_faces"], 1)
        self.assertEqual(result["new_nonincident_intersections"], 1)
        self.assertFalse(result["pass"])

    def test_repeated_vertex_index_face_is_counted_not_dropped(self) -> None:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float
        )
        faces = np.asarray([(0, 1, 1)], dtype=int)
        result = audit_mesh_pair(
            vertices, faces, vertices, faces, epsilon=EPSILON
        )
        self.assertEqual(result["treatment"]["repeated_vertex_index_faces"], 1)
        self.assertEqual(result["degenerate_faces"], 1)
        self.assertEqual(result["new_nonincident_intersections"], 1)
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
