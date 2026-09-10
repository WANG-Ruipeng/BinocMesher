"""Independent finite-pointset check of the bounded flip-graph enumeration.

The reference enumeration never calls a flip, insertion, or Delaunay helper.
For a convex boundary and one interior point, every triangulation consists of
the interior point's cyclic neighbor star and triangulated outer pockets.
Enumerating neighbor subsets and all pocket triangulations therefore supplies
an independent finite combinatorial check, including the collinear subsets in
this campaign. Floating geometric predicates remain fixture-level checks, not
an exact-predicate or general-position theorem.
"""
from collections import Counter
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'graph_lift_probe'))
from model import DOMAINS

from connectivity import enumerate_triangulations


def normalized_faces(faces):
    return tuple(sorted(tuple(sorted(int(v) for v in face)) for face in faces))


def signed_double_area(a, b, c):
    return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


@lru_cache(maxsize=None)
def convex_polygon_triangulations(indices):
    """Root each convex polygon triangulation at its first-to-last edge."""
    if len(indices) < 3:
        return ((),)
    result = []
    for pivot in range(1, len(indices)-1):
        triangle = (indices[0], indices[pivot], indices[-1])
        for left, right in product(
                convex_polygon_triangulations(indices[:pivot+1]),
                convex_polygon_triangulations(indices[pivot:])):
            result.append((triangle,) + left + right)
    return tuple(result)


def independent_triangulations(polygon, q):
    """All one-interior-point triangulations by stars and outer pockets."""
    polygon, q = np.asarray(polygon, float), np.asarray(q, float)
    m = len(polygon)
    result = {}
    for degree in range(3, m+1):
        for neighbors in combinations(range(m), degree):
            following = neighbors[1:] + neighbors[:1]
            # Positive star areas mean q is strictly inside the neighbor hull.
            # Collinear center/diagonal triples are excluded, not perturbed.
            if any(signed_double_area(q, polygon[a], polygon[b]) <= 1e-13
                   for a, b in zip(neighbors, following)):
                continue
            star = tuple((m, a, b) for a, b in zip(neighbors, following))
            pockets = []
            for a, b in zip(neighbors, following):
                chain = tuple((a+offset) % m for offset in range((b-a) % m+1))
                pockets.append(convex_polygon_triangulations(chain))
            for choices in product(*pockets):
                faces = star + tuple(face for pocket in choices for face in pocket)
                result[normalized_faces(faces)] = faces
    return result


def rooted_faces(m, root):
    order = tuple(range(root, m)) + tuple(range(root))
    return np.asarray([(order[0], order[i], order[i+1])
                       for i in range(1, m-1)], dtype=int)


def fixture_pointsets(polygon):
    """Rebuild all campaign XY sampling rules without connectivity helpers."""
    m = len(polygon)
    after = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0]*after[:, 1] - after[:, 0]*polygon[:, 1]
    centers = {
        'vertex_mean': polygon.mean(axis=0),
        'area_centroid': ((polygon+after)*cross[:, None]).sum(axis=0)/(3*cross.sum()),
    }
    for root in (0, 1):
        faces = rooted_faces(m, root)
        areas = [abs(signed_double_area(*polygon[face])) for face in faces]
        centers[f'largest_face_root{root}'] = polygon[faces[int(np.argmax(areas))]].mean(axis=0)
        edges = Counter(tuple(sorted((int(face[j]), int(face[(j+1) % 3]))))
                        for face in faces for j in range(3))
        interior = [edge for edge, count in edges.items() if count == 2]
        edge = min(interior, key=lambda e: (-float(np.sum((polygon[e[0]]-polygon[e[1]])**2)), e))
        centers[f'longest_edge_root{root}'] = polygon[list(edge)].mean(axis=0)
    return centers


class OrbitCompletenessTests(unittest.TestCase):
    def assert_valid_reference_mesh(self, polygon, q, faces):
        m = len(polygon)
        xy = np.vstack((polygon, q))
        self.assertEqual(len(faces), m)
        self.assertEqual(set(v for face in faces for v in face), set(range(m+1)))
        areas = [signed_double_area(*xy[list(face)]) for face in faces]
        self.assertTrue(all(area > 1e-13 for area in areas))
        polygon_area2 = sum(polygon[i, 0]*polygon[(i+1) % m, 1] -
                            polygon[(i+1) % m, 0]*polygon[i, 1] for i in range(m))
        self.assertAlmostEqual(sum(areas), polygon_area2, places=12)
        edges = Counter(tuple(sorted((face[j], face[(j+1) % 3])))
                        for face in faces for j in range(3))
        expected_boundary = {tuple(sorted((i, (i+1) % m))) for i in range(m)}
        self.assertEqual({edge for edge, count in edges.items() if count == 1}, expected_boundary)
        self.assertTrue(all(count in (1, 2) for count in edges.values()))
        self.assertEqual(m+1-len(edges)+len(faces), 1)

    def test_convex_pocket_counts_are_catalan(self):
        for vertices, count in ((2, 1), (3, 1), (4, 2), (5, 5), (6, 14)):
            choices = convex_polygon_triangulations(tuple(range(vertices)))
            self.assertEqual(len(choices), count)
            self.assertEqual(len({normalized_faces(faces) for faces in choices}), count)

    def test_square_center_has_only_one_valid_triangulation(self):
        polygon = DOMAINS['whole_square']
        reference = independent_triangulations(polygon, np.array([.5, .5]))
        self.assertEqual(len(reference), 1)

    def test_every_campaign_xy_pointset_matches_independent_enumeration(self):
        for domain, polygon in DOMAINS.items():
            m = len(polygon)
            start = np.asarray([(i, (i+1) % m, m) for i in range(m)], int)
            for rule, q in fixture_pointsets(polygon).items():
                with self.subTest(domain=domain, rule=rule):
                    reference = independent_triangulations(polygon, q)
                    self.assertTrue(reference)
                    for faces in reference.values():
                        self.assert_valid_reference_mesh(polygon, q, faces)
                    # Nonplanar Z values cannot affect this XY connectivity set.
                    vertices = np.column_stack((np.vstack((polygon, q)), np.arange(m+1)**2/7.))
                    original_vertices, original_start = vertices.copy(), start.copy()
                    actual, stats = enumerate_triangulations(vertices, start, m, max_states=256)
                    self.assertEqual({normalized_faces(faces) for faces in actual}, set(reference))
                    self.assertEqual(stats['states'], len(reference))
                    self.assertEqual(stats['oracle_queries'], 0)
                    np.testing.assert_array_equal(vertices, original_vertices)
                    np.testing.assert_array_equal(start, original_start)


if __name__ == '__main__':
    unittest.main()
