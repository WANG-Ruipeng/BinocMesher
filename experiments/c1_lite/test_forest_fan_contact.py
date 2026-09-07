import unittest
from forest_fan_contact import certify_fan_contact


class FanContactTests(unittest.TestCase):
    def run_case(self, triangle, ids, **kwargs):
        return certify_fan_contact(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)),
            (1, 1, 0), ((0, 0), (0, 1), (0, 2), (0, 3)), triangle, ids, (0, 4), **kwargs)

    def test_disjoint(self):
        r = self.run_case(((0, 0, 1), (2, 0, 1), (0, 2, 1)), ((1, 0), (1, 1), (1, 2)))
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(r['triangles_checked'], 4)

    def test_legal_shared_boundary_edge(self):
        r = self.run_case(((0, 0, 0), (2, 0, 0), (1, -1, 0)), ((0, 0), (0, 1), (0, 9)))
        self.assertEqual(r['status'], 'PASS')

    def test_foreign_crossing_has_exact_witness(self):
        r = self.run_case(((0.5, 0.25, -1), (0.5, 0.25, 1), (0.75, 0.25, 1)), ((1, 0), (1, 1), (1, 2)))
        self.assertEqual(r['status'], 'REJECT_POLICY_CONTACT')
        self.assertIn('witness', r['exact_contact'])
        self.assertFalse(r['baseline_novelty_checked'])

    def test_same_coordinates_other_element_are_not_shared_ids(self):
        r = self.run_case(((0, 0, 0), (2, 0, 0), (1, -1, 0)), ((1, 0), (1, 1), (1, 9)))
        self.assertEqual(r['status'], 'REJECT_POLICY_CONTACT')

    def test_arithmetic_budget_is_unknown(self):
        r = self.run_case(((0, 0, 1), (2, 0, 1), (0, 2, 1)), ((1, 0), (1, 1), (1, 2)), max_work=1)
        self.assertEqual(r['status'], 'UNKNOWN')

    def test_colliding_center_id_is_instrumentation_unknown(self):
        r = self.run_case(((0, 0, 1), (2, 0, 1), (0, 2, 1)), ((0, 4), (1, 1), (1, 2)))
        self.assertEqual(r['status'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
