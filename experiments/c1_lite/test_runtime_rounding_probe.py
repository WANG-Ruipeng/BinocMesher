from fractions import Fraction as F
import unittest
import numpy as np
from runtime_rounding_probe import boundary_lookup,endpoint_contract,fan_metrics


class RuntimeRoundingProbeTests(unittest.TestCase):
    def test_duplicate_coordinates_stop(self):
        report=boundary_lookup(np.array([[0,0,0],[0,0,0]]),[[0,0,0]])
        self.assertTrue(report['status'].startswith('STOP'))
        self.assertEqual(report['coordinate_match_counts'],[2])

    def test_no_epsilon_for_missing_coordinate(self):
        report=boundary_lookup(np.array([[0,0,0]]),[[0,0,2**-100]])
        self.assertEqual(report['coordinate_match_counts'],[0])

    def test_exact_diagonal_contract(self):
        self.assertTrue(endpoint_contract([.5,.5,0],[0,0,0],[1,1,0])['strictly_inside_diagonal'])
        self.assertFalse(endpoint_contract([.5,.5,2**-100],[0,0,0],[1,1,0])['collinear_exact'])
        self.assertFalse(endpoint_contract([0,0,0],[0,0,0],[1,1,0])['strictly_inside_diagonal'])

    def test_prospective_area_and_error_are_exact(self):
        p=[[0,0,0],[2,0,0],[2,2,0],[0,2,0]];q=[[F(x) for x in row] for row in p]
        result=fan_metrics(p,[1,1,0],q,(F(1),F(1),F(0)))
        self.assertTrue(result['all_fan_triangles_nonzero_exact'])
        self.assertEqual(result['fan_triangle_area_approx'],[1,1,1,1])
        self.assertEqual(result['center_max_coordinate_quantization_error'],{'numerator':0,'denominator':1})


if __name__=='__main__':unittest.main()
