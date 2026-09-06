from fractions import Fraction as F
import unittest
import numpy as np

from runtime_breakpoint_crosscheck import round_fraction_binary32, exact_mesh_metrics, witness_matches, compare_workers


class BreakpointCrosscheckTests(unittest.TestCase):
    def test_exact_ties_even_ignores_double_seed_rounding(self):
        midpoint = F(1) + F(1, 2**24)
        self.assertEqual(round_fraction_binary32(midpoint), 1.0)
        self.assertEqual(round_fraction_binary32(midpoint + F(1, 2**80)), float(np.nextafter(np.float32(1), np.float32(2))))
        self.assertEqual(round_fraction_binary32(-midpoint-F(1, 2**80)), -float(np.nextafter(np.float32(1), np.float32(2))))

    def test_subnormal_tie(self):
        self.assertEqual(round_fraction_binary32(F(1, 2**150)), 0.0)
        self.assertEqual(round_fraction_binary32(F(1, 2**150)+F(1, 2**200)), float(np.nextafter(np.float32(0), np.float32(1))))

    def test_exact_zero_area_not_small_area_threshold(self):
        v = np.array([[0,0,0],[1,0,0],[2,0,0],[0,2**-120,0]], dtype=float)
        report = exact_mesh_metrics(v, np.array([[0,0,1],[0,1,2],[0,1,3]]))
        self.assertEqual(report['repeated_index_faces'], 1)
        self.assertEqual(report['exact_zero_area_faces'], 2)

    def test_coordinate_match_is_not_identity_and_preserves_orientation(self):
        v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0]], dtype=float)
        witness = {'event':'synthetic','source_vertices':['a','b','c'],'owners':[],
            'shared_source_vertices':['a'],'repeated_effective_SourceVID':False,
            'positions_t0':[[{'numerator':int(x),'denominator':1} for x in row] for row in v[:3]]}
        row = witness_matches(v, np.array([[0,1,2]]), [witness])[0]
        self.assertEqual(row['vertex_coordinate_match_counts'], [2,1,1])
        self.assertEqual(row['identity_equivalence'], 'UNKNOWN')
        self.assertEqual(witness_matches(v,np.array([[0,2,1]]),[witness])[0]['oriented_coordinate_face_match_count'],0)

    def test_incomplete_workers_cannot_pass(self):
        r={'times':[], 'cases':[], 'status':'STOP_DIAGNOSTIC'}
        self.assertNotEqual(compare_workers(r,r)['status'],'PASS_OBSERVED_OMP_DETERMINISM')


if __name__ == '__main__':
    unittest.main()
