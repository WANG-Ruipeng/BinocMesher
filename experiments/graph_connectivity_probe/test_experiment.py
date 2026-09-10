import unittest
import numpy as np

from experiment import (construct_points, build_orbits, screen, RULES, oriented_canonical,
                        GraphModel, canonical_faces)
from model import DOMAINS


class ExperimentTests(unittest.TestCase):
    def test_point_budget_alias_and_immutable_reconnection(self):
        model=GraphModel(-np.ones(4),np.ones(4),np.array([0.,.1,.8,.2]),np.array([.3,.2,.9,.4]))
        for polygon in DOMAINS.values():
            snapshot=model.receipt()
            sets,budget=construct_points(model,polygon)
            self.assertEqual(len(sets),7)
            self.assertEqual(budget['actual_all_point_rules_interior_queries'],6)
            before={name:p['vertices'].copy() for name,p in sets.items()}
            result=build_orbits(sets,polygon)
            for name,points in result.items():
                np.testing.assert_array_equal(points['vertices'],before[name])
                self.assertEqual(points['budget']['interior_model_queries'],0 if name=='xyz_mean' else 1)
                self.assertEqual(points['flip_stats']['oracle_queries'],0)
                self.assertLessEqual(len(points['orbit']),36)
                for faces in points['orbit']:
                    self.assertEqual(len(faces),len(polygon))
                    self.assertEqual(canonical_faces(faces),canonical_faces(oriented_canonical(points['vertices'],faces)))
            self.assertEqual(snapshot,model.receipt())

    def test_frozen_metric_bridge_preflight(self):
        import json
        from experiment import PRIOR, measure_case
        frozen=PRIOR/'artifacts'/'run_20260907_130141'
        inputs=json.loads((frozen/'inputs.json').read_text())['input_cases']
        for index in (3,15):
            old=json.loads((frozen/f'case_{index:02d}.json').read_text())
            result=measure_case(inputs[index],old)
            self.assertTrue(result['all_invariants_passed'])
            self.assertLess(result['bridge_maximum_differences']['distance'],1e-10)
            self.assertLess(result['bridge_maximum_differences']['normal_degrees'],1e-8)

    def test_normal_and_distance_screens(self):
        def record(a,b,c):
            return {'metrics':[{'symmetric_mean_distance':v,
                'normal_correspondence_symmetric_mean_degrees':v} for v in (a,b,c)]}
        result=screen(record(.1,.1,.1),record(.2,.2,.2),1.)
        self.assertEqual(result['status'],'RESOLVED_IMPROVEMENT')
        result=screen(record(18.,18.,18.),record(15.,15.,15.),1.,
                      'normal_correspondence_symmetric_mean_degrees')
        self.assertEqual(result['status'],'RESOLVED_REGRESSION')


if __name__=='__main__':
    unittest.main()
