import unittest
from sidewall_contract import audit_sidewall, require_window_boundary_contract


def cylinder(transform=lambda x,y,t: (x,y,t)):
    return [[*transform(x,y,t), float(t)] for t in (0,1,2)
            for x,y in ((0,0),(1,0),(1,1),(0,1),(.5,.5))]


class SidewallContractTests(unittest.TestCase):
    def test_translation(self):
        self.assertTrue(audit_sidewall(cylinder())['pass'])

    def test_stretch_without_rotation(self):
        self.assertTrue(audit_sidewall(cylinder(lambda x,y,t: ((1+t)*x, (1+2*t)*y, 0)))['pass'])

    def test_rotating_edge_rejected_even_in_spatial_plane(self):
        self.assertFalse(audit_sidewall(cylinder(lambda x,y,t: (x, y+t*x, 0)))['pass'])

    def test_no_epsilon_acceptance(self):
        self.assertFalse(audit_sidewall(cylinder(lambda x,y,t: (x, y, t*x*1e-12)))['pass'])

    def test_collapse_between_levels_rejected(self):
        self.assertFalse(audit_sidewall(cylinder(lambda x,y,t: ((1-2*t)*x,y,0)))['pass'])

    def test_forged_or_legacy_pass_cannot_authorize_window(self):
        record = {'pass': True, 'sidewall_geometry_compatible': True,
                  'vertices4': cylinder(lambda x,y,t: (x,y,t*x))}
        with self.assertRaisesRegex(ValueError, 'REJECT_LINEAR_SIDEWALL_WINDOW'):
            require_window_boundary_contract(record)

    def test_invalid_layout(self):
        values = cylinder()
        values[1][3] = .25
        with self.assertRaises(ValueError):
            audit_sidewall(values)


if __name__ == '__main__':
    unittest.main()
