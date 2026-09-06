import unittest
from fractions import Fraction as F
from window_admission import REQUIRED_GATES, decide_admission, select_action


class AdmissionTests(unittest.TestCase):
    def complete(self):
        return {key: {'status': 'PASS'} for key in REQUIRED_GATES}

    def test_missing_evidence_never_admits(self):
        self.assertEqual(decide_admission({})['status'], 'FAIL_CLOSED')
        for key in REQUIRED_GATES:
            gates = self.complete()
            del gates[key]
            decision = decide_admission(gates)
            self.assertFalse(decision['runtime_plan_authorized'])
            self.assertIn(key, decision['unresolved_gates'])

    def test_reject_and_unknown_have_distinct_reasons(self):
        gates = self.complete()
        gates['interface_degeneracy_policy'] = {'status': 'REJECT', 'reason': 'Inherited interface degeneracy.'}
        gates['binary32_geometry_contact'] = {'status': 'UNKNOWN'}
        result = decide_admission(gates)
        self.assertEqual(result['rejected_gates'], ['interface_degeneracy_policy'])
        self.assertEqual(result['unresolved_gates'], ['binary32_geometry_contact'])

    def test_failed_window_is_baseline_even_at_root(self):
        decision = decide_admission({})
        for t in (-1, 0, F(1, 2), 1, F(3, 2), 2, 3):
            self.assertEqual(select_action(t, 0, 1, 2, decision), 'BASELINE')
        self.assertFalse(decision['root_only_c0_fallback_allowed'])

    def test_complete_gate_contract_and_endpoints(self):
        decision = decide_admission(self.complete())
        self.assertEqual(decision['status'], 'ADMIT')
        self.assertEqual(select_action(0, 0, 1, 2, decision), 'BASELINE')
        self.assertEqual(select_action(2, 0, 1, 2, decision), 'BASELINE')
        self.assertEqual(select_action(F(1, 2), 0, 1, 2, decision), 'CERTIFIED_WINDOW_INTERIOR')
        self.assertEqual(select_action(1, 0, 1, 2, decision), 'CERTIFIED_WINDOW_ROOT')

    def test_stale_admit_flag_cannot_bypass_missing_gate(self):
        self.assertEqual(select_action(1, 0, 1, 2, {'status': 'ADMIT'}), 'BASELINE')

    def test_invalid_gates_and_windows(self):
        for status in (True, 'PASS_IDEAL_ONLY', None):
            with self.assertRaises(ValueError):
                decide_admission({'source_ownership': {'status': status}})
        with self.assertRaises(ValueError):
            select_action(0, 0, 0, 1, decide_admission({}))


if __name__ == '__main__':
    unittest.main()
