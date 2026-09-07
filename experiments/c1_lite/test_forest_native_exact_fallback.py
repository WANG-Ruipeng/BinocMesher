from fractions import Fraction as F
import unittest
from unittest.mock import patch

from forest_native_patch import audit_query, spec_from_source, _modules
from test_forest_native_patch import make_snapshot, add_external


class NativeExactFallbackTests(unittest.TestCase):
    def test_exactly_decides_former_unknown_contact(self):
        snapshot, document = make_snapshot()
        add_external(snapshot, 4, [[0.5, 0.5, 0], [1.5, 0.5, 0], [1, 1.5, 0]])
        _, old = audit_query(snapshot, spec_from_source(document), F(1))
        self.assertEqual(old['status'], 'UNKNOWN')
        _, result = audit_query(snapshot, spec_from_source(document), F(1), exact_contact_fallback=True)
        self.assertEqual(result['status'], 'REJECT')
        self.assertTrue(result['certified_necessary_policy_failure'])
        self.assertEqual(result['reason_code'], 'FORBIDDEN_ACTUAL_PATCH_RETAINED_CONTACT')
        self.assertEqual(result['exact_contact_fallback']['exact_contact']['relation'], 'FORBIDDEN_CONTACT')
        self.assertFalse(result['exact_contact_fallback']['baseline_novelty_checked'])

    def test_exact_shared_collar_pass_covers_entire_retained_mesh(self):
        snapshot, document = make_snapshot()
        before = [a.tobytes() for mesh in snapshot['meshes'] for a in mesh]
        _, geometry = _modules()
        with patch.object(geometry, '_contact', return_value=(None, 'No sufficient separator')):
            plan, result = audit_query(snapshot, spec_from_source(document), F(1), exact_contact_fallback=True)
        self.assertEqual(result['status'], 'PASS', result)
        self.assertIsNotNone(plan)
        self.assertEqual(result['retained_faces_checked'], result['expected_retained_faces'])
        self.assertGreater(result['certificate_counts']['EXACT_FOUR_FAN_TRIANGLE_CONTACT'], 0)
        self.assertEqual(before, [a.tobytes() for mesh in snapshot['meshes'] for a in mesh])

    def test_budget_unknown_is_not_policy_rejection(self):
        snapshot, document = make_snapshot()
        add_external(snapshot, 4, [[0.5, 0.5, 0], [1.5, 0.5, 0], [1, 1.5, 0]])
        with patch('forest_fan_contact.certify_fan_contact', return_value={
                'status': 'UNKNOWN', 'arithmetic_work': 1, 'reason': 'test arithmetic limit'}):
            _, result = audit_query(snapshot, spec_from_source(document), F(1), exact_contact_fallback=True)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertNotIn('certified_necessary_policy_failure', result)


if __name__ == '__main__':
    unittest.main()
