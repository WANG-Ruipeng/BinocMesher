import unittest

from run_forest_formal import classify_compiler


class FormalRunnerTests(unittest.TestCase):
    def test_unknown_never_classified_as_proven_rejection(self):
        self.assertIsNone(classify_compiler({'status': 'UNKNOWN',
            'reason': 'LEGACY_EXHAUSTIVE_SELECTOR_NO_CANDIDATE: not actually complete'}))

    def test_legacy_continuous_or_quantization_diagnostic_is_not_schedule_rejection(self):
        self.assertIsNone(classify_compiler({'status': 'REJECT_FIXED_SOURCE_COMPILER',
            'reason': 'LEGACY_MAPPING_OR_QUANTIZATION_GATE_REJECTED'}))

    def test_native_identity_gap_is_not_policy_rejection(self):
        self.assertIsNone(classify_compiler({'status': 'REJECT_FIXED_SOURCE_COMPILER',
            'reason': 'SELECTED_NATIVE_LEGACY_IDENTITY_DISAGREEMENT: owner'}))

    def test_declared_exhaustive_selector_failure_is_bound(self):
        result = classify_compiler({'status': 'REJECT_FIXED_SOURCE_COMPILER',
            'reason': 'LEGACY_EXHAUSTIVE_SELECTOR_NO_CANDIDATE: full selector domain exhausted',
            'rejected_stage': 'legacy_selector'})
        self.assertEqual(result['gate'], 'legacy_selector')
        self.assertTrue(result['certified_necessary_policy_failure'])
        self.assertEqual(len(result['evidence_binding']), 64)


if __name__ == '__main__':
    unittest.main()
