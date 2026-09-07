import unittest
from fractions import Fraction as F

from forest_schedule_policy import make_schedule, summarize_decisions


class ForestSchedulePolicyTests(unittest.TestCase):
    def camera(self):
        return {'times_seconds': [(i+.5)/24 for i in range(64)],
            'time_mapping': {'delta_seconds': {'numerator': 5910997028921913, 'denominator': 9007199254740992}}}

    def test_first_root_original_natural_phase(self):
        result = make_schedule(self.camera(), F(5,4), F(3,2), F(7,4))
        self.assertEqual(result['hit_frame_numbers'], list(range(21,29)))
        self.assertEqual([q['frame_number'] for q in result['natural']], list(range(20,30)))
        self.assertEqual(len(result['all_queries']), 11)
        self.assertEqual(result['exact_root']['time_mode'], 'exact')
        self.assertTrue(result['root_excluded_from_natural_rates'])

    def test_second_root_phase(self):
        result = make_schedule(self.camera(), F(9,4), F(5,2), F(11,4))
        self.assertEqual(result['hit_frame_numbers'], list(range(37,45)))

    def test_actual_clock_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, 'Actual time scale differs'):
            make_schedule(self.camera(), F(5,4), F(3,2), F(7,4), actual_delta=.7)

    def test_unknown_never_becomes_zero_admission(self):
        result = summarize_decisions([{'event_id': 'a', 'decision': 'UNKNOWN'}], expected_count=1)
        self.assertIsNone(result['policy_admission_rate'])
        self.assertEqual(result['policy_admission_rate_bounds'], [0, 1])

    def test_all_proven_rejection_is_policy_zero_not_runtime_failure(self):
        event = {'event_id': 'a', 'decision': 'REJECTED_FIXED_POLICY',
                 'decisive_rejection': {'certified_necessary_policy_failure': True}}
        result = summarize_decisions([event], expected_count=1)
        self.assertEqual(result['policy_admission_rate'], 0)
        self.assertIsNone(result['conditional_runtime_commit_rate'])

    def test_source_pass_cannot_become_runtime_admit(self):
        with self.assertRaisesRegex(ValueError, 'cannot authorize'):
            summarize_decisions([{'event_id': 'a', 'decision': 'ADMITTED_REQUESTED_SCHEDULE'}], expected_count=1)

    def test_uncertified_reject_refused(self):
        with self.assertRaisesRegex(ValueError, 'lacks a decisive'):
            summarize_decisions([{'event_id': 'a', 'decision': 'REJECTED_FIXED_POLICY'}], expected_count=1)

    def test_duplicate_population_refused(self):
        with self.assertRaisesRegex(ValueError, 'Population'):
            summarize_decisions([{'event_id': 'a', 'decision': 'UNKNOWN'}]*2, expected_count=2)


if __name__ == '__main__':
    unittest.main()
