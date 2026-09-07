from copy import deepcopy
import unittest

from run_forest_native_admission import decide_runtime


def event():
    queries = [{'key': 'root', 'kind': 'exact_root'}, {'key': 'natural', 'kind': 'natural'}]
    return {'decision': 'UNKNOWN', 'schedule': {'all_queries': queries},
            'runtime': {'status': 'PREPARING', 'cases': []}}


class NativeAdmissionTests(unittest.TestCase):
    def test_root_alone_never_admits_entire_schedule(self):
        e = event()
        e['runtime']['cases'] = [{'query': e['schedule']['all_queries'][0], 'audit': {'status': 'PASS'}}]
        decide_runtime(e)
        self.assertEqual(e['decision'], 'UNKNOWN')
        self.assertEqual(e['runtime']['status'], 'UNKNOWN_REQUESTED_SCHEDULE')

    def test_all_queries_only_prepare_until_cache_reverified(self):
        e = event()
        e['runtime']['cases'] = [{'query': q, 'audit': {'status': 'PASS'}} for q in e['schedule']['all_queries']]
        decide_runtime(e)
        self.assertEqual(e['decision'], 'UNKNOWN')
        self.assertEqual(e['runtime']['status'], 'PREPARED_ENTIRE_REQUESTED_SCHEDULE')

    def test_unknown_separator_does_not_become_geometry_rejection(self):
        e = event()
        e['runtime']['cases'] = [{'query': q, 'audit': {'status': 'UNKNOWN', 'reason': 'No sufficient separator'}}
                                 for q in e['schedule']['all_queries']]
        decide_runtime(e)
        self.assertEqual(e['decision'], 'UNKNOWN')

    def test_necessary_actual_failure_discards_prior_proposals(self):
        e = event()
        e['runtime']['cases'] = [
            {'query': e['schedule']['all_queries'][0], 'audit': {'status': 'PASS'}},
            {'query': e['schedule']['all_queries'][1], 'audit': {'status': 'REJECT', 'reason': 'Actual graph fails',
                'certified_necessary_policy_failure': True}}]
        decide_runtime(e)
        self.assertEqual(e['decision'], 'REJECTED_FIXED_POLICY')
        self.assertTrue(e['decisive_rejection']['certified_necessary_policy_failure'])
        self.assertEqual(e['runtime']['discarded_proposals'], 1)
        self.assertFalse(e['runtime']['published_partial_results'])


if __name__ == '__main__':
    unittest.main()
