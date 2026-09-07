import unittest
import numpy as np
from forest_observer_contract import original_identity_receipt
from run_forest_native_admission import decide_runtime


class ObserverContractTests(unittest.TestCase):
    def snapshot(self):
        shifts = np.arange(20, dtype=np.int32).reshape(5, 4)
        shifts.flags.writeable = False
        return dict(identity_status=1, source_vid_encoding_version=2,
                    source_vid_encoding='ORIGINAL_EFFECTIVE_SOURCE_VID', source_vid_shifts=shifts)

    def test_original_receipt(self):
        self.assertEqual(original_identity_receipt(self.snapshot())['per_element_shifts'][4], [16, 17, 18, 19])

    def test_old_observer_is_not_source_identity(self):
        snapshot = self.snapshot()
        for key in ('source_vid_encoding_version', 'source_vid_encoding', 'source_vid_shifts'):
            changed = dict(snapshot)
            changed.pop(key)
            with self.assertRaises(ValueError):
                original_identity_receipt(changed)

    def test_normalized_keys_are_unverified(self):
        snapshot = self.snapshot()
        snapshot['source_vid_encoding'] = 'MERGER_NORMALIZED_UNVERIFIED'
        with self.assertRaises(ValueError):
            original_identity_receipt(snapshot)

    def test_mutable_or_wrong_layout_rejected(self):
        for shifts in (np.zeros((5, 4), np.int32), np.zeros((4, 5), np.int32), np.zeros((5, 4), np.int64)):
            snapshot = self.snapshot()
            snapshot['source_vid_shifts'] = shifts
            with self.assertRaises(ValueError):
                original_identity_receipt(snapshot)

    def test_unwitnessed_reject_cannot_resolve_population(self):
        query = {'key': 'root'}
        event = {'decision': 'UNKNOWN', 'schedule': {'all_queries': [query]},
                 'runtime': {'cases': [{'query': query, 'audit': {'status': 'REJECT'}}]}}
        decide_runtime(event)
        self.assertEqual(event['decision'], 'UNKNOWN')
        self.assertEqual(event['runtime']['status'], 'UNKNOWN_REQUESTED_SCHEDULE')


if __name__ == '__main__':
    unittest.main()
