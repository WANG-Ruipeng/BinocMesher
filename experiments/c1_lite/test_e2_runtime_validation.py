import copy
import unittest

import numpy as np

from e2_runtime_validation import canonical_vid, validate_e2_runtime


def fixture():
    v = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
                  [4, 0, 0], [5, 0, 0], [4, 1, 0]], dtype=np.float64)
    f = np.array([[0, 1, 2], [4, 5, 6], [0, 2, 3]], dtype=np.int32)
    t = np.arange(7, dtype=np.int32)
    vids = np.array([[i, 0, i+100, 0] for i in range(7)], dtype=np.int32)
    owners = np.array([
        [0, 0, 0, 10, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 11, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 12, 0, 0, 0, 2, 0, 2, 3],
        [0, 0, 0, 13, 0, 0, 0, 1, 4, 5, 6]], dtype=np.int64)
    rv = np.vstack((v, [1., 1., 0.]))
    rf = np.array([[7, 0, 1], [4, 5, 6], [7, 1, 2],
                   [7, 2, 3], [7, 3, 0]], dtype=np.int32)
    rt = np.append(t, np.int32(1))
    requested = [tuple(row[:7]) for row in owners[:3]]
    return dict(baseline=(v, f, t), result=(rv, rf, rt), vertex_ledger=vids,
                owner_ledger=owners, boundary_cycle=vids[:4].copy(),
                suppression_owners=requested, consumed_owners=list(requested),
                expected_center=np.array([1., 1., 0.]))


class E2RuntimeValidationTests(unittest.TestCase):
    def setUp(self):
        self.args = fixture()

    def check_stop(self, fragment=None):
        report = validate_e2_runtime(**self.args)
        self.assertFalse(report['pass'], report)
        if fragment:
            self.assertIn(fragment, ' '.join(report['errors']))
        return report

    def test_valid_full_owner_contract(self):
        report = validate_e2_runtime(**self.args)
        self.assertTrue(report['pass'], report)
        self.assertEqual(report['expected_removed_face_rows'], [0, 2])
        self.assertEqual(report['expected_raw_owner_count'], 3)
        self.assertEqual(report['geometry_admission'], 'NOT_CERTIFIED')

    def test_owner_and_request_permutation(self):
        self.args['owner_ledger'] = self.args['owner_ledger'][::-1]
        self.args['suppression_owners'].reverse()
        self.args['consumed_owners'].reverse()
        self.assertTrue(validate_e2_runtime(**self.args)['pass'])

    def test_reversed_ordered_vid_and_tokens(self):
        self.args['vertex_ledger'] = self.args['vertex_ledger'][:, [2, 3, 0, 1]]
        self.args['boundary_cycle'] = [f'{i}:0|{i+100}:0' for i in range(4)]
        self.assertTrue(validate_e2_runtime(**self.args)['pass'])
        self.assertEqual(canonical_vid('100:0|0:0'), (0, 0, 100, 0))

    def test_same_coordinates_different_vid_do_not_alias(self):
        self.args['baseline'][0][4] = self.args['baseline'][0][0]
        self.args['result'][0][4] = self.args['baseline'][0][0]
        report = validate_e2_runtime(**self.args)
        self.assertTrue(report['pass'], report)
        self.assertEqual(report['boundary_global_ids'], [0, 1, 2, 3])

    def test_same_coordinate_wrong_vid_is_not_boundary(self):
        self.args['vertex_ledger'][0] = [900, 0, 901, 0]
        self.check_stop('Boundary SourceVID missing')

    def test_canonical_alias_conflict(self):
        self.args['vertex_ledger'][4] = self.args['vertex_ledger'][0][[2, 3, 0, 1]]
        self.check_stop('canonical alias')

    def test_frozen_cycle_alias(self):
        self.args['boundary_cycle'][3] = self.args['boundary_cycle'][0]
        self.check_stop('Frozen boundary contains canonical aliases')

    def test_partial_requested_owner_replicas(self):
        self.args['suppression_owners'].pop(1)
        self.args['consumed_owners'].pop(1)
        self.check_stop('omits raw owner replicas')

    def test_partial_actual_consumption(self):
        self.args['consumed_owners'].pop()
        self.check_stop('not exact-once')

    def test_duplicate_actual_consumption(self):
        self.args['consumed_owners'].append(self.args['consumed_owners'][0])
        self.check_stop('not exact-once')

    def test_unrequested_consumption(self):
        self.args['consumed_owners'].append(tuple(self.args['owner_ledger'][3, :7]))
        self.check_stop('not exact-once')

    def test_duplicate_raw_owner_ledger(self):
        self.args['owner_ledger'] = np.vstack((self.args['owner_ledger'], self.args['owner_ledger'][0]))
        self.check_stop('duplicates a raw owner')

    def test_consumption_set_cannot_hide_multiplicity(self):
        self.args['consumed_owners'] = set(self.args['consumed_owners'])
        self.check_stop('preserve multiplicity')

    def test_ledger_requires_complete_baseline_face_coverage(self):
        self.args['owner_ledger'] = self.args['owner_ledger'][:3]
        self.check_stop('cover every baseline face')

    def test_owner_row_face_mismatch(self):
        self.args['owner_ledger'][0, 7] = 1
        self.check_stop('disagree with canonical baseline face')

    def test_steal_remote_face_same_net_counts(self):
        self.args['result'][1][1] = [7, 1, 2]
        self.args['result'][1][2] = self.args['baseline'][1][2]
        self.check_stop('External face row changed')

    def test_authorize_remote_instead_of_patch_face(self):
        request = [tuple(self.args['owner_ledger'][i, :7]) for i in (0, 1, 3)]
        self.args['suppression_owners'] = request
        self.args['consumed_owners'] = request
        self.check_stop('frozen oriented quad boundary')

    def test_tag_change(self):
        self.args['result'][2][4] += 1
        self.check_stop('tag prefix changed')

    def test_negative_zero_vertex_change_is_not_equal_bits(self):
        self.args['result'][0][0, 2] = -0.0
        self.check_stop('vertex prefix changed')

    def test_external_face_rotation_still_changes_row(self):
        self.args['result'][1][1] = [5, 6, 4]
        self.check_stop('External face row changed')

    def test_fan_cyclic_rotation_permitted(self):
        for index in (0, 2, 3, 4):
            self.args['result'][1][index] = np.roll(self.args['result'][1][index], 1)
        self.assertTrue(validate_e2_runtime(**self.args)['pass'])

    def test_reversed_fan_rejected(self):
        self.args['result'][1][0] = [7, 1, 0]
        self.check_stop('prescribed oriented fan')

    def test_duplicate_fan_rejected(self):
        self.args['result'][1][4] = self.args['result'][1][0]
        self.check_stop('prescribed oriented fan')

    def test_fan_slot_swap_rejected(self):
        f = self.args['result'][1]
        f[[0, 2]] = f[[2, 0]]
        self.check_stop('prescribed oriented fan')

    def test_exact_zero_area_rejected(self):
        self.args['result'][0][-1] = [1, 0, 0]
        self.args['expected_center'] = None
        self.check_stop('exactly zero-area')

    def test_tiny_nonzero_area_no_tolerance(self):
        self.args['result'][0][-1] = [1, 2**-120, 0]
        self.args['expected_center'] = None
        self.assertTrue(validate_e2_runtime(**self.args)['pass'])

    def test_optional_expected_center_is_explicit(self):
        self.args['expected_center'] = None
        report = validate_e2_runtime(**self.args)
        self.assertTrue(report['pass'])
        self.assertFalse(report['expected_center_checked'])

    def test_wrong_center_bits(self):
        self.args['expected_center'][0] = np.nextafter(1., 2.)
        self.check_stop('expected center bits')

    def test_nonfinite(self):
        self.args['result'][0][-1, 0] = np.nan
        self.check_stop('nonfinite')

    def test_float_ledger_rejected(self):
        self.args['owner_ledger'] = self.args['owner_ledger'].astype(float)
        self.check_stop('integer dtype')

    def test_inputs_not_mutated(self):
        before = copy.deepcopy(self.args)
        validate_e2_runtime(**self.args)
        for key in ('baseline', 'result'):
            for a, b in zip(self.args[key], before[key]):
                self.assertEqual(a.tobytes(), b.tobytes())
        self.assertTrue(np.array_equal(self.args['owner_ledger'], before['owner_ledger']))


if __name__ == '__main__':
    unittest.main()
