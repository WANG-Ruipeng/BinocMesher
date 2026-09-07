"""Synthetic actual-ledger geometry tests; no native cache or renderer."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forest_native_patch import spec_from_source, audit_query
from test_window_runtime import fixture
from binocmesher.window_runtime import resolve_support


def make_snapshot(element=2):
    mesh, vids, owners, _, document = fixture()
    owners[:, 0] = element
    document = deepcopy(document)
    document['element'] = element
    for row in document['segments']+document['breakpoint_points']:
        for owner in row['owners']:
            owner[0] = element
    meshes = [(np.empty((0, 3), float), np.empty((0, 3), np.int32), np.empty(0, np.int32))
              for _ in range(5)]
    vertices = [np.empty((0, 4), np.int32) for _ in range(5)]
    ledgers = [np.empty((0, 11), np.int32) for _ in range(5)]
    meshes[element], vertices[element], ledgers[element] = mesh, vids, owners
    snapshot = dict(meshes=tuple(meshes), vertex_ledgers=tuple(vertices), owner_ledgers=tuple(ledgers),
                    identity_status=1, source_vid_encoding_version=2,
                    source_vid_encoding='ORIGINAL_EFFECTIVE_SOURCE_VID',
                    source_vid_shifts=np.zeros((5, 4), np.int32))
    freeze(snapshot)
    return snapshot, document


def freeze(snapshot):
    for a in [a for mesh in snapshot['meshes'] for a in mesh]+list(snapshot['vertex_ledgers'])+list(snapshot['owner_ledgers']):
        a.flags.writeable = False
    if isinstance(snapshot.get('source_vid_shifts'), np.ndarray):
        snapshot['source_vid_shifts'].flags.writeable = False


def edit_array(snapshot, category, element, index=None):
    snapshot.pop('_forest_native_patch_cache', None)
    rows = list(snapshot[category])
    if index is None:
        result = rows[element].copy()
        rows[element] = result
    else:
        mesh = list(rows[element])
        result = mesh[index].copy()
        mesh[index] = result
        rows[element] = tuple(mesh)
    snapshot[category] = tuple(rows)
    return result


def add_external(snapshot, element, coords, face=(0, 1, 2)):
    meshes = list(snapshot['meshes'])
    meshes[element] = (np.asarray(coords, float), np.asarray([face], np.int32), np.ones(len(coords), np.int32))
    vids = list(snapshot['vertex_ledgers'])
    owners = list(snapshot['owner_ledgers'])
    vids[element] = np.asarray([(i, 0, i+100, 0) for i in range(len(coords))], np.int32)
    owners[element] = np.asarray([(element, 0, 0, 0, 0, 0, 0, 0, *face)], np.int32)
    snapshot.update(meshes=tuple(meshes), vertex_ledgers=tuple(vids), owner_ledgers=tuple(owners))
    freeze(snapshot)


def translate_source(snapshot, document, element, translation):
    translation = np.asarray(translation, np.int32)
    vids = edit_array(snapshot, 'vertex_ledgers', element)
    vids[:] += translation
    def moved(label):
        a, b = [tuple(map(int, part.split(':'))) for part in label.split('|')]
        row = np.asarray((*a, *b), np.int32)+translation
        return str(row[0])+':'+str(row[1])+'|'+str(row[2])+':'+str(row[3])
    document['boundary_cycle'] = [moved(x) for x in document['boundary_cycle']]
    for row in document['segments']+document['breakpoint_points']:
        row['boundary_cycle'] = [moved(x) for x in row['boundary_cycle']]
        row['source_faces'] = [[moved(x) for x in face] for face in row['source_faces']]
    shifts = snapshot['source_vid_shifts'].copy()
    shifts[element] = vids.min(axis=0)
    snapshot['source_vid_shifts'] = shifts
    freeze(snapshot)


class NativePatchTests(unittest.TestCase):
    def test_all_five_elements_preserve_owner_identity_and_full_resolver_parity(self):
        for element in range(5):
            with self.subTest(element=element):
                snapshot, doc = make_snapshot(element)
                spec = spec_from_source(doc)
                plan, report = audit_query(snapshot, spec, F(1))
                self.assertEqual(report['status'], 'PASS', report)
                expected = resolve_support(snapshot['meshes'][element], snapshot['vertex_ledgers'][element],
                    snapshot['owner_ledgers'][element], spec, F(1))
                self.assertEqual((plan['boundary_actual_ids'], plan['removed_face_rows'], plan['consumed_owners']), expected)
                self.assertTrue(all(o[0] == element for o in plan['consumed_owners']))
                self.assertEqual(report['retained_faces_checked'], 8)

    def test_spec_immutable_and_does_not_mutate_source(self):
        _, doc = make_snapshot()
        original = deepcopy(doc)
        spec = spec_from_source(doc)
        self.assertEqual(doc, original)
        with self.assertRaises(FrozenInstanceError):
            spec.element = 0
        self.assertEqual(spec.center(F(1)).tolist(), [1, 1, 0])

    def test_mixed_element_source_rejected_by_parser(self):
        _, doc = make_snapshot()
        doc['segments'][0]['owners'][0][0] = 3
        with self.assertRaises(ValueError):
            spec_from_source(doc)

    def test_partial_owner_equivalence_is_rejected(self):
        snapshot, doc = make_snapshot()
        for row in doc['segments']+doc['breakpoint_points']:
            row['owners'] = row['owners'][:2]
        plan, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertIsNone(plan)
        self.assertEqual(report['status'], 'REJECT')
        self.assertIn('Partial suppression', report['reason'])

    def test_unrequested_malformed_owner_ledger_remains_unknown(self):
        snapshot, doc = make_snapshot()
        owners = edit_array(snapshot, 'owner_ledgers', 2)
        owners[8, 8] = (owners[8, 8]+1) % 8
        freeze(snapshot)
        plan, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertIsNone(plan)
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertIn('disagree', report['reason'])

    def test_duplicate_owner_anywhere_is_unknown(self):
        snapshot, doc = make_snapshot()
        owners = edit_array(snapshot, 'owner_ledgers', 2)
        owners[8, :7] = owners[7, :7]
        freeze(snapshot)
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertIn('Duplicate raw owner', report['reason'])

    def test_native_missing_requested_owner_is_reject(self):
        snapshot, doc = make_snapshot()
        owners = edit_array(snapshot, 'owner_ledgers', 2)
        owners[0, 3] = 999
        freeze(snapshot)
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'REJECT')
        self.assertIn('not emitted', report['reason'])

    def test_nonbinary32_actual_input_is_unknown(self):
        snapshot, doc = make_snapshot()
        v = edit_array(snapshot, 'meshes', 2, 0)
        v[4, 2] = 0.1
        freeze(snapshot)
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'UNKNOWN')

    def test_actual_center_on_boundary_is_reject(self):
        snapshot, doc = make_snapshot()
        doc['anchors']['root']['position'] = [0, 1, 0]
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'REJECT')
        self.assertIn('strictly inside', report['reason'])

    def test_actual_nonconvex_boundary_is_reject(self):
        snapshot, doc = make_snapshot()
        v = edit_array(snapshot, 'meshes', 2, 0)
        v[2] = [1, 0, 0]
        freeze(snapshot)
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'REJECT')
        self.assertIn('strict convex', report['reason'])

    def test_shared_geometric_degeneracy_is_reject_not_unknown(self):
        snapshot, doc = make_snapshot()
        v = edit_array(snapshot, 'meshes', 2, 0)
        v[4] = [1, 0, 0]
        freeze(snapshot)
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'REJECT')
        self.assertIn('Degenerate', report['reason'])
        self.assertTrue(report['certified_necessary_policy_failure'])
        self.assertEqual(report['boundary_actual_ids'], [0, 1, 2, 3])
        self.assertEqual(report['replaced_face_rows'], [0, 1])
        self.assertEqual(len(report['consumed_owners']), 3)
        self.assertEqual(report['failing_face_exact_cross'], ['0', '0', '0'])
        self.assertEqual(len(report['failing_face_actual_ids']), 3)
        self.assertEqual(report['failing_face_binary32_coordinates'],
                         snapshot['meshes'][2][0][report['failing_face_actual_ids']].tolist())
        self.assertTrue(report['failing_face_shared_boundary_ids'])

    def test_cross_element_same_index_is_not_shared_identity(self):
        snapshot, doc = make_snapshot()
        add_external(snapshot, 0, [[0, 0, 0], [2, 0, 0], [1, -1, 0]])
        plan, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertIsNone(plan)
        self.assertEqual(report['status'], 'UNKNOWN', report)
        self.assertEqual(report['failing_element'], 0)

    def test_unshared_interior_contact_is_never_pass(self):
        snapshot, doc = make_snapshot()
        add_external(snapshot, 4, [[0.5, 0.5, 0], [1.5, 0.5, 0], [1, 1.5, 0]])
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'UNKNOWN', report)

    def test_remote_degenerate_cross_element_is_safely_separated(self):
        snapshot, doc = make_snapshot()
        add_external(snapshot, 0, [[10, 0, 0], [11, 0, 0], [12, 0, 0]])
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(report['retained_faces_checked'], 9)
        self.assertEqual(report['expected_retained_faces'], 9)

    def test_missing_interface_collar_is_reject(self):
        snapshot, doc = make_snapshot()
        f = edit_array(snapshot, 'meshes', 2, 1)
        f[2] = f[3]
        owners = edit_array(snapshot, 'owner_ledgers', 2)
        owners[2, 8:] = f[2]
        freeze(snapshot)
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'REJECT')
        self.assertEqual(report['stage'], 'actual_interface')

    def test_local_only_never_returns_committable_plan(self):
        snapshot, doc = make_snapshot()
        plan, report = audit_query(snapshot, spec_from_source(doc), F(1), full_exterior=False)
        self.assertIsNone(plan)
        self.assertEqual(report['status'], 'PASS_LOCAL_ONLY')

    def test_cached_index_does_not_modify_baseline_and_rebinding_is_unknown(self):
        snapshot, doc = make_snapshot()
        before = [a.tobytes() for mesh in snapshot['meshes'] for a in mesh]
        spec = spec_from_source(doc)
        _, first = audit_query(snapshot, spec, F(1))
        cache = snapshot['_forest_native_patch_cache']
        _, second = audit_query(snapshot, spec, F(1, 2))
        self.assertIs(cache, snapshot['_forest_native_patch_cache'])
        self.assertEqual(first['status'], second['status'])
        self.assertEqual(before, [a.tobytes() for mesh in snapshot['meshes'] for a in mesh])
        rows = list(snapshot['vertex_ledgers'])
        rows[2] = rows[2].copy()
        rows[2].flags.writeable = False
        snapshot['vertex_ledgers'] = tuple(rows)
        _, third = audit_query(snapshot, spec, F(1))
        self.assertEqual(third['status'], 'UNKNOWN')
        self.assertIn('binding changed', third['reason'])

    def test_identity_observer_cap_is_unknown(self):
        snapshot, doc = make_snapshot()
        snapshot['identity_status'] = 3
        _, report = audit_query(snapshot, spec_from_source(doc), F(1))
        self.assertEqual(report['status'], 'UNKNOWN')

    def test_missing_or_legacy_vid_encoding_is_unknown_before_identity_resolution(self):
        for metadata in ({}, {'source_vid_encoding_version': 0,
                             'source_vid_encoding': 'MERGER_NORMALIZED_UNVERIFIED'},
                         {'source_vid_encoding_version': 2,
                          'source_vid_encoding': 'MERGER_NORMALIZED_UNVERIFIED'}):
            snapshot, doc = make_snapshot()
            snapshot.pop('source_vid_encoding_version')
            snapshot.pop('source_vid_encoding')
            snapshot.update(metadata)
            _, report = audit_query(snapshot, spec_from_source(doc), F(1))
            self.assertEqual(report['status'], 'UNKNOWN')
            self.assertIn('encoding is unverified', report['reason'])
            self.assertNotIn('certified_necessary_policy_failure', report)

    def test_shift_metadata_must_be_readonly_int32_five_by_four(self):
        candidates = [None, np.zeros((4, 4), np.int32), np.zeros((5, 4), np.int64),
                      np.zeros((5, 4), np.int32), -np.ones((5, 4), np.int32)]
        for i, shifts in enumerate(candidates):
            snapshot, doc = make_snapshot()
            if isinstance(shifts, np.ndarray) and i != 3:
                shifts.flags.writeable = False
            snapshot['source_vid_shifts'] = shifts
            _, report = audit_query(snapshot, spec_from_source(doc), F(1))
            self.assertEqual(report['status'], 'UNKNOWN', (i, report))
            self.assertIn('shift binding', report['reason'])

    def test_nonzero_ordered_field_shifts_restore_identity_without_changing_geometry(self):
        for element, translation in ((0, (1000, 7, 1500, 9)), (2, (5000, 11, 200, 3)),
                                     (4, (100000, 1, 200000, 2))):
            with self.subTest(element=element):
                snapshot, doc = make_snapshot(element)
                before = [a.tobytes() for mesh in snapshot['meshes'] for a in mesh]
                owner_bytes = snapshot['owner_ledgers'][element].tobytes()
                translate_source(snapshot, doc, element, translation)
                spec = spec_from_source(doc)
                _, report = audit_query(snapshot, spec, F(1))
                self.assertEqual(report['status'], 'PASS', report)
                original = snapshot['vertex_ledgers'][element].copy()
                minima = original.min(axis=0)
                normalized = edit_array(snapshot, 'vertex_ledgers', element)
                normalized[:] -= minima
                snapshot.update(source_vid_encoding_version=0,
                                source_vid_encoding='MERGER_NORMALIZED_UNVERIFIED')
                freeze(snapshot)
                _, old = audit_query(snapshot, spec, F(1))
                self.assertEqual(old['status'], 'UNKNOWN', old)
                restored = edit_array(snapshot, 'vertex_ledgers', element)
                restored[:] += minima
                snapshot.update(source_vid_encoding_version=2,
                                source_vid_encoding='ORIGINAL_EFFECTIVE_SOURCE_VID')
                freeze(snapshot)
                _, new = audit_query(snapshot, spec, F(1))
                self.assertEqual(new['status'], 'PASS', new)
                self.assertEqual(original.tobytes(), restored.tobytes())
                self.assertEqual(before, [a.tobytes() for mesh in snapshot['meshes'] for a in mesh])
                self.assertEqual(owner_bytes, snapshot['owner_ledgers'][element].tobytes())
                self.assertFalse(new['source_vid_shifts_applied_by_python'])

    def test_query_specific_minima_do_not_change_stable_boundary_source_identity(self):
        observed = []
        for tau, remote in ((F(1, 2), (10, 1, 110, 3)), (F(3, 2), (50, 2, 150, 4))):
            snapshot, doc = make_snapshot(2)
            translate_source(snapshot, doc, 2, (1000, 7, 1500, 9))
            vids = edit_array(snapshot, 'vertex_ledgers', 2)
            for k in range(4, 8):
                vids[k] = np.asarray(remote)+(k, 0, k, 0)
            shifts = snapshot['source_vid_shifts'].copy()
            shifts[2] = vids.min(axis=0)
            snapshot['source_vid_shifts'] = shifts
            freeze(snapshot)
            plan, report = audit_query(snapshot, spec_from_source(doc), tau)
            self.assertEqual(report['status'], 'PASS', report)
            observed.append((doc['boundary_cycle'], shifts[2].tolist(), plan['boundary_actual_ids']))
        self.assertEqual(observed[0][0], observed[1][0])
        self.assertNotEqual(observed[0][1], observed[1][1])
        self.assertEqual(observed[0][2], observed[1][2])

    def test_rebinding_shift_metadata_invalidates_cached_index(self):
        snapshot, doc = make_snapshot()
        spec = spec_from_source(doc)
        self.assertEqual(audit_query(snapshot, spec, F(1))[1]['status'], 'PASS')
        snapshot['source_vid_shifts'] = snapshot['source_vid_shifts'].copy()
        snapshot['source_vid_shifts'].flags.writeable = False
        _, report = audit_query(snapshot, spec, F(1))
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertIn('binding changed', report['reason'])

    def test_internal_contact_exception_is_unknown_not_geometric_reject(self):
        snapshot, doc = make_snapshot()
        spec = spec_from_source(doc)
        with patch('binocmesher.window_geometry_runtime._contact', side_effect=ValueError('internal diagnostic failure')):
            plan, report = audit_query(snapshot, spec, F(1))
        self.assertIsNone(plan)
        self.assertEqual(report['status'], 'UNKNOWN')
        self.assertEqual(report['exception_type'], 'ValueError')
        self.assertEqual(report['stage'], 'actual_five_element_exterior')

    def test_endpoint_and_outside_are_baseline_only(self):
        snapshot, doc = make_snapshot()
        spec = spec_from_source(doc)
        for tau in (F(-1), F(0), F(2), F(3)):
            plan, report = audit_query(snapshot, spec, tau)
            self.assertIsNone(plan)
            self.assertEqual(report['status'], 'PASS_BASELINE_ONLY')


if __name__ == '__main__':
    unittest.main()
