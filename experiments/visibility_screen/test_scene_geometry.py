"""Frozen five-element parity and real dynamic namespaces; no native queries."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scene_geometry as m
import forest_native_patch as old
import forest_component_union as old_union
from forest_component_graph import actual_footprints as old_footprints
from forest_native_campaign import mesh_receipt
from forest_observer_contract import original_identity_receipt as old_identity
from test_forest_native_patch import make_snapshot, freeze, add_external
from test_forest_component_union import scene as old_scene
from test_window_runtime import fixture


def stable(report):
    return {k: v for k, v in report.items() if k != 'wall_seconds'}


def snapshot(n=2, element=1):
    mesh, vids, owners, _, document = fixture()
    owners[:, 0] = element
    document = deepcopy(document); document['element'] = element
    for row in document['segments']+document['breakpoint_points']:
        for owner in row['owners']: owner[0] = element
    meshes = [(np.empty((0, 3), float), np.empty((0, 3), np.int32), np.empty(0, np.int32)) for _ in range(n)]
    ids = [np.empty((0, 4), np.int32) for _ in range(n)]
    ledgers = [np.empty((0, 11), np.int32) for _ in range(n)]
    meshes[element], ids[element], ledgers[element] = mesh, vids, owners
    result = {'meshes': tuple(meshes), 'vertex_ledgers': tuple(ids), 'owner_ledgers': tuple(ledgers),
        'identity_status': 1, 'n_elements': n, 'extra_smooth': False,
        'source_vid_encoding_version': 2, 'source_vid_encoding': 'ORIGINAL_EFFECTIVE_SOURCE_VID',
        'source_vid_shifts': np.zeros((n, 4), np.int32)}
    freeze(result)
    return result, document


def union_scene(n, elements):
    # Independent generation of each actual namespace, never adapter padding.
    seed, _, _, _, _ = fixture()
    meshes, records = [], []
    for element in range(n):
        if element not in elements:
            mesh = (np.empty((0, 3), float), np.empty((0, 3), np.int32), np.empty(0, np.int32))
        else:
            v, f, t = seed
            origin = np.asarray([10*element, 0, 0])
            mesh = (v.copy()+origin, f.copy(), t.copy())
            cycle = (0, 1, 2, 3)
            plan = {'element': element, 'boundary_actual_ids': cycle, 'removed_face_rows': (0, 1),
                'center': tuple([10*element+1., 1., 0.]), 'baseline_vertex_count': 8, 'baseline_face_count': 10,
                'new_center_id': 8, 'fan_faces': tuple((cycle[i], cycle[(i+1) % 4], 8) for i in range(4)),
                'consumed_owners': [(element, 0, 0, i, 0, 0, 0) for i in (0, 1)], 'source_digest': 'synthetic'}
            records.append({'event_id': f'event{element}', 'component_id': 'joint', 'plan': plan})
        for array in mesh: array.flags.writeable = False
        meshes.append(mesh)
    return tuple(meshes), records


class SceneGeometryTests(unittest.TestCase):
    def test_every_five_element_namespace_exact_plan_report_parity(self):
        for element in range(5):
            with self.subTest(element=element):
                s, doc = make_snapshot(element)
                a, b = old.spec_from_source(doc), m.spec_from_source(doc, 5)
                self.assertEqual(a.__dict__, b.__dict__)
                old_plan, old_report = old.audit_query(s, a, F(1), exact_contact_fallback=True)
                new_plan, new_report = m.audit_query(s, b, F(1), exact_contact_fallback=True)
                self.assertEqual(new_report['status'], 'PASS')
                self.assertEqual(new_plan, old_plan); self.assertEqual(stable(new_report), stable(old_report))

    def test_five_element_support_and_identity_receipt_parity(self):
        s, doc = make_snapshot(4)
        a, b = old.spec_from_source(doc), m.spec_from_source(doc, 5)
        before = old_footprints(s, [{'event_id': 'x', 'element': 4, 'spec': a}], F(1))
        after = m.actual_footprints(s, [{'event_id': 'x', 'element': 4, 'spec': b}], F(1))
        self.assertEqual(before, after); self.assertEqual(old_identity(s), m.original_identity_receipt(s))

    def test_five_element_union_full_report_and_output_parity(self):
        meshes, records = old_scene([('z', 'joint', 1, (0, 0, 0), 1, 0),
                                    ('a', 'joint', 1, (5, 0, 0), 1, 0), ('x', 'other', 4, (20, 0, 0), 1, 0)])
        a, ra = old_union.compile_union(meshes, records)
        b, rb = m.compile_union(meshes, records)
        self.assertEqual(ra, rb); self.assertEqual(mesh_receipt(a), mesh_receipt(b))
        self.assertEqual(old_union.check_pair_interactions(meshes, records), m.check_pair_interactions(meshes, records))
        for ma, mb in zip(a, b):
            for x, y in zip(ma, mb): np.testing.assert_array_equal(x, y)

    def test_five_element_policy_rejection_parity(self):
        s, doc = make_snapshot(2)
        add_external(s, 4, [[.5, .5, 0], [1.5, .5, 0], [1, 1.5, 0]])
        a = old.audit_query(s, old.spec_from_source(doc), F(1), exact_contact_fallback=True)
        b = m.audit_query(s, m.spec_from_source(doc, 5), F(1), exact_contact_fallback=True)
        self.assertEqual(a[0], b[0]); self.assertEqual(stable(a[1]), stable(b[1]))
        self.assertEqual(b[1]['status'], 'REJECT')

    def test_two_and_six_elements_keep_actual_owner_namespace(self):
        for n in (2, 6):
            with self.subTest(n=n):
                element = n-1; s, doc = snapshot(n, element); spec = m.spec_from_source(doc, n)
                plan, report = m.audit_query(s, spec, F(1), exact_contact_fallback=True)
                self.assertEqual(report['status'], 'PASS', report)
                self.assertEqual(plan['element'], element)
                self.assertTrue(all(o[0] == element for o in plan['consumed_owners']))
                self.assertEqual(report['retained_faces_checked'], 8)
                footprint = m.actual_footprints(s, [{'event_id': 'e', 'element': element, 'spec': spec}], F(1))[0]
                self.assertEqual(footprint['status'], 'COMPLETE_ACTUAL_REQUESTED_SUPPORT')
                self.assertEqual(footprint['element'], element)
                self.assertEqual(len(m.original_identity_receipt(s)['per_element_shifts']), n)

    def test_two_and_six_element_union_and_permutation(self):
        for n in (2, 6):
            meshes, records = union_scene(n, (0, n-1))
            a, ra = m.compile_union(meshes, records, expected_components={'joint': [r['event_id'] for r in records]})
            b, rb = m.compile_union(meshes, records[::-1], expected_components={'joint': [r['event_id'] for r in records]})
            self.assertEqual(ra['status'], 'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED', ra)
            self.assertEqual(len(a), n); self.assertEqual(ra, rb); self.assertEqual(mesh_receipt(a), mesh_receipt(b))
            self.assertEqual(ra['event_mapping'][f'event{n-1}']['element'], n-1)
            self.assertEqual(ra['expected_triangle_pairs'], 32)

    def test_complete_candidate_proxy_for_sixth_element(self):
        s, _ = snapshot(6, 5)
        owners = s['owner_ledgers'][5][:, :7].tolist()
        row = m.actual_footprints(s, [{'event_id': 'blocked', 'element': 5, 'spec': None,
            'candidate_actual_owners': owners}], F(1))[0]
        self.assertEqual(row['status'], 'COMPLETE_ACTUAL_REQUESTED_SUPPORT')
        self.assertEqual(row['kind'], 'BASELINE_BLOCKED_CURRENT_CANDIDATE_DOMAIN')
        self.assertFalse(row['future_replacement_bound'])
        self.assertEqual(row['source_face_rows'], list(range(10)))

    def test_cross_element_collision_not_missed_beyond_forest_five(self):
        s, doc = snapshot(6, 0)
        add_external(s, 5, [[.5, .5, 0], [1.5, .5, 0], [1, 1.5, 0]])
        plan, report = m.audit_query(s, m.spec_from_source(doc, 6), F(1), exact_contact_fallback=True)
        self.assertIsNone(plan); self.assertEqual(report['status'], 'REJECT', report)
        self.assertEqual(report['failing_element'], 5)

    def test_wrong_element_count_or_shift_shape_is_unknown(self):
        s, doc = snapshot(6, 5); spec = m.spec_from_source(doc, 6)
        s['n_elements'] = 5
        self.assertEqual(m.audit_query(s, spec, F(1))[1]['status'], 'UNKNOWN')
        s['n_elements'] = 6; s['source_vid_shifts'] = np.zeros((5, 4), np.int32); s['source_vid_shifts'].flags.writeable = False
        self.assertEqual(m.audit_query(s, spec, F(1))[1]['status'], 'UNKNOWN')
        with self.assertRaises(ValueError): m.original_identity_receipt(s)

    def test_source_owner_must_be_within_declared_actual_count(self):
        _, doc = snapshot(6, 5)
        with self.assertRaises(ValueError): m.spec_from_source(doc, 5)
        with self.assertRaises(ValueError): m.spec_from_source(doc, 0)

    def test_smooth_snapshot_cannot_inherit_raw_geometry_admission(self):
        s, doc = snapshot(); s['extra_smooth'] = True
        self.assertEqual(m.audit_query(s, m.spec_from_source(doc, 2), F(1))[1]['status'], 'UNKNOWN')

    def test_small_face_dtype_fresh_center_overflow_is_rejected(self):
        meshes, records = union_scene(2, (0,))
        v, f, tags = meshes[0]
        v = np.concatenate((v, np.zeros((128-len(v), 3))))
        tags = np.concatenate((tags, np.zeros(128-len(tags), dtype=tags.dtype)))
        f = f.astype(np.int8)
        for array in (v, f, tags): array.flags.writeable = False
        meshes = ((v, f, tags), meshes[1])
        plan = records[0]['plan']; plan['baseline_vertex_count'] = 128; plan['new_center_id'] = 128
        plan['fan_faces'] = tuple((i, (i+1) % 4, 128) for i in range(4))
        result, report = m.compile_union(meshes, records)
        self.assertIsNone(result); self.assertIn('dtype capacity', report['reason'])


if __name__ == '__main__':
    unittest.main()
