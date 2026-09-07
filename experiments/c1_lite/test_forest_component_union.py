"""Synthetic canonical-union tests; no native slicing or scene campaign."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forest_component_union import compile_union, check_pair_interactions
from forest_native_campaign import mesh_receipt
from test_window_runtime import fixture


def scene(configurations):
    """(event,component,element,origin,scale,center_z) synthetic source disks."""
    seed, _, _, _, _ = fixture()
    by_element = {i: [] for i in range(5)}
    for config in configurations:
        by_element[config[2]].append(config)
    meshes, records = [], []
    for element in range(5):
        blocks = by_element[element]
        vertices, faces, tags = [], [], []
        for ordinal, (eid, cid, _, origin, scale, center_z) in enumerate(blocks):
            v, f, t = seed
            offset = 8*ordinal
            vertices.append(v*scale+np.asarray(origin))
            faces.append(f+offset)
            tags.append(t)
            cycle = tuple(offset+i for i in range(4))
            removed = (10*ordinal, 10*ordinal+1)
            total_v, total_f = 8*len(blocks), 10*len(blocks)
            center = [origin[0]+scale, origin[1]+scale, center_z]
            plan = {'element': element, 'boundary_actual_ids': cycle, 'removed_face_rows': removed,
                    'center': center, 'new_center_id': total_v, 'baseline_vertex_count': total_v,
                    'baseline_face_count': total_f,
                    'fan_faces': [(cycle[i], cycle[(i+1) % 4], total_v) for i in range(4)],
                    'consumed_owners': [(element, 0, 0, 10*ordinal+i, 0, 0, 0) for i in (0, 1)],
                    'source_digest': 'synthetic-'+eid}
            records.append({'event_id': eid, 'component_id': cid, 'plan': plan})
        if blocks:
            mesh = (np.concatenate(vertices), np.concatenate(faces), np.concatenate(tags))
        else:
            mesh = (np.empty((0, 3), float), np.empty((0, 3), np.int32), np.empty(0, np.int32))
        for a in mesh:
            a.flags.writeable = False
        meshes.append(mesh)
    return tuple(meshes), records


class ComponentUnionTests(unittest.TestCase):
    def test_empty_union_preserves_all_baseline_objects(self):
        meshes, _ = scene([('a', 'c', 0, (0, 0, 0), 1, 0)])
        output, report = compile_union(meshes, [], expected_components={})
        self.assertEqual(report['status'], 'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED', report)
        self.assertTrue(all(a is b for a, b in zip(meshes, output)))
        self.assertEqual(report['event_mapping'], {})
        self.assertFalse(report['whole_schedule_admitted'])

    def test_input_permutation_has_identical_arrays_and_canonical_ids(self):
        meshes, records = scene([('z', 'c', 1, (0, 0, 0), 1, 0),
                                 ('a', 'c', 1, (5, 0, 0), 1, 0)])
        first, a = compile_union(meshes, records, expected_components={'c': ['z', 'a']})
        second, b = compile_union(meshes, list(reversed(records)), expected_components={'c': ['a', 'z']})
        self.assertEqual(a['status'], 'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED', a)
        self.assertEqual(mesh_receipt(first), mesh_receipt(second))
        self.assertEqual(a['event_mapping'], b['event_mapping'])
        self.assertEqual(a['canonical_event_order'], ['a', 'z'])
        self.assertEqual(a['event_mapping']['a']['new_center_id'], 16)
        self.assertEqual(a['event_mapping']['z']['new_center_id'], 17)
        self.assertEqual(a['source_faces_consumed_once'], 4)
        self.assertEqual(a['raw_owners_consumed_once'], 4)
        self.assertTrue(a['component_membership_verified'])
        self.assertEqual(a['expected_triangle_pairs'], 32)
        self.assertEqual(a['counts']['triangle_pairs_excluded_by_support_aabb']+
                         a['counts']['exact_triangle_pair_calls'], 32)

    def test_multiple_elements_rebase_centers_with_element_namespace(self):
        meshes, records = scene([('a', 'ca', 0, (0, 0, 0), 1, 0),
                                 ('b', 'cb', 4, (10, 0, 0), 1, 0)])
        output, report = compile_union(meshes, records)
        self.assertEqual(report['status'], 'PASS_UNION_CONSTRUCTED_NOT_SCHEDULE_CERTIFIED', report)
        self.assertEqual(report['event_mapping']['a']['new_center_id'], 8)
        self.assertEqual(report['event_mapping']['b']['new_center_id'], 8)
        self.assertEqual(report['event_mapping']['b']['element'], 4)
        self.assertIs(output[2], meshes[2])
        self.assertFalse(report['component_membership_verified'])

    def test_shared_source_faces_and_boundary_are_unsupported(self):
        meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 0)])
        duplicate = deepcopy(records[0])
        duplicate['event_id'] = 'b'
        output, report = compile_union(meshes, [*records, duplicate])
        self.assertIsNone(output)
        self.assertEqual(report['status'], 'UNSUPPORTED_SHARED_SUPPORT')
        self.assertIn('boundary identity', report['reason'])
        self.assertFalse(report['partial_output_published'])

    def test_raw_owner_cannot_be_consumed_by_two_disjoint_patches(self):
        meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 0),
                                 ('b', 'c', 0, (5, 0, 0), 1, 0)])
        records[1]['plan']['consumed_owners'][0] = records[0]['plan']['consumed_owners'][0]
        output, report = compile_union(meshes, records)
        self.assertIsNone(output)
        self.assertEqual(report['status'], 'UNSUPPORTED_SHARED_SUPPORT')
        self.assertIn('raw owner', report['reason'])

    def test_missing_component_member_never_returns_partial_output(self):
        meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 0)])
        output, report = compile_union(meshes, records, expected_components={'c': ['a', 'b']})
        self.assertIsNone(output)
        self.assertEqual(report['status'], 'UNKNOWN_INPUT_OR_PROOF')
        self.assertIn('component membership', report['reason'])

    def test_cross_element_same_numeric_ids_never_count_as_shared(self):
        meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 0),
                                 ('b', 'c', 1, (0, 0, 0), 1, 0)])
        report = check_pair_interactions(meshes, records)
        self.assertEqual(report['status'], 'REJECT_POLICY_CONTACT', report)
        self.assertEqual(report['failing_pair']['exact_contact']['shared_ids'], [])

    def test_two_fan_intersection_has_exact_witness(self):
        meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 3),
                                 ('b', 'c', 1, (0, 0, 2), 1, 0.5)])
        output, report = compile_union(meshes, records)
        self.assertIsNone(output)
        self.assertEqual(report['status'], 'REJECT_POLICY_CONTACT', report)
        self.assertEqual(report['failing_pair']['relation'], 'Q_Q')
        self.assertTrue(report['failing_pair']['exact_contact']['witness'])
        self.assertFalse(report['new_contact_relative_to_baseline_proven'])

    def test_fan_other_removed_contact_is_checked_after_disjoint_fans(self):
        meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 0),
                                 ('b', 'c', 1, (-2, -2, 0), 3, 2)])
        report = check_pair_interactions(meshes, records)
        self.assertEqual(report['status'], 'REJECT_POLICY_CONTACT', report)
        self.assertEqual(report['failing_pair']['relation'], 'Q_A_OTHER_REMOVED_B')
        self.assertEqual(report['counts']['Q_Q_checked'], 16)

    def test_forced_proof_error_does_not_mutate_or_publish_baseline(self):
        meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 0.5),
                                 ('b', 'c', 1, (0, 0, 0.25), 1, 0.75)])
        before = mesh_receipt(meshes)
        with patch('forest_component_union.check_triangle_contact', side_effect=ValueError('forced proof failure')):
            output, report = compile_union(meshes, records)
        self.assertIsNone(output)
        self.assertEqual(report['status'], 'UNKNOWN_INPUT_OR_PROOF')
        self.assertEqual(mesh_receipt(meshes), before)
        self.assertFalse(report['partial_output_published'])

    def test_output_mapping_binds_all_four_sector_rows(self):
        meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 0),
                                 ('b', 'c', 0, (5, 0, 0), 1, 0)])
        before = mesh_receipt(meshes)
        output, report = compile_union(meshes, records)
        self.assertEqual(mesh_receipt(meshes), before)
        for mapping in report['event_mapping'].values():
            faces = output[mapping['element']][1]
            self.assertEqual(faces[mapping['fan_face_rows_by_sector']].tolist(), mapping['fan_actual_vertex_ids'])
        for element in range(5):
            v, f, tags = meshes[element]
            self.assertEqual(output[element][0][:len(v)].tobytes(), v.tobytes())
            self.assertEqual(output[element][2][:len(tags)].tobytes(), tags.tobytes())
        self.assertTrue(all(not a.flags.writeable for mesh in output for a in mesh))

    def test_bad_single_event_center_or_fan_never_constructed(self):
        for field, value in (('center', [0, 1, 0]), ('fan_faces', [(0, 1, 8)]*4)):
            meshes, records = scene([('a', 'c', 0, (0, 0, 0), 1, 0)])
            records[0]['plan'][field] = value
            output, report = compile_union(meshes, records)
            self.assertIsNone(output)
            self.assertEqual(report['status'], 'UNKNOWN_INPUT_OR_PROOF')


if __name__ == '__main__':
    unittest.main()
