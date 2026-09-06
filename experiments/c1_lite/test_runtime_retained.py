#!/usr/bin/env python3
"""Synthetic raw-emission tests; no real cache, runtime, or output writes."""
from fractions import Fraction as F
import unittest

from window_source import HVID, Hypervertex, SourceVID, Record, Inventory
from runtime_retained import (RetainedStop, build_retained_unit, iter_retained_triangles,
    partition_points, required_splits, compact_manifest, audit_retained_unit)


def fixture(times=(2, 6), *, repeated=False, element=0, replicas=1):
    hv = {HVID(i, 0): Hypervertex((i, i*i, 0), 0, 0, 1) for i in range(3)}
    vids = tuple(SourceVID.canonical(key, key) for key in hv)
    polygon = (vids[0], vids[0], vids[2]) if repeated else vids
    records = tuple(Record(0, 0, i, element, times,
        tuple((polygon,) for _ in range(len(times)-1))) for i in range(replicas))
    return Inventory(records, hv, 4, 8, vids, 'synthetic', (), 0), vids


class RuntimeRetainedTests(unittest.TestCase):
    def test_raw_window_filters_but_includes_exact_endpoints(self):
        inv, _ = fixture()
        for t in (F(3, 2), F(13, 2)):
            unit = build_retained_unit(inv, t, t)
            self.assertEqual(unit.triangles, ())
            self.assertEqual(unit.ledger[0]['state'], 'skip')
        for t in (F(2), F(6)):
            self.assertEqual(len(build_retained_unit(inv, t, t).triangles), 1)

    def test_original_times_are_added_even_when_old_partition_omits_them(self):
        inv, _ = fixture((3, 5))
        points = dict(partition_points(inv, F(2), F(7, 2), F(6)))
        self.assertIn('raw_original_record_time', points[F(3)])
        self.assertIn('raw_original_record_time', points[F(5)])
        self.assertEqual([t for t, _ in required_splits(inv, F(5, 2), F(7, 2))], [F(3)])
        with self.assertRaisesRegex(RetainedStop, 'UNPARTITIONED_RAW_TIME_CELL'):
            list(iter_retained_triangles(inv, F(5, 2), F(7, 2)))
        unknown = audit_retained_unit(inv, F(5, 2), F(7, 2))
        self.assertEqual(unknown['status'], 'UNKNOWN')
        self.assertEqual(unknown['required_splits'], [{'numerator': 3, 'denominator': 1}])

    def test_branch_is_open_and_endpoint_is_separate(self):
        inv, _ = fixture((3, 5))
        open_cell = build_retained_unit(inv, F(5, 2), F(3))
        endpoint = build_retained_unit(inv, F(3), F(3))
        self.assertEqual(len(open_cell.triangles), 0)
        self.assertEqual(len(endpoint.triangles), 1)
        self.assertEqual(compact_manifest(open_cell)['time_domain'], 'open_cell_with_endpoint_limits')

    def test_replicated_faces_keep_all_owners_and_exact_suppression(self):
        inv, _ = fixture(replicas=2)
        unit = build_retained_unit(inv, F(3), F(3))
        self.assertEqual(len(unit.triangles), 1)
        owners = unit.triangles[0]['owners']
        self.assertEqual(len(owners), 2)
        with self.assertRaisesRegex(RetainedStop, 'PARTIAL_SUPPRESSION'):
            build_retained_unit(inv, F(3), F(3), owners[:1])
        suppressed = build_retained_unit(inv, F(3), F(3), owners)
        self.assertEqual(suppressed.triangles, ())
        self.assertTrue(all(r['state'] == 'suppress' for r in suppressed.ledger))

    def test_unemitted_suppression_is_rejected(self):
        inv, _ = fixture()
        with self.assertRaisesRegex(RetainedStop, 'SUPPRESSION_NOT_RAW_EMITTED'):
            build_retained_unit(inv, F(3, 2), F(3, 2), [(0, 0, 0, 0, 0, 0, 0)])

    def test_equal_coordinates_are_not_equal_identity(self):
        inv, vids = fixture()
        alias = HVID(9, 0)
        inv.hypervertices[alias] = inv.hypervertices[HVID(0, 0)]
        other = SourceVID.canonical(alias, alias)
        inv.sources += (other,)
        inv.records += (Record(0, 0, 9, 0, (2, 6), (((other, vids[1], vids[2]),),)),)
        self.assertEqual(len(build_retained_unit(inv, F(3), F(3)).triangles), 2)

    def test_repeated_vid_is_preserved_and_interface_witness_is_namespaced(self):
        inv, vids = fixture(repeated=True, replicas=2)
        unit = build_retained_unit(inv, F(3), F(3))
        report = compact_manifest(unit, boundary_cycle=[vids[2].text()], element=0)
        self.assertEqual(report['retained_source_face_count'], 1)
        self.assertEqual(report['repeated_effective_SourceVID_faces'], 1)
        self.assertEqual(report['interface_degenerate_source_faces'], 1)
        self.assertEqual(report['interface_gate'], 'REJECT_BASELINE_DEGENERACY_AT_INTERFACE')
        self.assertEqual(compact_manifest(unit, boundary_cycle=[vids[2].text()], element=1)
                         ['interface_degenerate_source_faces'], 0)

    def test_collinear_distinct_vids_are_degenerate_without_identity_collapse(self):
        inv, vids = fixture()
        inv.hypervertices = {key: Hypervertex((i, 0, 0), 0, 0, 1)
                             for i, key in enumerate(inv.hypervertices)}
        report = compact_manifest(build_retained_unit(inv, F(3), F(3)),
                                  boundary_cycle=[vids[0].text()], element=0)
        self.assertEqual(report['repeated_effective_SourceVID_faces'], 0)
        self.assertEqual(report['identically_zero_area_source_faces'], 1)

    def test_smooth_never_reuses_raw_vids(self):
        inv, _ = fixture()
        with self.assertRaisesRegex(RetainedStop, 'UNSUPPORTED_EXTRA_SMOOTH'):
            list(iter_retained_triangles(inv, F(3), F(3), mode='extra_smooth'))
        self.assertEqual(audit_retained_unit(inv, F(3), F(3), mode='extra_smooth')['status'], 'UNKNOWN')

    def test_every_serialized_raw_owner_has_one_ledger_state(self):
        inv, _ = fixture((2, 4, 6), replicas=2)
        unit = build_retained_unit(inv, F(3), F(3))
        report = compact_manifest(unit)
        self.assertEqual(report['raw_owner_count'], 4)
        self.assertEqual(report['owner_state_counts'], {'retained': 2, 'skip': 2})
        self.assertEqual(len({tuple(row['owner']) for row in unit.ledger}), 4)

    def test_singleton_selects_right_interval(self):
        inv, _ = fixture((2, 4, 6))
        unit = build_retained_unit(inv, F(4), F(4))
        self.assertEqual(unit.triangles[0]['owners'][0][4], 1)

    def test_witness_limit_does_not_reduce_denominator(self):
        inv, vids = fixture(repeated=True)
        report = compact_manifest(build_retained_unit(inv, F(3), F(3)),
                                  boundary_cycle=[vids[2].text()], element=0, max_witnesses=0)
        self.assertEqual(report['interface_degenerate_source_faces'], 1)
        self.assertEqual(report['interface_degenerate_witnesses'], [])


if __name__ == '__main__':
    unittest.main()
