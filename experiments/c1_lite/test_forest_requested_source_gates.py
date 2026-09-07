import unittest
from fractions import Fraction as F
from types import SimpleNamespace

from forest_requested_source_gates import audit_snapshot, support_for_query
from processed_mesh import HVID, SourceVID, RawTriangle, TriangleRef


def vertex(i):
    return SourceVID.canonical(HVID(i, 0), HVID(i, 0))


def triangle(face, index, event=False):
    return RawTriangle(TriangleRef(3, 0, 0, index, 0, 0, 0),
        tuple(vertex(i) for i in face), ((0., 0., 0.),)*3, (True,)*3, event)


def fixture():
    faces = [(0, 1, 2), (0, 2, 3), (1, 0, 4), (2, 1, 4), (3, 2, 4), (0, 3, 4)]
    raw = [triangle(face, i, i < 2) for i, face in enumerate(faces)]
    snapshot = SimpleNamespace(event_id='event', tau=F(3, 2), event_candidates_complete=True,
        halo_complete=True, actual_raw_triangles=raw,
        details_by_owner={t.reference.values(): {'native_legacy_identity_equal': True} for t in raw})
    support = {'owners': [list(t.reference.values()) for t in raw[:2]],
        'source_faces': [[v.text() for v in t.source_vertices] for t in raw[:2]],
        'boundary_cycle': [vertex(i).text() for i in range(4)]}
    return snapshot, support


class RequestedSourceTests(unittest.TestCase):
    def test_closed_interface_pass_is_not_native_admission(self):
        snap, support = fixture()
        result = audit_snapshot(snap, support, element=3)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['actual_native_array_equivalence'], 'NOT_TESTED')

    def test_retained_raw_replicas_are_one_quotient_face(self):
        snap, support = fixture()
        snap.actual_raw_triangles.append(triangle((1, 0, 4), 20))
        snap.details_by_owner[snap.actual_raw_triangles[-1].reference.values()] = {'native_legacy_identity_equal': True}
        self.assertEqual(audit_snapshot(snap, support, element=3)['status'], 'PASS')

    def test_selected_non_event_replica_must_be_consumed(self):
        snap, support = fixture()
        snap.actual_raw_triangles.append(triangle((0, 1, 2), 20))
        result = audit_snapshot(snap, support, element=3)
        self.assertEqual(result['reason_code'], 'PARTIAL_RAW_OWNER_CLASS_SUPPRESSION')
        self.assertTrue(result['certified_necessary_policy_failure'])

    def test_vertex_only_retained_link_defect_is_not_missed(self):
        snap, support = fixture()
        snap.actual_raw_triangles.append(triangle((0, 5, 6), 20))
        result = audit_snapshot(snap, support, element=3)
        self.assertEqual(result['status'], 'REJECT')
        self.assertIn('RETAINED_LINK_NOT_ONE_PATH', [w['kind'] for w in result['interface']['witnesses']])

    def test_missing_emission_rejects_fixed_owners(self):
        snap, support = fixture()
        snap.actual_raw_triangles = snap.actual_raw_triangles[1:]
        self.assertEqual(audit_snapshot(snap, support, element=3)['reason_code'], 'SOURCE_OWNER_NOT_RAW_EMITTED')

    def test_incomplete_halo_is_unknown(self):
        snap, support = fixture()
        snap.halo_complete = False
        self.assertEqual(audit_snapshot(snap, support, element=3)['status'], 'UNKNOWN')

    def test_native_identity_unknown_does_not_claim_actual(self):
        snap, support = fixture()
        snap.details_by_owner[snap.actual_raw_triangles[2].reference.values()]['native_legacy_identity_equal'] = False
        result = audit_snapshot(snap, support, element=3)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['native_ordered_identity_model'], 'UNKNOWN')

    def test_support_lookup_distinguishes_root_and_cells(self):
        fj = lambda x: {'numerator': F(x).numerator, 'denominator': F(x).denominator}
        source = {'breakpoint_points': [{'time': fj(F(3, 2))}],
                  'segments': [{'t0': fj(F(5, 4)), 't1': fj(F(3, 2))}]}
        self.assertEqual(support_for_query(source, F(11, 8))[1:], (F(11, 8), 'CONSTANT_INTEGER_PREDICATE_OPEN_CELL'))
        self.assertEqual(support_for_query(source, F(3, 2))[1:], (F(3, 2), 'EXACT_SINGLETON'))
        with self.assertRaises(ValueError):
            support_for_query(source, F(2))


if __name__ == '__main__':
    unittest.main()
