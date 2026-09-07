from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forest_source_compiler import (compile_event, select_existing, required_times,
    levels, compile_kernel)
from processed_mesh import HVID, Hypervertex, SourceVID, TriangleRef, RawTriangle


def fixture(element=0):
    xy = [(0, 0), (2, 0), (2, 2), (0, 2), (-1, -1), (3, -1), (3, 3), (-1, 3)]
    vertices = [SourceVID(HVID(i, 0), HVID(i+100, 0)) for i in range(8)]
    hvs = {}
    for i, p in enumerate(xy):
        hvs[HVID(i, 0)] = Hypervertex((*p, 0.), 0, 0, 1)
        hvs[HVID(i+100, 0)] = Hypervertex((*p, 0.), 4, 0, 1)
    faces = [(0, 1, 2), (0, 2, 3)]
    for i in range(4):
        j = (i+1) % 4
        faces.extend([(j, i, i+4), (j, i+4, j+4)])
    raw, details = [], {}
    for index, face in enumerate(faces):
        ref = TriangleRef(element, 0, 0, index, 0, 0, 0)
        sources = tuple(vertices[i] for i in face)
        raw.append(RawTriangle(ref, sources, tuple((*xy[i], 0.) for i in face), (True,)*3, index < 2))
        details[ref.values()] = {'actual_raw_emitted': True, 'native_legacy_identity_equal': True,
                                 'ordered_raw_vids': sources}
    snapshot = SimpleNamespace(raw_triangles=tuple(raw), details_by_owner=details,
        halo_vertex_ids=frozenset(vertices[:4]), halo_complete=True, event_candidates_complete=True)
    snapshots = {('synthetic', t): deepcopy(snapshot) for t in required_times(F(2), (0, 4, 0, 4))}
    return snapshots, hvs


def compile_fixture(snapshots=None, element=0):
    base, hvs = fixture(element)
    return compile_event('synthetic', F(2), (0, 4, 0, 4), snapshots or base,
        hvs, [1., 1., 0.], cache_digest='synthetic-only', group_count=2,
        maximum_discrete_time=4, element=element)


class CompilerTests(unittest.TestCase):
    def test_existing_selector_and_exact_contract(self):
        result = compile_fixture()
        self.assertEqual(result['status'], 'PASS_SOURCE_COMPILER', result)
        self.assertEqual([x['name'] for x in result['selector_attempts']], ['lower', 'critical', 'upper'])
        self.assertTrue(result['legacy_three_level_compatibility'])
        self.assertEqual(len(result['source']['segments']), 2)
        self.assertEqual(len(result['source']['breakpoint_points']), 3)
        self.assertFalse(result['runtime_admitted'])

    def test_actual_element_is_not_remapped_to_zero(self):
        result = compile_fixture(element=3)
        self.assertEqual(result['status'], 'PASS_SOURCE_COMPILER', result)
        self.assertEqual(result['source']['element'], 3)
        self.assertTrue(all(o[0] == 3 for s in result['source']['segments'] for o in s['owners']))

    def test_missing_halo_is_unknown_not_method_rejection(self):
        snapshots, _ = fixture()
        snapshots['synthetic', F(1)].halo_complete = False
        self.assertEqual(compile_fixture(snapshots)['status'], 'UNKNOWN')

    def test_missing_midcell_is_unknown(self):
        snapshots, _ = fixture()
        del snapshots['synthetic', F(3, 2)]
        self.assertEqual(compile_fixture(snapshots)['status'], 'UNKNOWN')

    def test_ghost_superset_owner_is_fixed_raw_gate_rejection(self):
        snapshots, _ = fixture()
        snapshots['synthetic', F(1)].details_by_owner[(0, 0, 0, 0, 0, 0, 0)]['actual_raw_emitted'] = False
        result = compile_fixture(snapshots)
        self.assertEqual(result['status'], 'REJECT_FIXED_SOURCE_COMPILER')
        self.assertEqual(result['rejected_stage'], 'raw_owner_emission')
        self.assertEqual(result['decisive_witness']['owner'], [0, 0, 0, 0, 0, 0, 0])

    def test_native_legacy_binding_difference_is_unknown(self):
        snapshots, _ = fixture()
        snapshots['synthetic', F(1)].details_by_owner[(0, 0, 0, 0, 0, 0, 0)]['native_legacy_identity_equal'] = False
        self.assertEqual(compile_fixture(snapshots)['status'], 'UNKNOWN')

    def test_complete_no_event_candidates_is_fixed_selector_rejection(self):
        snapshots, _ = fixture()
        empty = snapshots['synthetic', F(1)]
        empty.raw_triangles = tuple(RawTriangle(r.reference, r.source_vertices, r.positions, r.in_view, False) for r in empty.raw_triangles)
        empty.halo_vertex_ids = frozenset()
        result = compile_fixture(snapshots)
        self.assertEqual(result['status'], 'REJECT_FIXED_SOURCE_COMPILER')
        self.assertEqual(result['selector_attempts'][0]['complete_event_raw_candidates'], 0)

    def test_lower_payload_survives_later_selector_failure(self):
        snapshots, _ = fixture()
        target = snapshots['synthetic', F(2)]
        target.raw_triangles = tuple(RawTriangle(r.reference, r.source_vertices, r.positions, r.in_view, False) for r in target.raw_triangles)
        target.halo_vertex_ids = frozenset()
        result = compile_fixture(snapshots)
        self.assertEqual(result['status'], 'REJECT_FIXED_SOURCE_COMPILER')
        self.assertIn('lower', result['ordinary_patches'])
        self.assertEqual(result['selector_attempts'][-1]['name'], 'critical')

    def test_snapshot_input_order_does_not_change_legacy_traversal(self):
        snapshots, _ = fixture()
        expected = select_existing(snapshots, 'synthetic', F(1))[0]
        snapshots['synthetic', F(1)].raw_triangles = tuple(reversed(snapshots['synthetic', F(1)].raw_triangles))
        actual = select_existing(snapshots, 'synthetic', F(1))[0]
        self.assertEqual(actual['boundary_cycle'], expected['boundary_cycle'])
        self.assertEqual(actual['source_faces'], expected['source_faces'])

    def test_e2_partition_contains_real_integer_breakpoint(self):
        self.assertEqual(required_times(F(104, 5), (22, 16, 24, 20)),
            [F(102, 5), F(103, 5), F(104, 5), F(209, 10), F(21), F(211, 10), F(106, 5)])

    def test_unverified_noninteger_group_threshold_domain_is_unknown(self):
        snapshots, hvs = fixture()
        result = compile_event('synthetic', F(2), (0, 4, 0, 4), snapshots, hvs,
            [1., 1., 0.], cache_digest='synthetic', group_count=3, maximum_discrete_time=4)
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_empty_registry_does_not_forge_kernel_pass(self):
        self.assertEqual(compile_kernel([], {})['status'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
